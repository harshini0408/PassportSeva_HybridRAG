"""API checks with a mocked provider: no Groq key, network or quota is used."""
from __future__ import annotations

import json
from dataclasses import replace

import groq
import httpx
import pytest
from fastapi.testclient import TestClient

from app.llm import GroqLLM, LLMError, Passage, parse_reply
from app.main import create_app
from app.rag.config import load_settings
from app.rag.index import IndexError_
from app.rag.retrieve import make_retriever
from app.service import AskService, validate_evidence_role
from app.settings import AppSettings

SECRET = "gsk_test_secret_value_123"


class FakeLLM:
    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.calls, self.last_passages = reply, error, 0, []

    def generate(self, question, passages):
        self.calls += 1
        self.last_passages = passages
        if self.error:
            raise self.error
        return self.reply(passages) if callable(self.reply) else self.reply


def cite_first(passages: list[Passage]) -> str:
    p = passages[0]
    return json.dumps({"sufficient": True, "answer": f"Do the documented steps. [{p.label}]",
                       "citations": [p.label]})


def cite_all(passages: list[Passage]) -> str:
    labels = [p.label for p in passages]
    return json.dumps({"sufficient": True, "answer": "Steps: " + "".join(f"[{l}]" for l in labels),
                       "citations": labels})


@pytest.fixture(scope="session")
def retriever():
    s = load_settings()
    if not (s.index_dir / "index_meta.json").is_file():
        pytest.skip("index not built; run: python -m app.rag.cli build")
    return make_retriever(s)


def make_client(retriever, llm, **kw):
    settings = AppSettings(groq_api_key=SECRET, **kw)
    svc = AskService(settings, llm, retriever_factory=lambda _s: retriever)
    return TestClient(create_app(service=svc, settings=settings, warm=False)), svc


def test_cited_answer_maps_ids_to_backend_metadata(retriever):
    llm = FakeLLM(cite_all)
    client, _ = make_client(retriever, llm)
    r = client.post("/api/ask", json={"question": "How do I reschedule my appointment?"})
    body = r.json()
    assert r.status_code == 200 and body["status"] == "answered" and body["grounded"] is True
    assert body["mode"] == "hybrid" and body["llm_used"] is True
    assert body["citations"] and all(c["url"].startswith("https://") for c in body["citations"])
    assert all(c["evidence_role"] == "answer_evidence" for c in body["citations"])
    pdf = [c for c in body["citations"] if c["pdf_pages"]]
    assert pdf and pdf[0]["pdf_pages"] == [5, 6] and pdf[0]["url"].endswith("#page=5")
    # no routing-only or disputed-limit passage was sent to the model
    assert all("twice within one year" not in p.text for p in llm.last_passages)


@pytest.mark.parametrize("reply", [
    json.dumps({"sufficient": True, "answer": "Go. [S9]", "citations": ["S9"]}),
    json.dumps({"sufficient": True, "answer": "Go. [S1]", "citations": ["S1", "S42"]}),
    json.dumps({"sufficient": True, "answer": "Go. [S7]", "citations": ["S1"]}),
    json.dumps({"sufficient": True, "answer": "Uncited claim.", "citations": []}),
    json.dumps({"sufficient": False, "answer": "", "citations": []}),
    "not json at all",
])
def test_unusable_citations_are_not_presented_as_grounded(retriever, reply):
    client, _ = make_client(retriever, FakeLLM(reply))
    body = client.post("/api/ask", json={"question": "How do I book my visit after submitting the form?"}).json()
    assert body["status"] == "insufficient_evidence" and body["grounded"] is False
    assert body["citations"] == []


def test_numeric_reschedule_limit_in_answer_is_rejected(retriever):
    reply = json.dumps({"sufficient": True, "answer": "You can reschedule twice. [S1]", "citations": ["S1"]})
    client, _ = make_client(retriever, FakeLLM(reply))
    body = client.post("/api/ask", json={"question": "How do I reschedule my appointment?"}).json()
    assert body["status"] == "insufficient_evidence" and not body["grounded"]


@pytest.mark.parametrize("payload", [
    {"question": ""}, {"question": "   \n\t "}, {"question": "x" * 501},
    {"question": "ok", "mode": "fuzzy"}, {}, {"question": 5},
])
def test_invalid_requests_rejected(retriever, payload):
    llm = FakeLLM(cite_first)
    client, _ = make_client(retriever, llm)
    assert client.post("/api/ask", json=payload).status_code == 422
    assert llm.calls == 0


def test_configured_length_limit(retriever):
    client, _ = make_client(retriever, FakeLLM(cite_first), max_question_chars=40)
    assert client.post("/api/ask", json={"question": "a" * 41}).status_code == 422


def test_missing_index(tmp_path):
    def broken(_s):
        raise IndexError_(f"no index at {tmp_path}; run the build command first")
    settings = AppSettings(groq_api_key=SECRET)
    llm = FakeLLM(cite_first)
    app = create_app(service=AskService(settings, llm, retriever_factory=broken), settings=settings, warm=False)
    with TestClient(app) as client:          # runs the startup load, as uvicorn does
        h = client.get("/api/health").json()
        assert h["status"] == "degraded" and h["index"]["ready"] is False and "no index" in h["index"]["detail"]
        r = client.post("/api/ask", json={"question": "How do I apply?"})
        assert r.status_code == 503 and r.json()["error"]["code"] == "index_unavailable"
    assert llm.calls == 0


def test_health_ready_never_calls_groq_or_leaks_key(retriever):
    llm = FakeLLM(cite_first)
    client, svc = make_client(retriever, llm)
    svc.load()
    r = client.get("/api/health")
    assert r.json()["status"] == "ok" and r.json()["index"]["chunk_count"] == 18
    assert r.json()["groq"] == {"configured": True, "model": "openai/gpt-oss-20b"}
    assert llm.calls == 0 and SECRET not in r.text


def test_health_reports_groq_unconfigured(retriever):
    settings = AppSettings(groq_api_key="")
    svc = AskService(settings, FakeLLM(), retriever_factory=lambda _s: retriever)
    svc.load()
    h = TestClient(create_app(service=svc, settings=settings, warm=False)).get("/api/health").json()
    assert h["groq"]["configured"] is False and h["status"] == "degraded"


@pytest.mark.parametrize("error,status,code", [
    (LLMError("groq_rate_limited", "The answer generator is rate limited. Try again shortly.", 429), 429, "groq_rate_limited"),
    (LLMError("groq_timeout", "The answer generator timed out. Try again.", 504), 504, "groq_timeout"),
    (LLMError("groq_unavailable", "The answer generator is unavailable right now.", 502), 502, "groq_unavailable"),
    (LLMError("groq_not_configured", "not configured", 503), 503, "groq_not_configured"),
])
def test_provider_failures_are_safe_and_distinct(retriever, error, status, code):
    client, _ = make_client(retriever, FakeLLM(error=error))
    r = client.post("/api/ask", json={"question": "How do I book my visit after submitting the form?"})
    assert r.status_code == status and r.json()["status"] == "error"
    assert r.json()["error"]["code"] == code and r.json()["grounded"] is False and SECRET not in r.text


def test_unconfigured_groq_makes_no_network_call(retriever):
    settings = AppSettings(groq_api_key="")
    svc = AskService(settings, GroqLLM(settings), retriever_factory=lambda _s: retriever)
    r = TestClient(create_app(service=svc, settings=settings, warm=False)).post(
        "/api/ask", json={"question": "How do I book my visit after submitting the form?"})
    assert r.status_code == 503 and r.json()["error"]["code"] == "groq_not_configured"


class _Boom:
    def __init__(self, exc):
        self.chat = self
        self.completions = self
        self.exc = exc

    def create(self, **_):
        raise self.exc


def _groq_error(cls, status):
    req = httpx.Request("POST", "https://api.groq.com/x")
    return cls("provider said: SECRET-PAYLOAD", response=httpx.Response(status, request=req), body={"k": "SECRET-PAYLOAD"})


@pytest.mark.parametrize("exc,code", [
    (_groq_error(groq.RateLimitError, 429), "groq_rate_limited"),
    (_groq_error(groq.AuthenticationError, 401), "groq_auth_rejected"),
    (_groq_error(groq.InternalServerError, 500), "groq_unavailable"),
    (groq.APITimeoutError(request=httpx.Request("POST", "https://api.groq.com/x")), "groq_timeout"),
    (groq.APIConnectionError(request=httpx.Request("POST", "https://api.groq.com/x")), "groq_unavailable"),
])
def test_groq_sdk_errors_map_to_safe_codes(exc, code):
    llm = GroqLLM(AppSettings(groq_api_key=SECRET))
    llm._client = _Boom(exc)
    with pytest.raises(LLMError) as ei:
        llm.generate("q", [Passage("S1", "t", "T", "S")])
    assert ei.value.code == code and "SECRET-PAYLOAD" not in ei.value.message


@pytest.mark.parametrize("question,status,link_part", [
    ("What exact documents do I need?", "clarification", "docAdvisor"),
    ("Give me the complete checklist for a Normal adult reissue.", "official_redirect", "reissuePassport"),
    ("Give me the complete checklist for a fresh passport.", "official_redirect", "attachmentAdvFreshInp"),
    ("Which official link should I use to apply for a Reissue ordinary passport?", "official_redirect", "ApplyReissue"),
    ("Which link should I use?", "clarification", "passportindia.gov.in"),
    ("Has my passport been dispatched today?", "live_status_redirect", "statusTracker"),
    ("How many times can I reschedule?", "conflicting_sources", "passportindia.gov.in"),
    ("How do I apply for Tatkaal for a child?", "out_of_scope", None),
])
def test_policy_decisions_bypass_groq(retriever, question, status, link_part):
    llm = FakeLLM(cite_first)
    client, _ = make_client(retriever, llm)
    r = client.post("/api/ask", json={"question": question})
    body = r.json()
    assert r.status_code == 200 and body["status"] == status and llm.calls == 0
    assert body["llm_used"] is False and body["grounded"] is False
    assert "ask whether" not in body["answer"].lower()
    assert "do not answer with a numeric limit" not in body["answer"].lower()
    if link_part:
        assert any(link_part in l["url"] for l in body["official_links"])
    if status == "clarification":
        assert "Fresh" in body["clarification_question"] and "Reissue" in body["clarification_question"]
    if status == "conflicting_sources":
        assert not any(w in body["answer"].lower() for w in ("twice", "three times", "2 times", "3 times"))
        assert len(body["citations"]) == 3


def test_unexpected_evidence_role_rejected():
    assert validate_evidence_role("answer_evidence") == "answer_evidence"
    assert validate_evidence_role("routing_only") == "routing_only"
    with pytest.raises(ValueError, match="unexpected evidence_role"):
        validate_evidence_role("checklist")


def test_modes_are_selectable(retriever):
    for mode in ("semantic", "hybrid", "rerank"):
        client, _ = make_client(retriever, FakeLLM(cite_first))
        r = client.post("/api/ask", json={"question": "What does ARN stand for?", "mode": mode})
        assert r.status_code == 200 and r.json()["mode"] == mode and r.json()["status"] == "answered"


def test_openapi_documents_response_shape(retriever):
    client, _ = make_client(retriever, FakeLLM())
    spec = client.get("/openapi.json").json()
    assert {"AskRequest", "AskResponse", "Citation", "HealthResponse"} <= set(spec["components"]["schemas"])
    assert {"422", "429", "502", "503", "504"} <= set(spec["paths"]["/api/ask"]["post"]["responses"])


def test_parse_reply_variants():
    assert parse_reply('```json\n{"sufficient": true, "answer": "a [S1]", "citations": ["s1"]}\n```').citations == ["S1"]
    assert parse_reply("nope") is None

