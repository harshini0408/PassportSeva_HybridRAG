"""Orchestration: routing policy -> retrieval -> Groq -> verified citations."""
from __future__ import annotations

import re
import threading
from typing import Callable, cast

from .llm import AnswerLLM, LLMError, Passage, parse_reply
from .rag.config import Settings as RagSettings
from .rag.config import load_settings as load_rag_settings
from .rag.index import IndexError_
from .rag.ingest import Chunk
from .rag.policy import Decision, classify
from .rag.retrieve import Retriever, make_retriever
from .schemas import (
    AskResponse,
    Citation,
    ErrorCode,
    ErrorInfo,
    EvidenceRole,
    GroqHealth,
    HealthResponse,
    IndexHealth,
    Mode,
    OfficialLink,
)
from .settings import AppSettings

REISSUE_HINT = re.compile(r"re-?issue|renew\w*|expir\w*|already have|previously held|old passport", re.I)
FRESH_HINT = re.compile(r"\b(fresh|first(-|\s+)time|first (ordinary )?passport|never held|new passport)\b", re.I)
APPT_HINT = re.compile(r"\b(appointment|book\w*|reschedul\w*|slot|psk|popsk)\b", re.I)

CLARIFY_Q = (
    "Is this for a Fresh application (you have never held an ordinary passport) or a "
    "Reissue (you previously held one)? If Reissue, is the passport close to expiry or already expired?"
)
LINK_CLARIFY_Q = (
    "Which official link do you need — starting a Fresh or Reissue application, "
    "checking the Document Advisor, or booking/rescheduling an appointment?"
)
INSUFFICIENT = (
    "The indexed official sources don't contain enough to answer this reliably. "
    "Please check the official Passport Seva portal."
)
# A numeric reschedule limit must never be stated (sources disagree).
LIMIT_CLAIM = re.compile(
    r"\b(once|twice|thrice|two|three|2|3)\b[^.\n]{0,30}\b(times|re-?schedul\w*|attempts?)\b|"
    r"\b(re-?schedul\w*|attempts?)\b[^.\n]{0,30}\b(once|twice|thrice|two|three|2|3)\b",
    re.I,
)
MARKER = re.compile(r"\[S(\d+)\]", re.I)


def validate_evidence_role(raw_role: str) -> EvidenceRole:
    """Reject unexpected evidence_role values so they never silently become answer_evidence."""
    if raw_role == "answer_evidence":
        return "answer_evidence"
    if raw_role == "routing_only":
        return "routing_only"
    raise ValueError(f"unexpected evidence_role: {raw_role!r}")


def _citation(label: str, chunk: Chunk) -> Citation:
    url = chunk.source_url + (f"#page={chunk.pdf_pages[0]}" if chunk.pdf_pages else "")
    role = validate_evidence_role(chunk.evidence_role)
    return Citation(
        id=label,
        title=chunk.title,
        section=chunk.section,
        url=url,
        pdf_pages=chunk.pdf_pages,
        source_id=chunk.source_id,
        reviewed_on=chunk.reviewed_on,
        evidence_role=role,
    )


class AskService:
    """Holds the retriever (loaded once, reloaded only if the index was missing) and the LLM."""

    def __init__(
        self,
        settings: AppSettings,
        llm: AnswerLLM,
        retriever_factory: Callable[[RagSettings], Retriever] = make_retriever,
        rag_settings: RagSettings | None = None,
    ):
        self.s = settings
        self.llm = llm
        self.rag_settings = rag_settings or load_rag_settings()
        self._factory = retriever_factory
        self._retriever: Retriever | None = None
        self._index_error: str | None = None
        self._load_lock = threading.Lock()
        self._model_lock = threading.Lock()  # SentenceTransformer/FAISS calls are serialised

    # ------------------------------------------------------------ resources
    def load(self, warm: bool = False) -> bool:
        """Load the retriever once. Safe to call repeatedly; blocking, so run off the event loop."""
        with self._load_lock:
            if self._retriever is not None:
                return True
            try:
                r = self._factory(self.rag_settings)
                if warm:  # load the embedder now (hybrid needs it), not on the first request
                    r.semantic("warm up", 1)
                    if self.s.warm_reranker:  # optional: hybrid never uses the cross-encoder
                        r.reranker.predict([("warm up", "warm up")])
                self._retriever, self._index_error = r, None
                return True
            except IndexError_ as exc:
                self._index_error = str(exc)
            except Exception as exc:  # noqa: BLE001
                self._index_error = f"retrieval resources failed to load ({type(exc).__name__})"
            return False

    def health(self) -> HealthResponse:
        if self._retriever is not None:
            m = self._retriever.idx.meta
            index = IndexHealth(
                ready=True,
                chunk_count=m["chunk_count"],
                embedding_model=m["embedding_model"],
                similarity_metric=m["similarity_metric"],
            )
        else:
            index = IndexHealth(ready=False, detail=self._index_error or "index not loaded yet")
        ok = index.ready and self.s.groq_configured
        return HealthResponse(
            status="ok" if ok else "degraded",
            index=index,
            groq=GroqHealth(configured=self.s.groq_configured, model=self.s.groq_model),
        )

    # ----------------------------------------------------------------- ask
    def ask(self, question: str, mode: Mode) -> tuple[AskResponse, int]:
        if self._retriever is None:
            self.load()
        r = self._retriever
        if r is None:
            return self._error(
                mode,
                "index_unavailable",
                "The retrieval index is not available. Build it with: python -m app.rag.cli build",
                503,
            ), 503
        decision = classify(question, r.manifest_conflict_handling)
        b = decision.behavior
        if b != "answer":
            return self._policy_response(r, question, mode, decision), 200
        return self._grounded(r, question, mode)

    def _link(self, r: Retriever, source_id: str, label: str) -> OfficialLink | None:
        s = r.source_link(source_id)
        return OfficialLink(label=label, url=s["url"]) if s else None

    def _policy_response(self, r: Retriever, question: str, mode: Mode, d: Decision) -> AskResponse:
        if d.behavior == "clarification":
            links = [
                l
                for l in (
                    self._link(r, "fresh_advisor", "Fresh Passport Document Advisor"),
                    self._link(r, "reissue_advisor", "Reissue Document Advisor"),
                )
                if l
            ]
            return AskResponse(
                status="clarification",
                answer=(
                    "Document requirements depend on whether you are applying for a Fresh passport or a Reissue. "
                    "Please choose or type your application type below, or open the matching official Document Advisor."
                ),
                clarification_question=CLARIFY_Q,
                official_links=links,
                mode=mode,
                grounded=False,
                llm_used=False,
            )
        if d.behavior == "official_link":
            if REISSUE_HINT.search(question):
                links = [
                    l
                    for l in (
                        self._link(r, "reissue", "Apply For Re-Issue of Ordinary Passport (Official Guide)"),
                        self._link(r, "apply", "Process to apply for Fresh / Reissue Ordinary Passport (Portal)"),
                    )
                    if l
                ]
                return AskResponse(
                    status="official_redirect",
                    official_links=links,
                    answer=(
                        "To start or review a routine Reissue application, use the official Re-Issue of Ordinary "
                        "Passport guide or sign in on the main Passport Seva application portal below."
                    ),
                    mode=mode,
                    grounded=False,
                    llm_used=False,
                )
            if FRESH_HINT.search(question):
                links = [
                    l
                    for l in (
                        self._link(r, "apply", "Process to apply for Fresh / Reissue Ordinary Passport (Official Guide)"),
                        self._link(r, "fresh_advisor", "Fresh Passport Document Advisor"),
                    )
                    if l
                ]
                return AskResponse(
                    status="official_redirect",
                    official_links=links,
                    answer=(
                        "To start a Fresh ordinary passport application, use the official Passport Seva application "
                        "guide below."
                    ),
                    mode=mode,
                    grounded=False,
                    llm_used=False,
                )
            if APPT_HINT.search(question):
                links = [
                    l
                    for l in (
                        self._link(r, "apply", "Process to apply for Fresh / Reissue Ordinary Passport (Appointments)"),
                        self._link(r, "appointment", "New Process for Online Appointment Booking (PDF Guide)"),
                    )
                    if l
                ]
                return AskResponse(
                    status="official_redirect",
                    official_links=links,
                    answer=(
                        "For booking or rescheduling a Passport Seva appointment, use the official application "
                        "portal guide or the appointment booking PDF walkthrough below."
                    ),
                    mode=mode,
                    grounded=False,
                    llm_used=False,
                )
            links = [
                l
                for l in (
                    self._link(r, "apply", "Process to apply for Fresh / Reissue Ordinary Passport"),
                    self._link(r, "reissue", "Apply For Re-Issue of Ordinary Passport"),
                )
                if l
            ]
            return AskResponse(
                status="clarification",
                answer=(
                    "Several official Passport Seva pages are available depending on whether you are starting an "
                    "application, checking required documents, or managing an appointment."
                ),
                clarification_question=LINK_CLARIFY_Q,
                official_links=links,
                mode=mode,
                grounded=False,
                llm_used=False,
            )
        if d.behavior == "official_redirect":
            reissue = bool(REISSUE_HINT.search(question))
            links: list[OfficialLink] = []
            if reissue:
                for sid, lbl in (
                    ("reissue_advisor", "Reissue Document Advisor"),
                    ("reissue_reason", "Reissue Case Selector (approaching expiry vs. expired)"),
                ):
                    lnk = self._link(r, sid, lbl)
                    if lnk:
                        links.append(lnk)
                return AskResponse(
                    status="official_redirect",
                    official_links=links,
                    answer=(
                        "Because the exact document checklist depends on your reissue reason (such as validity "
                        "approaching expiry or already expired, change of address, and Non-ECR status), use the "
                        "official Reissue Document Advisor below and select the reason matching your situation."
                    ),
                    mode=mode,
                    grounded=False,
                    llm_used=False,
                )
            lnk = self._link(r, "fresh_advisor", "Fresh Passport Document Advisor")
            if lnk:
                links.append(lnk)
            return AskResponse(
                status="official_redirect",
                official_links=links,
                answer=(
                    "I can't give a verified exact document checklist. Use the official Fresh Passport Document "
                    "Advisor and select your details (such as Normal and Adult) to see the current list for your case."
                ),
                mode=mode,
                grounded=False,
                llm_used=False,
            )
        if d.behavior == "live_status_redirect":
            return AskResponse(
                status="live_status_redirect",
                answer=(
                    "Live application or dispatch status isn't available here. Use the official status "
                    "tracker. This guide never asks for passport numbers or application reference numbers."
                ),
                official_links=[OfficialLink(label="Passport Seva application status tracker", url=self.s.tracker_url)],
                mode=mode,
                grounded=False,
                llm_used=False,
            )
        if d.behavior == "conflicting_sources":
            srcs = r.conflict_sources()
            cites = [
                Citation(
                    id=f"C{i}",
                    title=s["title"],
                    section="Rescheduling limit (sources disagree)",
                    url=s["url"],
                    source_id=s["source_id"],
                    reviewed_on=self._reviewed(r, s["source_id"]),
                    evidence_role="answer_evidence",
                    pdf_pages=None,
                )
                for i, s in enumerate(srcs, 1)
            ]
            return AskResponse(
                status="conflicting_sources",
                citations=cites,
                official_links=[OfficialLink(label=s["title"], url=s["url"]) for s in srcs],
                answer=(
                    "Official Passport Seva sources disagree on the number of reschedules allowed under the Normal "
                    "process, so I can't give a single definitive limit. Please check the remaining reschedule "
                    "attempts shown in your Passport Seva account when confirming a change. To see the procedure, "
                    "ask me how to change an appointment."
                ),
                mode=mode,
                grounded=False,
                llm_used=False,
            )
        return AskResponse(
            status="out_of_scope",
            answer=d.note or "This guide covers adults applying in India for an ordinary passport under the Normal process.",
            mode=mode,
            grounded=False,
            llm_used=False,
        )

    @staticmethod
    def _reviewed(r: Retriever, source_id: str) -> str:
        return next((c.reviewed_on for c in r.idx.chunks.values() if c.source_id == source_id), "")

    def _grounded(self, r: Retriever, question: str, mode: Mode) -> tuple[AskResponse, int]:
        try:
            with self._model_lock:
                res = r.search(question, mode.value, top_k=5)
        except Exception:  # noqa: BLE001
            return self._error(mode, "retrieval_failed", "Retrieval failed unexpectedly.", 500), 500

        # Routing-only chunks are links, not evidence. Unexpected evidence_role values are never treated as evidence.
        links = [
            l
            for h in res.hits
            if h.chunk.evidence_role == "routing_only"
            for l in [self._link(r, h.chunk.source_id, h.chunk.title)]
            if l
        ]
        usable = [
            h
            for h in res.hits
            if h.chunk.evidence_role == "answer_evidence" and not h.chunk.conflict_ids
        ][: self.s.evidence_passages]
        if not usable:
            return self._insufficient(mode, links, False), 200
        labelled = {f"S{i}": h.chunk for i, h in enumerate(usable, 1)}
        passages = [Passage(l, c.text, c.title, c.section) for l, c in labelled.items()]
        try:
            raw = self.llm.generate(question, passages)
        except LLMError as exc:
            return self._error(mode, exc.code, exc.message, exc.http_status), exc.http_status

        parsed = parse_reply(raw)
        if parsed is None or not parsed.sufficient or not parsed.answer:
            return self._insufficient(mode, links, True), 200
        marker_ids = {f"S{n}" for n in MARKER.findall(parsed.answer)}
        cited = list(dict.fromkeys(parsed.citations))
        if not cited or any(c not in labelled for c in cited) or any(m not in labelled for m in marker_ids):
            return self._insufficient(mode, links, True), 200  # unknown or missing citation IDs
        if LIMIT_CLAIM.search(parsed.answer):
            return self._insufficient(mode, links, True), 200  # would state a disputed numeric limit
        return (
            AskResponse(
                status="answered",
                answer=parsed.answer,
                mode=mode,
                grounded=True,
                citations=[_citation(c, labelled[c]) for c in cited],
                official_links=[],
                llm_used=True,
            ),
            200,
        )

    def _insufficient(self, mode: Mode, links: list[OfficialLink], llm_used: bool) -> AskResponse:
        return AskResponse(
            status="insufficient_evidence",
            answer=INSUFFICIENT,
            mode=mode,
            grounded=False,
            official_links=links,
            llm_used=llm_used,
        )

    @staticmethod
    def _error(mode: Mode, code: ErrorCode | str, message: str, http_status: int) -> AskResponse:
        _ = http_status
        safe_code = cast(ErrorCode, code)
        return AskResponse(
            status="error",
            answer=message,
            mode=mode,
            grounded=False,
            error=ErrorInfo(code=safe_code, message=message),
        )
