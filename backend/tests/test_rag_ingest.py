import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from app.rag.bm25 import BM25
from app.rag.config import DATA_DIR, load_settings
from app.rag.ingest import (DatasetError, build_chunks, group_steps, load_manifest, load_seeds,
                            pack_blocks, reject_reason)
from app.rag.policy import classify
from app.rag.retrieve import rrf
from app.rag.text import tokenize


@pytest.fixture()
def data(tmp_path):
    dst = tmp_path / "passport_seva"
    shutil.copytree(DATA_DIR, dst, ignore=shutil.ignore_patterns("index", "raw"))
    return replace(load_settings(), data_dir=dst, index_dir=dst / "index")


def test_real_dataset_validates_and_excludes_smoke(data):
    chunks, _, _ = build_chunks(data)
    assert len(chunks) == 11
    smoke = [json.loads(l)["question"] for l in (data.data_dir / "smoke_questions.jsonl").read_text("utf-8").splitlines()]
    eval_qs = [json.loads(l)["question"] for l in (data.data_dir / "eval_questions.jsonl").read_text("utf-8").splitlines()]
    assert len(eval_qs) >= 12
    assert not any(q in c.text for c in chunks for q in (smoke + eval_qs))
    pdf = next(c for c in chunks if c.record_id == "reschedule_pdf")
    assert pdf.pdf_pages == [5, 6]
    assert sum(c.evidence_role == "routing_only" for c in chunks) == 3


@pytest.mark.parametrize("field,value,msg", [
    ("record_id", "arn", "duplicate record_id"),
    ("source_id", "nope", "unknown source_id"),
    ("reviewed_on", "2026-13-40", "invalid ISO date"),
    ("text", "   ", "empty text"),
    ("pdf_pages", [0, 1], "one-based"),
    ("source_url", "https://example.com", "does not match manifest"),
    ("evidence_use", "checklist", "evidence_use"),
])
def test_seed_validation_errors(data, field, value, msg):
    path = data.data_dir / "seed_records.jsonl"
    rows = [json.loads(l) for l in path.read_text("utf-8").splitlines()]
    rows[1][field] = value
    if field == "record_id":
        rows[1][field] = rows[0]["record_id"]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    manifest = load_manifest(data.data_dir / "source_manifest.json")
    with pytest.raises(DatasetError, match=msg):
        load_seeds(path, manifest)


def test_js_shell_and_empty_originals_rejected(data):
    raw = data.raw_dir
    raw.mkdir()
    (raw / "apply.txt").write_text("<html><script>window.__x=1</script>Please enable JavaScript</html>", "utf-8")
    (raw / "reissue.md").write_text("   \n", "utf-8")
    chunks, _, report = build_chunks(data)
    assert len(chunks) == 11 and not report.originals_accepted
    assert len(report.originals_rejected) == 2


def test_original_text_supersedes_matching_seed_once(data):
    raw = data.raw_dir
    raw.mkdir()
    (raw / "apply.txt").write_text(
        "## Application submission\n"
        "1. Submit the completed application form after reviewing every entry on screen.\n"
        "   If any detail is wrong, edit it before final submission.\n"
        "2. Note the Application Reference Number (ARN) shown by the portal for later steps.\n", "utf-8")
    chunks, _, report = build_chunks(data)
    assert not any(c.record_id == "arn" for c in chunks)
    orig = next(c for c in chunks if c.origin == "original_text")
    assert orig.supersedes == ["arn"]
    assert "2. Note the Application Reference Number" in orig.text and "edit it before" in orig.text
    assert sum("arn" in c.chunk_id or "submission" in c.chunk_id for c in chunks) == 1
    assert report.seeds_superseded


def test_numbered_steps_never_split():
    text = "\n".join(f"{i}. Step {i} text goes here with some words.\n   If condition {i}, do more." for i in range(1, 8))
    blocks = group_steps(text)
    assert len(blocks) == 7 and "If condition 3" in blocks[2]
    for chunk in pack_blocks(blocks, 120):
        for i in range(1, 8):
            assert (f"Step {i} " in chunk) == (f"condition {i}," in chunk)


def test_reject_reason():
    assert reject_reason("") == "empty"
    assert reject_reason("<script>var a=1</script>") is not None
    assert reject_reason("word " * 40) is None


def test_tokenizer_preserves_terms():
    toks = tokenize("Find your ARN at the PSK, then Schedule/Reschedule Appointment or Pay and Schedule Appointment")
    for t in ("arn", "psk", "schedule/reschedule", "label:schedule_reschedule_appointment",
              "label:pay_and_schedule_appointment"):
        assert t in toks


def test_bm25_exact_term_and_rrf():
    bm = BM25.build(["a", "b"], ["the ARN is generated", "visit the PSK"])
    assert bm.search("ARN", 2)[0][0] == "a"
    assert BM25.from_dict(bm.to_dict()).search("psk", 1)[0][0] == "b"
    fused = rrf([["x", "y"], ["y", "z"]], 60)
    assert fused[0][0] == "y"


@pytest.mark.parametrize("q,expected", [
    ("How many times can I reschedule?", "conflicting_sources"),
    ("What is the maximum number of reschedules?", "conflicting_sources"),
    ("How do I reschedule my appointment?", "answer"),
    ("I cannot make it on my booked day. Can I move it?", "answer"),
    ("What exact documents do I need?", "clarification"),
    ("Give me the complete checklist for a Normal adult reissue.", "official_redirect"),
    ("Has my passport been dispatched today?", "live_status_redirect"),
    ("How do I apply for Tatkaal for a child?", "out_of_scope"),
    ("I paid the fee but there is no appointment. Where do I go?", "answer"),
    ("Should I bring originals or just copies?", "answer"),
])
def test_policy(q, expected):
    assert classify(q).behavior == expected

