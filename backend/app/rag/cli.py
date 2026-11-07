"""Terminal entry point: build the index, run a query, run smoke checks.

    python -m app.rag.cli build
    python -m app.rag.cli query "How do I reschedule?" --mode rerank
    python -m app.rag.cli smoke
"""
from __future__ import annotations

import argparse
import json
import sys

from .config import load_settings
from .index import IndexError_, build_index
from .retrieve import MODES, Result, chunk_matches, make_retriever

# Question types whose expected behaviour is not "answer" are checked separately.
EVIDENCE_BEHAVIORS = {"answer", "official_redirect", "conflicting_sources"}


def citation(chunk) -> str:
    pages = ""
    if chunk.pdf_pages:
        first, last = chunk.pdf_pages[0], chunk.pdf_pages[-1]
        pages = f" (PDF page {first})" if first == last else f" (PDF pages {first}-{last})"
    return f"{chunk.title} > {chunk.section}{pages}\n    {chunk.source_url}" + (
        f"#page={chunk.pdf_pages[0]}" if chunk.pdf_pages else "")


def print_result(res: Result) -> None:
    print(f"Question : {res.question}")
    print(f"Mode     : {res.mode}")
    print(f"Decision : {res.decision.behavior} - {res.decision.reason}")
    if res.decision.note:
        print(f"Guidance : {res.decision.note}")
    for src in res.conflict_sources:
        print(f"  conflicting source: {src['title']} - {src['url']}")
    if not res.hits:
        print("(no passages shown for this decision)")
        return
    for h in res.hits:
        c = h.chunk
        role = "ROUTING ONLY - link to official guidance, not a checklist" if c.evidence_role == "routing_only" \
            else "answer evidence"
        extra = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in h.detail.items())
        print(f"\n[{h.rank}] {c.chunk_id}  score={h.score:.4f}  [{role}; {c.origin}; {c.applicant_scope}; {c.topic}]")
        print("    [text withheld: it states a reschedule limit that official sources dispute]" if h.withheld
              else "    " + c.text.replace("\n", "\n    "))
        print(f"    Citation: {citation(c)}  (reviewed {c.reviewed_on})")
        if extra:
            print(f"    ({extra})")


def cmd_build(args) -> int:
    s = load_settings()
    meta, report = build_index(s)
    print(f"Indexed {meta['chunk_count']} chunks {meta['chunk_origins']} -> {s.index_dir}")
    print(f"Embedding model : {meta['embedding_model']}")
    print(f"Similarity      : {meta['similarity_metric']} (dim {meta['embedding_dim']})")
    print(f"Originals accepted: {report.originals_accepted or 'none'}")
    for r in report.originals_rejected:
        print(f"Original REJECTED: {r}")
    for r in report.seeds_superseded:
        print(f"Seed superseded  : {r}")
    for r in report.duplicates_dropped:
        print(f"Duplicate dropped: {r}")
    for w in report.warnings:
        print(f"Warning: {w}")
    return 0


def cmd_query(args) -> int:
    res = make_retriever(load_settings()).search(args.question, args.mode, args.top_k)
    print_result(res)
    return 0


def _run_question_set(retriever, path, modes: list[str], top_k: int, label: str) -> tuple[int, dict]:
    questions = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    failures = 0
    routing_correct = 0
    retrieval_total = sum(
        1 for q in questions if q["expected_behavior"] in EVIDENCE_BEHAVIORS and q["acceptable_seed_record_ids"]
    )
    top1_hits = {m: 0 for m in modes}
    top5_hits = {m: 0 for m in modes}

    print(f"=== {label} ({path.name}, n={len(questions)}) ===")
    print(f"{'#':>2} {'expected behaviour':<20} {'decision':<20} " + " ".join(f"{m:>9}" for m in modes) + "  question")
    for n, q in enumerate(questions, 1):
        want = q["expected_behavior"]
        acceptable = set(q["acceptable_seed_record_ids"])
        cells, ok_row = [], True
        decision = None
        for mode in modes:
            res = retriever.search(q["question"], mode, top_k)
            decision = res.decision.behavior
            if want in EVIDENCE_BEHAVIORS and acceptable:
                rank = next((h.rank for h in res.hits if chunk_matches(h.chunk, acceptable)), None)
                cell = f"rank {rank}" if rank else "MISS"
                ok = rank is not None and decision == want
                if rank == 1:
                    top1_hits[mode] += 1
                if rank is not None and rank <= top_k:
                    top5_hits[mode] += 1
            else:
                cell = "ok" if decision == want else "WRONG"
                ok = decision == want and not res.hits if want in ("out_of_scope", "live_status_redirect") \
                    else decision == want
            cells.append(f"{cell:>9}")
            ok_row &= ok
        if decision == want:
            routing_correct += 1
        failures += not ok_row
        print(f"{n:>2} {want:<20} {decision:<20} " + " ".join(cells) + f"  {q['question']}")

    total_q = len(questions)
    print(f"\nRouting accuracy  : {routing_correct}/{total_q} ({100.0 * routing_correct / total_q:.1f}%)")
    if retrieval_total:
        for m in modes:
            t1 = top1_hits[m]
            t5 = top5_hits[m]
            print(
                f"Retrieval [{m:<8}]: top-1 = {t1}/{retrieval_total} ({100.0 * t1 / retrieval_total:.1f}%), "
                f"top-{top_k} = {t5}/{retrieval_total} ({100.0 * t5 / retrieval_total:.1f}%)"
            )
    print(f"Overall pass      : {total_q - failures}/{total_q} questions passed in all modes {modes}\n")
    return failures, {
        "total": total_q,
        "routing_correct": routing_correct,
        "retrieval_total": retrieval_total,
        "top1": top1_hits,
        "top5": top5_hits,
    }


def cmd_smoke(args) -> int:
    s = load_settings()
    retriever = make_retriever(s)
    modes = args.modes or list(MODES)
    failures, _ = _run_question_set(retriever, s.data_dir / "smoke_questions.jsonl", modes, s.top_k, "Starter smoke")
    return 1 if failures else 0


def cmd_eval(args) -> int:
    s = load_settings()
    retriever = make_retriever(s)
    modes = args.modes or list(MODES)
    f1, _ = _run_question_set(retriever, s.data_dir / "smoke_questions.jsonl", modes, s.top_k, "Starter smoke (regression)")
    f2, _ = _run_question_set(retriever, s.data_dir / "eval_questions.jsonl", modes, s.top_k, "Phase 4 evaluation set")
    return 1 if (f1 or f2) else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m app.rag.cli")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="validate data, build BM25 + FAISS").set_defaults(fn=cmd_build)
    q = sub.add_parser("query", help="retrieve passages and citations")
    q.add_argument("question")
    q.add_argument("--mode", choices=MODES, default="rerank")
    q.add_argument("--top-k", type=int, default=None)
    q.set_defaults(fn=cmd_query)
    sm = sub.add_parser("smoke", help="run smoke_questions.jsonl against the index")
    sm.add_argument("--modes", nargs="+", choices=MODES)
    sm.set_defaults(fn=cmd_smoke)
    ev = sub.add_parser("eval", help="run smoke and eval question sets with top-1/top-5 and routing metrics")
    ev.add_argument("--modes", nargs="+", choices=MODES)
    ev.set_defaults(fn=cmd_eval)
    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except (IndexError_, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
