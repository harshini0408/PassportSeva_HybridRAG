"""Semantic, hybrid (RRF) and hybrid + cross-encoder retrieval."""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import Settings
from .index import LoadedIndex, embed, load_embedder, load_index
from .ingest import Chunk
from .policy import Decision, classify

MODES = ("semantic", "hybrid", "rerank")


@dataclass
class Hit:
    rank: int
    chunk: Chunk
    score: float
    detail: dict = field(default_factory=dict)
    withheld: bool = False      # numeric reschedule limit must not be shown or used


@dataclass
class Result:
    question: str
    mode: str
    decision: Decision
    hits: list[Hit]
    conflict_sources: list[dict] = field(default_factory=list)


def rrf(rankings: list[list[str]], k: int) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for pos, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + pos)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


def chunk_matches(chunk: Chunk, acceptable: set[str]) -> bool:
    """A chunk satisfies an expected seed id directly or by replacing that seed."""
    return bool(chunk.record_id in acceptable or set(chunk.supersedes) & acceptable)


class Retriever:
    def __init__(self, settings: Settings, index: LoadedIndex | None = None):
        self.s = settings
        self.idx = index or load_index(settings)
        self._embedder = None
        self._reranker = None
        self.manifest_conflict_handling = ""
        self._conflict_sources: list[str] = []
        # Conflict metadata travels with the chunks; sources come from the index chunks.
        self._conflict_urls: dict[str, dict] = {}
        for c in self.idx.chunks.values():
            self._conflict_urls.setdefault(c.source_id, {"title": c.title, "url": c.source_url})

    def set_conflict_handling(self, manifest: dict) -> None:
        for conflict in manifest.get("known_conflicts", []):
            if conflict["id"] == "reschedule_limit":
                self.manifest_conflict_handling = conflict["handling"]
                self._conflict_sources = conflict["sources"]

    def conflict_sources(self) -> list[dict]:
        """Title/URL of the sources that disagree on the reschedule limit."""
        return [{"source_id": s, **self._conflict_urls[s]}
                for s in self._conflict_sources if s in self._conflict_urls]

    def source_link(self, source_id: str) -> dict | None:
        """Title/URL for any indexed source id (used for Document Advisor links)."""
        return self._conflict_urls.get(source_id)

    # ----------------------------------------------------------- components
    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = load_embedder(self.s.embedding_model)
        return self._embedder

    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(self.s.reranker_model)
        return self._reranker

    def semantic(self, query: str, k: int) -> list[tuple[str, float]]:
        vec = embed(self.embedder, [self.s.query_prefix + query])
        scores, ids = self.idx.faiss_index.search(vec, min(k, len(self.idx.order)))
        return [(self.idx.order[i], float(s)) for s, i in zip(scores[0], ids[0]) if i >= 0]

    def keyword(self, query: str, k: int) -> list[tuple[str, float]]:
        return self.idx.bm25.search(query, k)

    # --------------------------------------------------------------- search
    def search(self, question: str, mode: str = "rerank", top_k: int | None = None) -> Result:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        top_k = top_k or self.s.top_k
        n = self.s.candidates_per_ranker
        decision = classify(question, self.manifest_conflict_handling)
        hits: list[Hit] = []

        if decision.behavior in ("out_of_scope", "live_status_redirect"):
            return Result(question, mode, decision, [])      # no evidence should be shown

        sem = self.semantic(question, n)
        if mode == "semantic":
            ranked = sem
            details = {cid: {"semantic": s} for cid, s in sem}
        else:
            kw = self.keyword(question, n)
            fused = rrf([[c for c, _ in sem], [c for c, _ in kw]], self.s.rrf_k)
            sem_rank = {c: r for r, (c, _) in enumerate(sem, 1)}
            kw_rank = {c: r for r, (c, _) in enumerate(kw, 1)}
            details = {c: {"rrf": s, "semantic_rank": sem_rank.get(c), "bm25_rank": kw_rank.get(c)}
                       for c, s in fused}
            ranked = fused
            if mode == "rerank":
                shortlist = fused[: self.s.rerank_shortlist]
                pairs = [(question, self.idx.chunks[c].text) for c, _ in shortlist]
                ce = self.reranker.predict(pairs)
                for (c, _), sc in zip(shortlist, ce):
                    details[c]["cross_encoder"] = float(sc)
                ranked = sorted(((c, float(sc)) for (c, _), sc in zip(shortlist, ce)),
                                key=lambda x: (-x[1], x[0]))
        for rank, (cid, score) in enumerate(ranked[:top_k], start=1):
            chunk = self.idx.chunks[cid]
            hits.append(Hit(rank, chunk, score, details.get(cid, {}),
                            withheld=decision.behavior == "conflicting_sources" and bool(chunk.conflict_ids)))

        conflict_sources = []
        if decision.behavior == "conflicting_sources":
            conflict_sources = [{"source_id": s, **self._conflict_urls[s]}
                                for s in self._conflict_sources if s in self._conflict_urls]
        return Result(question, mode, decision, hits, conflict_sources)


def make_retriever(settings: Settings) -> Retriever:
    from .ingest import load_manifest
    r = Retriever(settings)
    r.set_conflict_handling(load_manifest(settings.data_dir / "source_manifest.json"))
    return r
