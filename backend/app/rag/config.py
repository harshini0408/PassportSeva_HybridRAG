"""Paths and tunables for ingestion and retrieval.

Settings come from ``PASSPORT_*`` environment variables only. This module
deliberately never reads ``backend/.env`` so the Groq key is not needed (or
loaded) at this checkpoint.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "passport_seva"


@dataclass(frozen=True)
class Settings:
    data_dir: Path = DATA_DIR
    index_dir: Path = DATA_DIR / "index"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Some embedding models (e5, bge) want a prefix on queries; empty for MiniLM.
    query_prefix: str = ""
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    chunk_max_chars: int = 900
    candidates_per_ranker: int = 20
    rrf_k: int = 60
    rerank_shortlist: int = 10
    top_k: int = 5

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"


def load_settings() -> Settings:
    env = os.environ.get
    base = Settings()
    data_dir = Path(env("PASSPORT_DATA_DIR", str(base.data_dir)))
    return Settings(
        data_dir=data_dir,
        index_dir=Path(env("PASSPORT_INDEX_DIR") or str(data_dir / "index")),
        embedding_model=env("PASSPORT_EMBEDDING_MODEL") or base.embedding_model,
        query_prefix=env("PASSPORT_EMBEDDING_QUERY_PREFIX") or base.query_prefix,
        reranker_model=env("PASSPORT_RERANKER_MODEL") or base.reranker_model,
        chunk_max_chars=int(env("PASSPORT_CHUNK_MAX_CHARS") or base.chunk_max_chars),
        candidates_per_ranker=int(env("PASSPORT_CANDIDATES_PER_RANKER") or base.candidates_per_ranker),
        rrf_k=int(env("PASSPORT_RRF_K") or base.rrf_k),
        rerank_shortlist=int(env("PASSPORT_RERANK_SHORTLIST") or base.rerank_shortlist),
        top_k=int(env("PASSPORT_TOP_K") or base.top_k),
    )
