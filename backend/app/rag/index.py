"""Build and load the BM25 + FAISS indexes from one shared chunk list."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .bm25 import BM25
from .config import Settings
from .ingest import Chunk, IngestReport, build_chunks

METRIC = "cosine (inner product over L2-normalised embeddings)"
SCHEMA_VERSION = 1


class IndexError_(RuntimeError):
    pass


def load_embedder(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def embed(model, texts: list[str]) -> np.ndarray:
    vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    return np.ascontiguousarray(vecs, dtype="float32")


def build_index(settings: Settings) -> tuple[dict[str, Any], IngestReport]:
    import faiss

    chunks, manifest, report = build_chunks(settings)
    if not chunks:
        raise IndexError_("no chunks to index")
    ids = [c.chunk_id for c in chunks]
    texts = [c.text for c in chunks]

    model = load_embedder(settings.embedding_model)
    vectors = embed(model, texts)
    index = faiss.IndexFlatIP(vectors.shape[1])    # exact search; fine at this size
    index.add(vectors)
    bm25 = BM25.build(ids, texts)

    out = settings.index_dir
    out.mkdir(parents=True, exist_ok=True)
    with (out / "chunks.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for c in chunks:
            fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    (out / "bm25.json").write_text(json.dumps(bm25.to_dict()), encoding="utf-8")
    faiss.write_index(index, str(out / "faiss.index"))
    meta = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": manifest["dataset_id"],
        "manifest_reviewed_on": manifest["reviewed_on"],
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embedding_model": settings.embedding_model,
        "query_prefix": settings.query_prefix,
        "similarity_metric": METRIC,
        "embedding_dim": int(vectors.shape[1]),
        "chunk_count": len(chunks),
        "chunk_ids": ids,
        "chunk_origins": {o: sum(1 for c in chunks if c.origin == o) for o in sorted({c.origin for c in chunks})},
    }
    (out / "index_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta, report


@dataclass
class LoadedIndex:
    meta: dict[str, Any]
    chunks: dict[str, Chunk]
    bm25: BM25
    faiss_index: Any
    order: list[str]


def load_index(settings: Settings) -> LoadedIndex:
    import faiss

    out = settings.index_dir
    meta_path = out / "index_meta.json"
    if not meta_path.is_file():
        raise IndexError_(f"no index at {out}; run the build command first")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta["embedding_model"] != settings.embedding_model:
        raise IndexError_(
            f"index was built with {meta['embedding_model']!r} but PASSPORT_EMBEDDING_MODEL is "
            f"{settings.embedding_model!r}; rebuild the index or switch the model back")
    try:
        chunks = [Chunk.from_dict(json.loads(l)) for l in
                  (out / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    except ValueError as exc:                     # includes DatasetError (unknown evidence role)
        raise IndexError_(f"index chunks are invalid: {exc}; rebuild the index") from exc
    order = [c.chunk_id for c in chunks]
    bm25 = BM25.from_dict(json.loads((out / "bm25.json").read_text(encoding="utf-8")))
    fi = faiss.read_index(str(out / "faiss.index"))
    if not (order == meta["chunk_ids"] == bm25.chunk_ids) or fi.ntotal != len(order):
        raise IndexError_("chunk ids differ between chunks, BM25 and FAISS; rebuild the index")
    return LoadedIndex(meta, {c.chunk_id: c for c in chunks}, bm25, fi, order)

