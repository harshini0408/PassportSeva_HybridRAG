"""BM25 (Okapi) over chunk ids, kept dependency-free so it is easy to persist."""
from __future__ import annotations

import math
from collections import Counter

from .text import tokenize


class BM25:
    def __init__(self, chunk_ids: list[str], docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.chunk_ids = chunk_ids
        self.k1, self.b = k1, b
        self.tf = [Counter(t) for t in docs_tokens]
        self.lens = [len(t) for t in docs_tokens]
        self.avgdl = (sum(self.lens) / len(self.lens)) if self.lens else 0.0
        df: Counter[str] = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(docs_tokens)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    @classmethod
    def build(cls, chunk_ids: list[str], texts: list[str]) -> "BM25":
        return cls(chunk_ids, [tokenize(t) for t in texts])

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        q = tokenize(query)
        scored = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            for term in q:
                f = tf.get(term)
                if f:
                    denom = f + self.k1 * (1 - self.b + self.b * self.lens[i] / (self.avgdl or 1))
                    s += self.idf[term] * f * (self.k1 + 1) / denom
            if s > 0:
                scored.append((self.chunk_ids[i], s))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]

    def to_dict(self) -> dict:
        return {"k1": self.k1, "b": self.b, "chunk_ids": self.chunk_ids,
                "tf": [dict(c) for c in self.tf]}

    @classmethod
    def from_dict(cls, d: dict) -> "BM25":
        obj = cls(d["chunk_ids"], [[] for _ in d["chunk_ids"]], d["k1"], d["b"])
        obj.tf = [Counter(x) for x in d["tf"]]
        obj.lens = [sum(c.values()) for c in obj.tf]
        obj.avgdl = (sum(obj.lens) / len(obj.lens)) if obj.lens else 0.0
        df: Counter[str] = Counter()
        for c in obj.tf:
            df.update(c.keys())
        n = len(obj.tf)
        obj.idf = {t: math.log(1 + (n - dd + 0.5) / (dd + 0.5)) for t, dd in df.items()}
        return obj

