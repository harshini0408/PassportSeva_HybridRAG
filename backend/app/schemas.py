"""Typed request/response models for the API (these drive the OpenAPI docs)."""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

MAX_QUESTION_HARD_LIMIT = 500

EvidenceRole = Literal["answer_evidence", "routing_only"]
ErrorCode = Literal[
    "groq_not_configured",
    "groq_auth_rejected",
    "groq_rate_limited",
    "groq_timeout",
    "groq_unavailable",
    "index_unavailable",
    "retrieval_failed",
]


class Mode(str, Enum):
    semantic = "semantic"
    hybrid = "hybrid"
    rerank = "rerank"


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        max_length=MAX_QUESTION_HARD_LIMIT,
        description="Plain-language question. Do not include passport numbers, ARNs or documents.",
        examples=["I can't make it on the day I booked. Can I move it?"],
    )
    mode: Mode = Field(Mode.hybrid, description="Retrieval mode. Default is hybrid.")

    @field_validator("question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be blank")
        return v


class Citation(BaseModel):
    id: str = Field(..., description="Evidence label used in the answer, e.g. S1.", examples=["S1"])
    title: str
    section: str
    url: str = Field(..., description="Official URL from backend metadata. PDF links end in #page=N.")
    pdf_pages: list[int] | None = Field(None, description="One-based PDF viewer pages, if the source is a PDF.")
    source_id: str
    reviewed_on: str
    evidence_role: EvidenceRole


class OfficialLink(BaseModel):
    label: str
    url: str


class ErrorInfo(BaseModel):
    code: ErrorCode
    message: str


AskStatus = Literal[
    "answered",
    "insufficient_evidence",
    "clarification",
    "official_redirect",
    "live_status_redirect",
    "conflicting_sources",
    "out_of_scope",
    "error",
]


class AskResponse(BaseModel):
    status: AskStatus = Field(
        ...,
        description=(
            "answered: grounded in cited evidence. insufficient_evidence: retrieval or model could not "
            "support an answer. clarification / official_redirect / live_status_redirect / "
            "conflicting_sources / out_of_scope: resolved by routing policy without calling the LLM. "
            "error: configuration or provider/index failure (see `error`)."
        ),
    )
    answer: str = Field(..., description="Text for the user. Cites evidence as [S1] when status is answered.")
    mode: Mode
    grounded: bool = Field(..., description="True only when the answer has at least one verified citation.")
    citations: list[Citation] = []
    official_links: list[OfficialLink] = []
    clarification_question: str | None = None
    llm_used: bool = False
    error: ErrorInfo | None = None


class IndexHealth(BaseModel):
    ready: bool
    chunk_count: int | None = None
    embedding_model: str | None = None
    similarity_metric: str | None = None
    detail: str | None = None


class GroqHealth(BaseModel):
    configured: bool
    model: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    backend: Literal["up"] = "up"
    index: IndexHealth
    groq: GroqHealth
