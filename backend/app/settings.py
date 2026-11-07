"""Application settings for the API layer (Groq, limits, official links).

Values come from the process environment, then ``backend/.env``. Empty values
count as unset. The API key is held in a private attribute and is never part of
``repr`` or any response.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
# Not part of the curated dataset; override with PASSPORT_TRACKER_URL if it changes.
DEFAULT_TRACKER_URL = "https://passportindia.gov.in/AppOnlineProject/statusTracker/trackStatusInpNew"


@dataclass(frozen=True)
class AppSettings:
    groq_api_key: str = field(default="", repr=False)
    groq_model: str = DEFAULT_GROQ_MODEL
    groq_timeout_s: float = 20.0
    max_question_chars: int = 500
    evidence_passages: int = 4
    default_mode: str = "hybrid"
    tracker_url: str = DEFAULT_TRACKER_URL

    @property
    def groq_configured(self) -> bool:
        return bool(self.groq_api_key.strip())


def load_app_settings() -> AppSettings:
    # override=False: real environment variables win over .env. Also makes
    # PASSPORT_* values in .env visible to the Phase 1 settings loader.
    load_dotenv(BACKEND_DIR / ".env", override=False)
    env = os.environ.get
    return AppSettings(
        groq_api_key=(env("GROQ_API_KEY") or "").strip(),
        groq_model=(env("GROQ_MODEL") or DEFAULT_GROQ_MODEL).strip(),
        groq_timeout_s=float(env("GROQ_TIMEOUT_SECONDS") or 20.0),
        max_question_chars=int(env("PASSPORT_MAX_QUESTION_CHARS") or 500),
        evidence_passages=int(env("PASSPORT_EVIDENCE_PASSAGES") or 4),
        tracker_url=(env("PASSPORT_TRACKER_URL") or DEFAULT_TRACKER_URL).strip(),
    )

