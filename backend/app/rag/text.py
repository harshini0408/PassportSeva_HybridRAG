"""Tokenisation shared by indexing and querying.

Acronyms (ARN, PSK, POPSK) stay whole tokens, slash/hyphen compounds are
emitted both whole and split, and known portal button labels also produce one
atomic ``label:...`` token so an exact label in a query matches strongly.
"""
from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)*")
STOPWORDS = frozenset(
    "a an the of to for and or in on at is are be it its this that with as by from do i my me can "
    "how what when where which should would if so".split()
)

PORTAL_LABELS = [
    "Pay and Schedule Appointment",
    "Schedule/Reschedule Appointment",
    "Reschedule Appointment",
    "Schedule Appointment",
    "Track Payment Status",
    "Apply for Fresh Passport/Re-issue of Passport",
    "View Saved/Submitted Applications",
    "Document Advisor",
]
_LABEL_PATTERNS = [
    (re.compile(re.escape(label).replace(r"\ ", r"\s+"), re.IGNORECASE),
     "label:" + re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_"))
    for label in PORTAL_LABELS
]


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for m in TOKEN_RE.finditer(text):
        tok = m.group(0).lower()
        parts = re.split(r"[-/]", tok)
        if len(parts) > 1:
            tokens.append(tok)
        for p in parts:
            if p and p not in STOPWORDS:
                tokens.append(p)
    for pattern, atom in _LABEL_PATTERNS:
        if pattern.search(text):
            tokens.append(atom)
    return tokens

