"""Rule-based routing of a question before/alongside retrieval.

Behaviours: answer, clarification, official_redirect, official_link, conflicting_sources,
live_status_redirect, out_of_scope. Deterministic on purpose so it can be
checked without a model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

OUT_OF_SCOPE = re.compile(
    r"\b(tatkaal|tatkal|child|children|minor|kid|baby|infant|diplomatic|official passport|"
    r"police clearance|pcc|visa|abroad|outside india|embassy|consulate|nri|lost passport|"
    r"damaged passport|surrender|emergency certificate)\b", re.I)
LIVE_STATUS = re.compile(
    r"\b(dispatch(ed)?|delivered|delivery|courier|speed ?post|where is my passport|"
    r"has my passport|is my passport (ready|printed)|application status|police verification status|"
    r"status of my (application|passport))\b", re.I)
RESCHEDULE = re.compile(r"re-?schedul\w*|change (my|the) (appointment|date|slot)|move (my|the|it)", re.I)
COUNT_INTENT = re.compile(
    r"\bhow (many|often|much)\b|\bnumber of (times|attempts|reschedul\w*)|\b(limit|maximum|max|cap)\b|"
    r"\bmore than (once|twice)\b|\bhow many times\b|\bremaining attempts\b|\bunlimited\b", re.I)
DOC_WORDS = re.compile(r"\b(documents?|checklist|proofs?|papers|certificates?|advisor)\b", re.I)
SPECIFIC_ASK = re.compile(r"\b(exact|complete|full|entire|list|checklist|which|what|where|choose|select|need|required|require)\b", re.I)
LINK_INTENT = re.compile(
    r"\b(which|what)\b[^.?]{0,25}\b(link|url|page|website)\b|\b(link|url)\b[^.?]{0,25}\b(use|click|open|go to)\b|"
    r"\bofficial (link|page|website)\b", re.I)
SCENARIO = re.compile(r"\b(fresh|first|new passport|re-?issue|renew\w*|expired|expiry|expiring)\b", re.I)


@dataclass
class Decision:
    behavior: str
    reason: str
    note: str = ""


def classify(question: str, reschedule_handling: str = "") -> Decision:
    q = question.strip()
    if OUT_OF_SCOPE.search(q):
        return Decision("out_of_scope", "outside adult / India / ordinary / Normal scope",
                        "This guide only covers adults applying in India for an ordinary passport "
                        "under the Normal process.")
    if LIVE_STATUS.search(q):
        return Decision("live_status_redirect", "asks for live application or dispatch status",
                        "Live status is not in this guide. Use Track Application Status on the official "
                        "Passport Seva portal.")
    if RESCHEDULE.search(q) and COUNT_INTENT.search(q):
        return Decision("conflicting_sources", "asks for the reschedule limit; official sources disagree",
                        reschedule_handling or
                        "Official sources disagree on the number. Check the attempts shown in your "
                        "Passport Seva account and confirm with official support.")
    if LINK_INTENT.search(q) and not DOC_WORDS.search(q):
        return Decision("official_link", "asks which official link to use",
                        "Show the matching official guide link from source metadata.")
    if DOC_WORDS.search(q) and SPECIFIC_ASK.search(q):
        if SCENARIO.search(q):
            return Decision("official_redirect", "asks for an exact checklist for a known scenario",
                            "No verified checklist is stored. The official Document Advisor gives the exact list.")
        return Decision("clarification", "asks about documents without Fresh/Reissue scenario",
                        "Document requirements depend on whether this is a Fresh application or a Reissue.")
    return Decision("answer", "in-scope procedural question")

