"""Load, validate and chunk the Passport Seva dataset.

Inputs: ``source_manifest.json``, ``seed_records.jsonl`` and optional original
files in ``raw/`` named ``<source_id>.pdf|.txt|.md``. ``smoke_questions.jsonl``
is never read here, so it cannot reach the index.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from .config import Settings

EVIDENCE_ROLES = {"answer_evidence", "routing_only"}
SEED_REQUIRED = (
    "record_id", "source_id", "title", "source_url", "reviewed_on", "section",
    "pdf_pages", "topic", "application_type", "evidence_use", "text_kind", "text",
)
GENERIC_HEADINGS = {"Full text", "Start of document"}
MIN_ORIGINAL_CHARS = 80
STEP_RE = re.compile(r"^\s*(?:step\s*)?(\d{1,2})\s*[.):\-]\s+\S", re.IGNORECASE)
HEADING_RE = re.compile(r"^\s*#{1,6}\s+(.+?)\s*$")
JS_SHELL_MARKERS = (
    "enable javascript", "javascript is disabled", "javascript is required",
    "<noscript", "<script", "window.__", "you need to enable javascript",
)


class DatasetError(ValueError):
    """Raised when the manifest or seed records fail validation."""


@dataclass
class Chunk:
    chunk_id: str
    source_id: str
    record_id: str | None
    title: str
    source_url: str
    section: str
    reviewed_on: str
    pdf_pages: list[int] | None
    topic: str
    applicant_scope: str       # seed application_type: general / fresh / reissue
    scope_note: str            # manifest scope (adults, India, ordinary, Normal)
    evidence_role: str         # answer_evidence / routing_only
    origin: str                # seed | original_pdf | original_text
    text: str
    supersedes: list[str] = field(default_factory=list)
    conflict_ids: list[str] = field(default_factory=list)
    text_sha: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Chunk":
        return cls(**d)


@dataclass
class IngestReport:
    seeds_loaded: int = 0
    originals_accepted: list[str] = field(default_factory=list)
    originals_rejected: list[str] = field(default_factory=list)   # "file: reason"
    seeds_superseded: list[str] = field(default_factory=list)
    duplicates_dropped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "section"


def _check_date(value: Any, where: str, errors: list[str]) -> None:
    try:
        date.fromisoformat(str(value))
    except ValueError:
        errors.append(f"{where}: invalid ISO date {value!r}")


# ---------------------------------------------------------------- validation
def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    _check_date(manifest.get("reviewed_on"), "manifest.reviewed_on", errors)
    seen: set[str] = set()
    for src in manifest.get("sources", []):
        sid = src.get("source_id")
        if not sid or sid in seen:
            errors.append(f"manifest: missing or duplicate source_id {sid!r}")
        seen.add(sid)
        for key in ("title", "url", "kind"):
            if not str(src.get(key, "")).strip():
                errors.append(f"manifest source {sid}: empty {key}")
        if not str(src.get("url", "")).startswith("https://"):
            errors.append(f"manifest source {sid}: url must be https")
        _check_date(src.get("reviewed_on"), f"manifest source {sid}.reviewed_on", errors)
    for conflict in manifest.get("known_conflicts", []):
        for sid in conflict.get("sources", []):
            if sid not in seen:
                errors.append(f"conflict {conflict.get('id')}: unknown source {sid}")
    if errors:
        raise DatasetError("\n".join(errors))
    return manifest


def load_seeds(path: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    sources = {s["source_id"]: s for s in manifest["sources"]}
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    seen: set[str] = set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {n}: invalid JSON ({exc})")
            continue
        rid = rec.get("record_id")
        where = f"line {n} ({rid})"
        for key in SEED_REQUIRED:
            if key not in rec:
                errors.append(f"{where}: missing field {key}")
        if errors and any(e.startswith(where) for e in errors):
            continue
        if not str(rid).strip() or rid in seen:
            errors.append(f"{where}: empty or duplicate record_id")
        seen.add(rid)
        src = sources.get(rec["source_id"])
        if src is None:
            errors.append(f"{where}: unknown source_id {rec['source_id']!r}")
        else:
            if rec["source_url"] != src["url"]:
                errors.append(f"{where}: source_url does not match manifest")
            if rec["title"] != src["title"]:
                errors.append(f"{where}: title does not match manifest")
        if not str(rec["text"]).strip():
            errors.append(f"{where}: empty text")
        if not str(rec["section"]).strip():
            errors.append(f"{where}: empty section")
        if rec["evidence_use"] not in EVIDENCE_ROLES:
            errors.append(f"{where}: evidence_use {rec['evidence_use']!r} not in {sorted(EVIDENCE_ROLES)}")
        _check_date(rec["reviewed_on"], where, errors)
        pages = rec["pdf_pages"]
        if pages is not None:
            ok = (isinstance(pages, list) and pages and
                  all(isinstance(p, int) and not isinstance(p, bool) and p >= 1 for p in pages)
                  and pages == sorted(set(pages)))
            if not ok:
                errors.append(f"{where}: pdf_pages must be sorted unique one-based ints, got {pages!r}")
            elif src is not None and src["kind"] != "pdf_guide":
                errors.append(f"{where}: pdf_pages set but source kind is {src['kind']}")
        elif src is not None and src["kind"] == "pdf_guide":
            errors.append(f"{where}: PDF source requires pdf_pages")
        records.append(rec)
    if errors:
        raise DatasetError("\n".join(errors))
    return records


# ------------------------------------------------------------- original files
def reject_reason(text: str) -> str | None:
    """Return why extracted/cleaned text is unusable, else None."""
    stripped = text.strip()
    if not stripped:
        return "empty"
    lowered = stripped.lower()
    if any(marker in lowered for marker in JS_SHELL_MARKERS):
        return "looks like a JavaScript shell or raw HTML"
    if lowered.startswith(("<!doctype", "<html")):
        return "raw HTML, save cleaned text instead"
    if len(re.findall(r"[A-Za-z]{2,}", stripped)) < 12 or len(stripped) < MIN_ORIGINAL_CHARS:
        return "too little text"
    return None


Line = tuple[str, int | None]          # (text, one-based pdf page or None)
Block = tuple[str, list[int]]          # (text, pages)
NOISE_RE = re.compile(r"^\s*Figure\s+\d+\s*$", re.IGNORECASE)
PDF_HEADING_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+([A-Z][^.:;]{2,58})\s*$")


def group_lines(lines: list[Line]) -> list[Block]:
    """Group lines into blocks. Each numbered step keeps its continuation and
    condition lines (wrapped text, bullets); blank-line paragraphs stay separate."""
    blocks: list[tuple[list[str], set[int]]] = []
    after_blank = False
    for raw, page in lines:
        line = raw.rstrip()
        if NOISE_RE.match(line):
            continue
        if not line.strip():
            after_blank = True
            continue
        starts_step = bool(STEP_RE.match(line))
        if starts_step or not blocks or (after_blank and not STEP_RE.match(blocks[-1][0][0])):
            blocks.append(([line.strip()], set()))
        else:
            blocks[-1][0].append(line.strip())
        if page is not None:
            blocks[-1][1].add(page)
        after_blank = False
    return [(" ".join(b), sorted(p)) for b, p in blocks]


def group_steps(text: str) -> list[str]:
    return [b for b, _ in group_lines([(l, None) for l in text.splitlines()])]


def pack_blocks(blocks: list[str], max_chars: int) -> list[str]:
    """Pack whole blocks into chunks. A numbered step is never split."""
    return ["\n".join(b for b, _ in grp) for grp in _pack(([(b, []) for b in blocks]), max_chars)]


def _pack(blocks: list[Block], max_chars: int) -> list[list[Block]]:
    groups: list[list[Block]] = []
    current: list[Block] = []
    size = 0
    for block in blocks:
        if current and size + len(block[0]) + 1 > max_chars:
            groups.append(current)
            current, size = [], 0
        current.append(block)
        size += len(block[0]) + 1
    if current:
        groups.append(current)
    return groups


def _extract_pdf(path: Path) -> dict[int, str]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:                      # pragma: no cover
        raise DatasetError("pypdf is required to ingest PDFs (pip install pypdf)") from exc
    reader = PdfReader(str(path))
    pages: dict[int, str] = {}
    for i, page in enumerate(reader.pages, start=1):   # one-based viewer pages
        pages[i] = (page.extract_text() or "").strip()
    return pages


def _find_original(raw_dir: Path, source_id: str) -> Path | None:
    for ext in (".pdf", ".txt", ".md"):
        p = raw_dir / f"{source_id}{ext}"
        if p.is_file():
            return p
    return None


def _heading_matches(heading: str, seed_section: str) -> bool:
    h = _norm(re.sub(r"^\s*#*\s*", "", heading))
    s = _norm(seed_section)
    return bool(h) and (h in s or s in h)


def _sectioned(lines: list[Line], pdf: bool) -> list[tuple[str, list[Line]]]:
    """Split lines into (heading, lines) sections; PDF headings carry across pages."""
    sections: list[tuple[str, list[Line]]] = []
    for i, (text, page) in enumerate(lines):
        if pdf:
            m = PDF_HEADING_RE.match(text)
            nxt = next((l for l, _ in lines[i + 1:] if l.strip()), "")
            # a wrapped step line is followed by lowercase text; a heading is not
            if m and not (nxt[:1].islower()):
                sections.append((f"{m.group(1)}. {m.group(2).strip()}", []))
                continue
        else:
            m = HEADING_RE.match(text)
            if m:
                sections.append((m.group(1), []))
                continue
        if not sections:
            sections.append(("Full text" if not pdf else "Start of document", []))
        sections[-1][1].append((text, page))
    return sections


def _original_chunks(path: Path, src: dict[str, Any], seeds: list[dict[str, Any]],
                     manifest: dict[str, Any], settings: Settings,
                     report: IngestReport) -> list[Chunk]:
    """Chunks from one original file. Each chunk lists the seeds it replaces."""
    sid = src["source_id"]
    base = dict(
        source_id=sid, record_id=None, title=src["title"], source_url=src["url"],
        reviewed_on=src["reviewed_on"], scope_note=manifest["scope"],
        evidence_role="answer_evidence", text_sha="",
    )
    my_seeds = [s for s in seeds if s["source_id"] == sid]
    topic_default = my_seeds[0]["topic"] if my_seeds else "general"
    lines: list[Line] = []
    usable: set[int] = set()

    if path.suffix == ".pdf":
        if src["kind"] != "pdf_guide":
            report.originals_rejected.append(f"{path.name}: source kind is {src['kind']}, not a PDF")
            return []
        try:
            pages = _extract_pdf(path)
        except Exception as exc:                    # corrupt PDF
            report.originals_rejected.append(f"{path.name}: unreadable PDF ({exc})")
            return []
        for num, text in pages.items():
            why = reject_reason(text)
            if why:
                report.warnings.append(f"{path.name} page {num}: no usable text ({why}); seeds for it are kept")
                continue
            usable.add(num)
            lines.extend((l, num) for l in text.splitlines())
        if not usable:
            report.originals_rejected.append(f"{path.name}: no page had usable text (scanned or image-only?)")
            return []
        origin, pdf = "original_pdf", True
    else:
        text = path.read_text(encoding="utf-8", errors="strict")
        why = reject_reason(text)
        if why:
            report.originals_rejected.append(f"{path.name}: {why}")
            return []
        lines = [(l, None) for l in text.splitlines()]
        origin, pdf = "original_text", False

    chunks: list[Chunk] = []
    for heading, sec_lines in _sectioned(lines, pdf):
        blocks = group_lines(sec_lines)
        if len(" ".join(b for b, _ in blocks).split()) < 8:
            continue                      # title-only or empty section
        seed = next((s for s in my_seeds if _heading_matches(heading, s["section"])
                     and (not pdf or (s["pdf_pages"] and set(s["pdf_pages"]) <= usable))), None)
        for n, grp in enumerate(_pack(blocks, settings.chunk_max_chars)):
            pages = sorted({p for _, ps in grp for p in ps})
            chunks.append(Chunk(
                chunk_id=f"{sid}:{_slug(heading)}:{n}", section=heading,
                pdf_pages=pages or None, topic=seed["topic"] if seed else topic_default,
                applicant_scope=seed["application_type"] if seed else "general",
                origin=origin, text=("\n".join(b for b, _ in grp) if heading in GENERIC_HEADINGS
                      else heading + "\n" + "\n".join(b for b, _ in grp)),
                supersedes=[seed["record_id"]] if seed else [], **base))
    if not chunks:
        report.originals_rejected.append(f"{path.name}: no chunks produced")
        return []
    report.originals_accepted.append(path.name)
    return chunks


# ---------------------------------------------------------------- conflicts
NUMBER_WORDS = r"(?:one|two|three|four|five|once|twice|thrice|\d+)"


def conflict_ids_for(text: str, source_id: str, manifest: dict[str, Any]) -> list[str]:
    """Tag chunks that may state the contested reschedule number."""
    out = []
    count_near_reschedule = re.compile(
        rf"\b{NUMBER_WORDS}\b[^.\n]{{0,40}}\b(re-?schedul\w*|times|attempts?|options?)\b|"
        rf"\b(re-?schedul\w*|cancel\w*)\b[^.\n]{{0,40}}\b(?:once|twice|thrice|two|three|four|five)\b", re.I)
    for conflict in manifest.get("known_conflicts", []):
        if conflict["id"] == "reschedule_limit" and source_id in conflict["sources"]:
            if re.search(r"re-?schedul", text, re.I) and count_near_reschedule.search(text):
                out.append(conflict["id"])
    return out

# ------------------------------------------------------------------ driver
def build_chunks(settings: Settings) -> tuple[list[Chunk], dict[str, Any], IngestReport]:
    manifest = load_manifest(settings.data_dir / "source_manifest.json")
    seeds = load_seeds(settings.data_dir / "seed_records.jsonl", manifest)
    report = IngestReport(seeds_loaded=len(seeds))
    sources = {s["source_id"]: s for s in manifest["sources"]}

    original_chunks: list[Chunk] = []
    if settings.raw_dir.is_dir():
        for sid, src in sources.items():
            path = _find_original(settings.raw_dir, sid)
            if path is not None:
                original_chunks.extend(_original_chunks(path, src, seeds, manifest, settings, report))
        for stray in settings.raw_dir.iterdir():
            if stray.is_file() and stray.stem not in sources:
                report.warnings.append(f"raw/{stray.name}: filename does not match any source_id; ignored")

    # A seed is dropped only when an original chunk replaces it, so evidence is indexed once.
    superseded = {rid: ch.chunk_id for ch in original_chunks for rid in ch.supersedes}

    seed_chunks: list[Chunk] = []
    for s in seeds:
        if s["record_id"] in superseded:
            report.seeds_superseded.append(f"{s['record_id']} (by {superseded[s['record_id']]})")
            continue
        seed_chunks.append(Chunk(
            chunk_id=f"{s['source_id']}:{s['record_id']}", source_id=s["source_id"],
            record_id=s["record_id"], title=s["title"], source_url=s["source_url"],
            section=s["section"], reviewed_on=s["reviewed_on"], pdf_pages=s["pdf_pages"],
            topic=s["topic"], applicant_scope=s["application_type"], scope_note=manifest["scope"],
            evidence_role=s["evidence_use"], origin="seed", text=s["text"].strip()))

    final: list[Chunk] = []
    seen_text: dict[str, str] = {}
    for ch in original_chunks + seed_chunks:
        key = _norm(ch.text)
        if key in seen_text:
            report.duplicates_dropped.append(f"{ch.chunk_id} duplicates {seen_text[key]}")
            continue
        seen_text[key] = ch.chunk_id
        ch.text_sha = _sha(ch.text)
        ch.conflict_ids = conflict_ids_for(ch.text, ch.source_id, manifest)
        final.append(ch)
    ids = [c.chunk_id for c in final]
    if len(ids) != len(set(ids)):
        raise DatasetError("duplicate chunk_id generated")
    return final, manifest, report
