"""Extract individual questions from CIA exam papers for the Exam Prep RAG.

Pipeline:
    PDF → pages → OCR if scanned → question boundary detection
      → image extraction → structured question objects → JSONL manifest

Each question becomes one retrieval document (not arbitrary chunks).
Images are saved alongside and referenced in metadata.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

import pymupdf


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExtractedQuestion:
    """One question extracted from an exam paper."""
    question_id: str
    question_number: str
    question_text: str
    ocr_text: str
    full_text: str
    has_image: bool
    image_paths: list[str]
    image_description: str
    subject: str | None
    unit: str | None
    exam: str | None
    year: str | None
    marks: int | None
    question_type: str | None
    source_file: str
    page_number: int
    options: list[str] | None = None


@dataclass
class IngestionStats:
    """Tracks ingestion pipeline statistics."""
    pdfs_processed: int = 0
    pages_processed: int = 0
    questions_extracted: int = 0
    questions_with_images: int = 0
    images_extracted: int = 0
    ocr_operations: int = 0
    boundary_failures: int = 0
    duplicates_detected: int = 0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_stem(path: Path) -> str:
    """Filesystem-safe identifier from a filename."""
    return re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")


def normalise_text(text: str) -> str:
    """Collapse whitespace and strip blank lines."""
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def deterministic_id(source_file: str, page: int, question_number: str) -> str:
    """SHA-256 based stable question ID."""
    raw = f"{source_file}|{page}|{question_number}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Metadata extraction from filename and header text
# ---------------------------------------------------------------------------

_MONTH_MAP = {
    "JAN": "January", "FEB": "February", "MAR": "March", "APR": "April",
    "MAY": "May", "JUNE": "June", "JUN": "June", "JULY": "July", "JUL": "July",
    "AUG": "August", "SEP": "September", "OCT": "October", "NOV": "November",
    "DEC": "December",
}

_SUBJECT_MAP = {
    "basic_electrical_engineering": "Basic Electrical Engineering",
    "basics_electronics": "Basics of Electronics",
    "basics_of_civil_engineering": "Basics of Civil Engineering",
    "biology_for_engineers": "Biology for Engineers",
    "engineering_chemistry": "Engineering Chemistry",
    "engineering_mathematics_1": "Engineering Mathematics I",
    "engineering_mathematics_2": "Engineering Mathematics II",
    "engineering_mechanics": "Engineering Mechanics",
    "mechanical": "Mechanical Engineering",
    "physics": "Engineering Physics",
}

_EXAM_TYPE_PATTERNS = [
    (re.compile(r"CIA[\s\-]*2", re.IGNORECASE), "CIA 2"),
    (re.compile(r"CIA[\s\-]*1", re.IGNORECASE), "CIA 1"),
    (re.compile(r"CIA[\s\-]*3", re.IGNORECASE), "CIA 3"),
    (re.compile(r"End\s*Semester|End\s*Sem", re.IGNORECASE), "End Semester"),
    (re.compile(r"Mid[\s\-]*Semester|Mid[\s\-]*Sem", re.IGNORECASE), "Mid Semester"),
]


def parse_filename_metadata(pdf_path: Path) -> dict:
    """Extract subject, exam period, and year from filename and folder."""
    filename = pdf_path.stem
    parent_dir = pdf_path.parent.name

    # Subject from folder name
    subject = _SUBJECT_MAP.get(parent_dir, parent_dir.replace("_", " ").title())

    # Year from filename: look for 4-digit year
    year_match = re.search(r"(20\d{2})", filename)
    year = year_match.group(1) if year_match else None

    # Month from filename
    exam_period = None
    for abbr, full in _MONTH_MAP.items():
        if abbr in filename.upper():
            exam_period = f"{full} {year}" if year else full
            break

    return {
        "subject": subject,
        "year": year,
        "exam_period": exam_period,
    }


def detect_exam_type(text: str) -> str | None:
    """Detect CIA / End Semester from header text."""
    for pattern, label in _EXAM_TYPE_PATTERNS:
        if pattern.search(text):
            return label
    # Default heuristic: if "End Semester" appears in header, it's end sem
    if re.search(r"End\s*Semester\s*Examination", text, re.IGNORECASE):
        return "End Semester"
    return None


# ---------------------------------------------------------------------------
# OCR (reuses project's Tesseract setup)
# ---------------------------------------------------------------------------

def render_page_image(page: pymupdf.Page, output_path: Path, dpi: int = 300) -> None:
    """Render a PDF page to a PNG image."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scale = dpi / 72
    pixmap = page.get_pixmap(
        matrix=pymupdf.Matrix(scale, scale),
        colorspace=pymupdf.csGRAY,
        alpha=False,
    )
    pixmap.save(str(output_path))


def run_tesseract_ocr(
    image_path: Path,
    tesseract_cmd: str = "tesseract",
    language: str = "eng",
    psm: int = 3,
) -> str:
    """OCR an image file and return the extracted text."""
    command = [
        tesseract_cmd, str(image_path), "stdout",
        "-l", language, "--psm", str(psm),
    ]
    result = subprocess.run(
        command, capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"Tesseract failed on {image_path}: {result.stderr.strip()}"
        )
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Image extraction from PDF pages
# ---------------------------------------------------------------------------

def extract_page_images(
    page: pymupdf.Page,
    doc: pymupdf.Document,
    output_dir: Path,
    paper_id: str,
    page_number: int,
) -> list[dict]:
    """Extract embedded images from a PDF page, save them, return metadata.

    Returns a list of dicts with keys: path, bbox, index, width, height.
    """
    images_info: list[dict] = []
    image_list = page.get_images(full=True)

    for img_index, img in enumerate(image_list, start=1):
        xref = img[0]
        try:
            base_image = doc.extract_image(xref)
        except Exception:
            continue

        if not base_image or not base_image.get("image"):
            continue

        ext = base_image.get("ext", "png")
        width = base_image.get("width", 0)
        height = base_image.get("height", 0)

        # Skip tiny images (likely decorative/borders) and full-page scans
        # Full-page scans are handled separately via page rendering
        page_rect = page.rect
        page_area = page_rect.width * page_rect.height
        img_area = width * height

        # If the image covers >90% of the page area, it's likely a full-page scan
        if img_area > 0.9 * page_area and len(image_list) == 1:
            continue  # Will be handled as scanned page

        # Skip very small images (borders, decorations)
        if width < 50 or height < 50:
            continue

        img_filename = f"p{page_number:03d}_img{img_index:02d}.{ext}"
        img_path = output_dir / img_filename
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img_path.write_bytes(base_image["image"])

        # Try to get the image's bounding box on the page
        bbox = None
        for img_rect in page.get_image_rects(xref):
            bbox = {
                "x0": img_rect.x0, "y0": img_rect.y0,
                "x1": img_rect.x1, "y1": img_rect.y1,
            }
            break

        images_info.append({
            "path": str(img_path.as_posix()),
            "bbox": bbox,
            "index": img_index,
            "width": width,
            "height": height,
        })

    return images_info


# ---------------------------------------------------------------------------
# Question boundary detection
# ---------------------------------------------------------------------------

# Patterns that indicate a new question
_QUESTION_PATTERNS = [
    # "1." "2." "10." "11." at start of line
    re.compile(r"^(\d{1,2})\.\s+(.+)", re.MULTILINE),
    # "1)" "2)" etc.
    re.compile(r"^(\d{1,2})\)\s+(.+)", re.MULTILINE),
    # "Q1." "Q.1" "Q 1."
    re.compile(r"^Q\.?\s*(\d{1,2})\.?\s+(.+)", re.MULTILINE | re.IGNORECASE),
]

# Pattern for marks: (8), (7), [10], (2 x 5 = 10), etc.
_MARKS_PATTERN = re.compile(
    r"\((\d{1,2})\)\s*$"
    r"|"
    r"\[(\d{1,2})\s*(?:marks?)?\]\s*$"
    r"|"
    r"\((\d{1,2})\s*(?:marks?|Marks?)\)\s*$",
    re.MULTILINE,
)

# Marks from structured patterns like "5 x 2 = 10 Marks" or "10 x 2 =20 Marks"
_BULK_MARKS_PATTERN = re.compile(
    r"(\d{1,2})\s*[xX×]\s*(\d{1,2})\s*=\s*(\d{1,3})\s*[Mm]arks?",
)

# Part headers
_PART_PATTERN = re.compile(
    r"^(?:PART|Part)\s*[-–—:]?\s*([A-C])\b",
    re.MULTILINE,
)

# MCQ options
_MCQ_OPTION_PATTERN = re.compile(
    r"^\s*([A-Da-d])\s*[.)]\s+(.+)",
    re.MULTILINE,
)

# Sub-question pattern (a), (b), (i), (ii)
_SUBQ_PATTERN = re.compile(
    r"^\s*\(([a-d]|[iv]{1,3})\)\s+(.+)",
    re.MULTILINE,
)

# Diagram reference keywords
_DIAGRAM_KEYWORDS = [
    "circuit", "diagram", "figure", "fig.", "shown below", "shown above",
    "given below", "given network", "following figure", "following circuit",
    "following diagram", "sketch", "draw", "graph", "plot", "table",
    "network", "shown in", "refer to", "the figure", "the diagram",
    "flowchart", "block diagram", "state diagram", "truth table",
]


def detect_marks_for_line(text: str) -> int | None:
    """Try to extract marks from the end of a question line or nearby."""
    m = _MARKS_PATTERN.search(text)
    if m:
        for g in m.groups():
            if g is not None:
                try:
                    return int(g)
                except ValueError:
                    pass
    return None


def detect_per_question_marks(header_text: str) -> int | None:
    """Detect per-question marks from bulk patterns like '10 x 2 = 20 Marks'."""
    m = _BULK_MARKS_PATTERN.search(header_text)
    if m:
        try:
            return int(m.group(2))  # The per-question marks
        except (ValueError, IndexError):
            pass
    return None


def classify_question_type(text: str) -> str | None:
    """Classify question as MCQ, numerical, proof, descriptive, etc."""
    text_lower = text.lower()
    if _MCQ_OPTION_PATTERN.search(text):
        return "mcq"
    if any(kw in text_lower for kw in [
        "calculate", "compute", "find the value", "evaluate",
        "determine the", "solve", "how many", "what is the value",
    ]):
        return "numerical"
    if any(kw in text_lower for kw in [
        "prove that", "prove:", "show that", "derive",
    ]):
        return "proof"
    if any(kw in text_lower for kw in [
        "define", "state", "what is", "what are", "list",
        "mention", "name", "write short notes",
    ]):
        return "short_answer"
    if any(kw in text_lower for kw in [
        "explain", "discuss", "describe", "elaborate",
        "compare", "differentiate", "distinguish",
    ]):
        return "descriptive"
    if any(kw in text_lower for kw in [
        "draw", "sketch", "plot",
    ]):
        return "diagram"
    return None


def has_diagram_reference(text: str) -> bool:
    """Check if question text references an image/diagram."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in _DIAGRAM_KEYWORDS)


def split_into_questions(page_text: str) -> list[dict]:
    """Split page text into individual questions.

    Returns a list of dicts with keys: number, text, marks, start_pos, end_pos.
    """
    if not page_text.strip():
        return []

    # Find all question starts
    question_starts: list[tuple[int, str, str]] = []  # (position, number, rest_of_line)

    for pattern in _QUESTION_PATTERNS:
        for m in pattern.finditer(page_text):
            question_starts.append((m.start(), m.group(1), m.group(2)))

    if not question_starts:
        # No numbered questions found; treat entire text as one block
        return [{
            "number": "1",
            "text": page_text.strip(),
            "marks": detect_marks_for_line(page_text),
            "start_pos": 0,
            "end_pos": len(page_text),
        }]

    # Sort by position
    question_starts.sort(key=lambda x: x[0])

    # Deduplicate: if two patterns match the same position, keep one
    seen_positions: set[int] = set()
    unique_starts: list[tuple[int, str, str]] = []
    for pos, num, rest in question_starts:
        # Merge nearby positions (within 5 chars)
        if not any(abs(pos - sp) < 5 for sp in seen_positions):
            unique_starts.append((pos, num, rest))
            seen_positions.add(pos)

    questions: list[dict] = []
    for i, (pos, num, _rest) in enumerate(unique_starts):
        # End position is the start of the next question, or end of text
        end_pos = unique_starts[i + 1][0] if i + 1 < len(unique_starts) else len(page_text)
        q_text = page_text[pos:end_pos].strip()

        marks = detect_marks_for_line(q_text)

        questions.append({
            "number": num,
            "text": q_text,
            "marks": marks,
            "start_pos": pos,
            "end_pos": end_pos,
        })

    return questions


def extract_mcq_options(text: str) -> list[str] | None:
    """Extract MCQ options from question text."""
    options = []
    for m in _MCQ_OPTION_PATTERN.finditer(text):
        options.append(f"{m.group(1).upper()}. {m.group(2).strip()}")
    return options if len(options) >= 2 else None


# ---------------------------------------------------------------------------
# Image description generation (from OCR — no vision model)
# ---------------------------------------------------------------------------

def generate_image_description(
    image_path: Path,
    question_text: str,
    tesseract_cmd: str = "tesseract",
) -> str:
    """Generate a textual description of a diagram image.

    Uses OCR to extract labels/values and combines with question context
    to produce a useful description for retrieval.
    """
    try:
        ocr_text = run_tesseract_ocr(image_path, tesseract_cmd)
    except RuntimeError:
        ocr_text = ""

    if not ocr_text.strip():
        if has_diagram_reference(question_text):
            return "Diagram or figure associated with this question (no text labels detected)."
        return ""

    # Extract meaningful tokens from OCR
    tokens = ocr_text.split()

    # Look for common diagram elements
    description_parts: list[str] = []

    # Detect diagram type from question context
    q_lower = question_text.lower()
    if "circuit" in q_lower or "resistance" in q_lower or "resistor" in q_lower:
        description_parts.append("Electrical circuit diagram")
    elif "graph" in q_lower or "plot" in q_lower:
        description_parts.append("Graph or plot")
    elif "flowchart" in q_lower:
        description_parts.append("Flowchart")
    elif "table" in q_lower:
        description_parts.append("Table")
    elif "block diagram" in q_lower:
        description_parts.append("Block diagram")
    elif "state diagram" in q_lower:
        description_parts.append("State diagram")
    elif "truth table" in q_lower:
        description_parts.append("Truth table")
    else:
        description_parts.append("Diagram or figure")

    # Extract values with units from OCR
    unit_pattern = re.compile(
        r"(\d+\.?\d*)\s*(Ω|ohm|V|A|mA|kΩ|MΩ|µF|nF|pF|mH|µH|H|Hz|kHz|MHz"
        r"|kg|m|cm|mm|N|kN|Pa|MPa|GPa|J|kJ|W|kW|mol|g|L|mL|°C|K|%|s|ms)",
        re.IGNORECASE,
    )
    values = unit_pattern.findall(ocr_text)
    if values:
        value_strs = [f"{v}{u}" for v, u in values]
        description_parts.append(f"containing values: {', '.join(value_strs)}")

    # Extract labels (capitalized words that aren't common English)
    label_pattern = re.compile(r"\b([A-Z][a-z]*(?:\s[A-Z][a-z]*)*)\b")
    common_words = {"The", "This", "That", "For", "And", "But", "With", "From", "Not"}
    labels = [
        m.group(1) for m in label_pattern.finditer(ocr_text)
        if m.group(1) not in common_words and len(m.group(1)) > 1
    ]
    if labels:
        unique_labels = list(dict.fromkeys(labels))[:10]  # Keep order, limit count
        description_parts.append(f"with labels: {', '.join(unique_labels)}")

    return ". ".join(description_parts) + "." if description_parts else ""


# ---------------------------------------------------------------------------
# Core PDF processing
# ---------------------------------------------------------------------------

def process_pdf(
    pdf_path: Path,
    image_output_dir: Path,
    tesseract_cmd: str,
    ocr_dpi: int = 300,
    min_native_chars: int = 80,
    stats: IngestionStats | None = None,
) -> list[ExtractedQuestion]:
    """Process a single PDF and extract all questions."""
    if stats is None:
        stats = IngestionStats()

    paper_id = safe_stem(pdf_path)
    paper_image_dir = image_output_dir / paper_id
    file_metadata = parse_filename_metadata(pdf_path)

    questions: list[ExtractedQuestion] = []
    seen_ids: set[str] = set()
    exam_type = None  # Detected from first page header

    doc = pymupdf.open(pdf_path)

    try:
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_number = page_idx + 1
            stats.pages_processed += 1

            # ---- Extract text ----
            native_text = normalise_text(page.get_text("text"))
            page_text = native_text
            ocr_text_full = ""
            is_scanned = False

            if len(native_text) < min_native_chars:
                # Page is scanned — OCR it
                is_scanned = True
                page_img_path = paper_image_dir / f"p{page_number:03d}_full.png"
                render_page_image(page, page_img_path, dpi=ocr_dpi)
                stats.ocr_operations += 1
                try:
                    ocr_text_full = run_tesseract_ocr(
                        page_img_path, tesseract_cmd
                    )
                    page_text = ocr_text_full
                except RuntimeError as e:
                    stats.errors.append(
                        f"OCR failed: {pdf_path.name} page {page_number}: {e}"
                    )
                    page_text = native_text  # Fall back to whatever we have

            # Detect exam type from first page
            if page_number == 1:
                exam_type = detect_exam_type(page_text)

            # ---- Extract embedded images (non-scanned pages) ----
            page_images: list[dict] = []
            if not is_scanned:
                page_images = extract_page_images(
                    page, doc, paper_image_dir, paper_id, page_number
                )
                stats.images_extracted += len(page_images)

            # ---- Split into questions ----
            raw_questions = split_into_questions(page_text)

            if not raw_questions:
                stats.boundary_failures += 1
                continue

            # If we only got one "question" covering the whole page and it's
            # very long, it's likely a boundary detection failure
            if (len(raw_questions) == 1
                    and len(raw_questions[0]["text"]) > 2000
                    and "1" not in raw_questions[0].get("number", "")):
                stats.boundary_failures += 1

            for rq in raw_questions:
                q_number = rq["number"]
                q_text = rq["text"]
                q_marks = rq["marks"]

                # Generate deterministic ID
                q_id = deterministic_id(pdf_path.name, page_number, q_number)
                if q_id in seen_ids:
                    stats.duplicates_detected += 1
                    continue
                seen_ids.add(q_id)

                # ---- Associate images with this question ----
                q_image_paths: list[str] = []
                q_has_image = False
                q_image_desc = ""

                # For scanned pages: if question references a diagram,
                # the full page image is the reference
                if is_scanned and has_diagram_reference(q_text):
                    page_img_path = paper_image_dir / f"p{page_number:03d}_full.png"
                    if page_img_path.exists():
                        q_image_paths.append(str(page_img_path.as_posix()))
                        q_has_image = True
                        q_image_desc = generate_image_description(
                            page_img_path, q_text, tesseract_cmd
                        )
                        stats.images_extracted += 1

                # For native-text pages: associate nearby images
                if page_images and not is_scanned:
                    # Simple heuristic: associate images whose vertical position
                    # falls within or near this question's text span
                    q_start_y = rq["start_pos"] / max(len(page_text), 1) * page.rect.height
                    q_end_y = rq["end_pos"] / max(len(page_text), 1) * page.rect.height

                    for img_info in page_images:
                        bbox = img_info.get("bbox")
                        if bbox:
                            img_mid_y = (bbox["y0"] + bbox["y1"]) / 2
                            # Associate if image center is within question's span
                            # (with some margin)
                            margin = page.rect.height * 0.1
                            if q_start_y - margin <= img_mid_y <= q_end_y + margin:
                                q_image_paths.append(img_info["path"])
                                q_has_image = True
                        else:
                            # No bbox info; associate with the question if it
                            # references a diagram
                            if has_diagram_reference(q_text):
                                q_image_paths.append(img_info["path"])
                                q_has_image = True

                    # Generate image descriptions for associated images
                    if q_has_image and q_image_paths:
                        descs = []
                        for ip in q_image_paths:
                            desc = generate_image_description(
                                Path(ip), q_text, tesseract_cmd
                            )
                            if desc:
                                descs.append(desc)
                        q_image_desc = " ".join(descs)

                if q_has_image:
                    stats.questions_with_images += 1

                # ---- MCQ options ----
                options = extract_mcq_options(q_text)

                # ---- Question type ----
                q_type = classify_question_type(q_text)

                # ---- Build full text for embedding ----
                full_text_parts = [q_text]
                if q_image_desc:
                    full_text_parts.append(f"[Diagram: {q_image_desc}]")
                full_text = "\n".join(full_text_parts)

                # ---- Create structured question ----
                question = ExtractedQuestion(
                    question_id=q_id,
                    question_number=q_number,
                    question_text=q_text,
                    ocr_text=ocr_text_full if is_scanned else "",
                    full_text=full_text,
                    has_image=q_has_image,
                    image_paths=q_image_paths,
                    image_description=q_image_desc,
                    subject=file_metadata["subject"],
                    unit=None,  # Would need syllabus mapping to detect
                    exam=exam_type or file_metadata.get("exam_period"),
                    year=file_metadata["year"],
                    marks=q_marks,
                    question_type=q_type,
                    source_file=pdf_path.name,
                    page_number=page_number,
                    options=options,
                )
                questions.append(question)
                stats.questions_extracted += 1
    finally:
        doc.close()

    stats.pdfs_processed += 1
    return questions


# ---------------------------------------------------------------------------
# Full ingestion pipeline
# ---------------------------------------------------------------------------

def ingest_question_papers(
    papers_dir: Path | str = Path("campus_rag_cia_papers"),
    output_dir: Path | str = Path("data"),
    tesseract_cmd: str | None = None,
    ocr_dpi: int = 300,
    min_native_chars: int = 80,
) -> tuple[list[ExtractedQuestion], IngestionStats]:
    """Run the full ingestion pipeline over all PDFs in the papers directory.

    Returns all extracted questions and ingestion statistics.
    """
    papers_dir = Path(papers_dir).resolve()
    output_dir = Path(output_dir).resolve()
    image_output_dir = output_dir / "question_images"
    manifest_path = output_dir / "questions_extracted.jsonl"

    if tesseract_cmd is None:
        tesseract_cmd = shutil.which("tesseract")
    if not tesseract_cmd:
        print("WARNING: Tesseract not found. Scanned pages will not be OCR'd.",
              file=sys.stderr)

    pdf_files = sorted(papers_dir.rglob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found under {papers_dir}", file=sys.stderr)
        return [], IngestionStats()

    print(f"Found {len(pdf_files)} PDFs in {papers_dir}")
    print(f"Image output: {image_output_dir}")
    print(f"Manifest: {manifest_path}")
    print()

    stats = IngestionStats()
    all_questions: list[ExtractedQuestion] = []

    for i, pdf_path in enumerate(pdf_files, start=1):
        print(f"[{i}/{len(pdf_files)}] Processing: {pdf_path.relative_to(papers_dir)}")
        try:
            questions = process_pdf(
                pdf_path=pdf_path,
                image_output_dir=image_output_dir,
                tesseract_cmd=tesseract_cmd,
                ocr_dpi=ocr_dpi,
                min_native_chars=min_native_chars,
                stats=stats,
            )
            all_questions.extend(questions)
            print(f"         → {len(questions)} questions extracted")
        except Exception as e:
            stats.errors.append(f"Failed to process {pdf_path.name}: {e}")
            print(f"         → ERROR: {e}", file=sys.stderr)

    # Write manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        for q in all_questions:
            record = asdict(q)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Write stats
    stats_path = output_dir / "question_extraction_stats.json"
    stats_dict = asdict(stats)
    stats_path.write_text(json.dumps(stats_dict, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print(f"PDFs processed:            {stats.pdfs_processed}")
    print(f"Pages processed:           {stats.pages_processed}")
    print(f"Questions extracted:        {stats.questions_extracted}")
    print(f"Questions with images:      {stats.questions_with_images}")
    print(f"Images extracted:           {stats.images_extracted}")
    print(f"OCR operations:            {stats.ocr_operations}")
    print(f"Boundary detection issues:  {stats.boundary_failures}")
    print(f"Duplicates detected:        {stats.duplicates_detected}")
    if stats.errors:
        print(f"Errors:                    {len(stats.errors)}")
        for err in stats.errors[:10]:
            print(f"  - {err}")
    print(f"\nManifest saved to: {manifest_path}")
    print(f"Stats saved to:    {stats_path}")

    return all_questions, stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract questions from CIA exam papers for the Exam Prep RAG."
    )
    parser.add_argument(
        "--papers-dir",
        type=Path,
        default=Path("campus_rag_cia_papers"),
        help="directory containing subject sub-folders with PDFs (default: campus_rag_cia_papers)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="base output directory (default: data)",
    )
    parser.add_argument(
        "--tesseract-cmd",
        help="path to tesseract executable (auto-detected if on PATH)",
    )
    parser.add_argument(
        "--ocr-dpi",
        type=int,
        default=300,
        help="DPI for rendering scanned pages (default: 300)",
    )
    parser.add_argument(
        "--min-native-chars",
        type=int,
        default=80,
        help="pages below this native text count are OCR'd (default: 80)",
    )
    args = parser.parse_args()

    _questions, stats = ingest_question_papers(
        papers_dir=args.papers_dir,
        output_dir=args.output_dir,
        tesseract_cmd=args.tesseract_cmd,
        ocr_dpi=args.ocr_dpi,
        min_native_chars=args.min_native_chars,
    )
    return 1 if stats.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
