"""Extract college-policy PDFs into retrieval-ready Markdown, CSV, and JSONL.

The extractor keeps native PDF text whenever it is present and uses Tesseract
only for image-only pages. Each extracted section retains its original PDF
name and page number so every future RAG answer can cite its source.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import pymupdf


@dataclass
class PageRecord:
    source_file: str
    source_path: str
    page_number: int
    extraction_method: str
    text: str
    table_files: list[str]
    ocr_image: str | None = None
    ocr_layout: str | None = None


def safe_stem(path: Path) -> str:
    """Return a stable, filesystem-safe document identifier."""
    return re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")


def normalise_text(text: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def csv_cell(value: object) -> str:
    return "" if value is None else normalise_text(str(value)).replace("\n", " ")


def markdown_table(rows: list[list[object]]) -> str:
    """Represent a table in Markdown for text embeddings and human review."""
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    cleaned = [[csv_cell(cell).replace("|", "\\|") for cell in row] for row in rows]
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]
    lines = [
        "| " + " | ".join(cleaned[0]) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in cleaned[1:])
    return "\n".join(lines)


def write_table(rows: list[list[object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = max((len(row) for row in rows), default=0)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for row in rows:
            writer.writerow([csv_cell(cell) for cell in row] + [""] * (width - len(row)))


def extract_tables(page: pymupdf.Page, table_dir: Path, document_id: str, page_number: int) -> tuple[list[str], list[str]]:
    """Save vector tables as CSV and return their relative paths and Markdown."""
    try:
        tables = page.find_tables().tables
    except Exception:
        return [], []  # Table detection must not block exporting page text.

    files: list[str] = []
    markdown: list[str] = []
    for number, table in enumerate(tables, start=1):
        rows = table.extract()
        if not rows:
            continue
        filename = f"{document_id}-p{page_number:03d}-table{number:02d}.csv"
        write_table(rows, table_dir / filename)
        files.append(f"tables/{filename}")
        markdown.append(f"Table {number}\n\n{markdown_table(rows)}")
    return files, markdown


def _clean_tesseract_rows(rows: csv.DictReader, min_confidence: float) -> str:
    """Filter Tesseract word rows and join their text in TSV reading order."""
    tokens: list[str] = []
    for row in rows:
        confidence_value = (row.get("conf") or "-1").strip()

        # Tesseract uses -1 for page, block, paragraph, and line structure.
        if confidence_value == "-1":
            continue
        try:
            confidence = float(confidence_value)
        except ValueError:
            continue  # Ignore malformed confidence values.

        if confidence < min_confidence:
            continue

        token = (row.get("text") or "").strip()
        if token:
            tokens.append(token)

    return " ".join(tokens)


def clean_tesseract_tsv(tsv_path: Path | str, min_confidence: float = 70) -> str:
    """Return confidence-filtered OCR text from a Tesseract TSV file.

    The returned string is ready to pass directly to a text chunker.
    """
    with Path(tsv_path).open("r", encoding="utf-8", newline="") as handle:
        return _clean_tesseract_rows(csv.DictReader(handle, delimiter="\t"), min_confidence)


def tsv_to_text(tsv: str, min_confidence: float = 70) -> str:
    """Clean in-memory Tesseract TSV output using the same file-parser rules."""
    return _clean_tesseract_rows(csv.DictReader(io.StringIO(tsv), delimiter="\t"), min_confidence)


def run_ocr(image_path: Path, tesseract_cmd: str, language: str, min_confidence: float) -> tuple[str, str]:
    """OCR a page and retain its TSV coordinates for scanned-table recovery."""
    # Let Tesseract determine the page layout.  These source PDFs contain a
    # letterhead and a multi-column table; treating each page as one uniform
    # block (PSM 6) turns borders and decorative marks into stray text.
    command = [tesseract_cmd, str(image_path), "stdout", "-l", language, "--psm", "3", "tsv"]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "Tesseract returned no details.")
    return tsv_to_text(result.stdout, min_confidence), result.stdout


def render_page(page: pymupdf.Page, image_path: Path, dpi: int) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    scale = dpi / 72
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), colorspace=pymupdf.csGRAY, alpha=False)
    pixmap.save(image_path)


def page_markdown(record: PageRecord, table_markdown: list[str]) -> str:
    sections = [f"# {Path(record.source_file).stem}", f"Source page: {record.page_number}", "", record.text]
    if table_markdown:
        sections.extend(["", "## Extracted tables", "", "\n\n".join(table_markdown)])
    return "\n".join(sections).strip() + "\n"


def extract_pdf(
    pdf_path: Path,
    output_dir: Path,
    min_native_chars: int,
    ocr_dpi: int,
    tesseract_cmd: str | None,
    language: str,
    ocr_min_confidence: float,
) -> Iterator[tuple[PageRecord, str]]:
    """Yield page records and their Markdown, applying OCR only when needed."""
    document_id = safe_stem(pdf_path)
    document_dir = output_dir / "documents" / document_id
    table_dir = output_dir / "tables"
    image_dir = output_dir / "ocr-images" / document_id
    layout_dir = output_dir / "ocr-layout" / document_id
    document_dir.mkdir(parents=True, exist_ok=True)

    with pymupdf.open(pdf_path) as document:
        for index, page in enumerate(document, start=1):
            native_text = normalise_text(page.get_text("text"))
            table_files, table_markdown = extract_tables(page, table_dir, document_id, index)
            method, ocr_image, ocr_layout, text = "native_text", None, None, native_text

            if len(native_text) < min_native_chars:
                if not tesseract_cmd:
                    method = "ocr_required"
                    text = "[Image-based page. Install Tesseract or pass --tesseract-cmd, then rerun.]"
                else:
                    image_path = image_dir / f"page-{index:03d}.png"
                    render_page(page, image_path, ocr_dpi)
                    try:
                        text, layout_tsv = run_ocr(image_path, tesseract_cmd, language, ocr_min_confidence)
                        method = "ocr"
                        ocr_image = str(image_path.relative_to(output_dir).as_posix())
                        layout_path = layout_dir / f"page-{index:03d}.tsv"
                        layout_path.parent.mkdir(parents=True, exist_ok=True)
                        layout_path.write_text(layout_tsv, encoding="utf-8")
                        ocr_layout = str(layout_path.relative_to(output_dir).as_posix())
                    except RuntimeError as error:
                        method = "ocr_failed"
                        text = f"[OCR failed on this image-based page: {error}]"

            record = PageRecord(
                source_file=pdf_path.name,
                source_path=str(pdf_path.resolve()),
                page_number=index,
                extraction_method=method,
                text=text,
                table_files=table_files,
                ocr_image=ocr_image,
                ocr_layout=ocr_layout,
            )
            markdown = page_markdown(record, table_markdown)
            (document_dir / f"page-{index:03d}.md").write_text(markdown, encoding="utf-8")
            yield record, markdown


def extract_collection(args: argparse.Namespace) -> int:
    input_dir, output_dir = Path(args.input_dir).resolve(), Path(args.output_dir).resolve()
    pdf_files = sorted(input_dir.rglob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found under {input_dir}", file=sys.stderr)
        return 2

    executable = args.tesseract_cmd or shutil.which("tesseract")
    output_dir.mkdir(parents=True, exist_ok=True)
    methods: dict[str, int] = {}
    with (output_dir / "pages.jsonl").open("w", encoding="utf-8") as manifest:
        for pdf_path in pdf_files:
            for record, _ in extract_pdf(
                pdf_path,
                output_dir,
                args.min_native_chars,
                args.ocr_dpi,
                executable,
                args.ocr_language,
                args.ocr_min_confidence,
            ):
                manifest.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
                methods[record.extraction_method] = methods.get(record.extraction_method, 0) + 1

    summary = {"input_directory": str(input_dir), "pdf_count": len(pdf_files), "page_count": sum(methods.values()), "extraction_methods": methods, "manifest": "pages.jsonl"}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if methods.get("ocr_required") or methods.get("ocr_failed"):
        print("OCR action needed: install Tesseract or pass --tesseract-cmd with its executable path.", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract PDFs into RAG-ready, page-cited text and tables.")
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract", help="extract all PDFs in a directory")
    extract.add_argument("--input-dir", default="RAG", help="directory containing PDFs (default: RAG)")
    extract.add_argument("--output-dir", default="data/extracted", help="destination for extracted artifacts")
    extract.add_argument("--min-native-chars", type=int, default=80, help="OCR pages below this native-text count")
    extract.add_argument("--ocr-dpi", type=int, default=300, help="DPI used when rendering scanned pages")
    extract.add_argument("--ocr-language", default="eng", help="Tesseract language code (default: eng)")
    extract.add_argument(
        "--ocr-min-confidence",
        type=float,
        default=70,
        help="minimum Tesseract word confidence to retain (default: 70)",
    )
    extract.add_argument("--tesseract-cmd", help="path to tesseract.exe if it is not on PATH")
    extract.set_defaults(handler=extract_collection)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
