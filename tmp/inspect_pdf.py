"""Quick inspection of sample CIA papers to understand their structure."""
import pymupdf
from pathlib import Path

papers_dir = Path("campus_rag_cia_papers")
pdfs = sorted(papers_dir.rglob("*.pdf"))
print(f"Total PDFs: {len(pdfs)}")
print()

# Inspect a few papers from different subjects
samples = [
    "basic_electrical_engineering/EEE101 FEB 2023.pdf",
    "basics_electronics/EIE101R01 DEC 2023.pdf",
    "engineering_mathematics_1/M1 DEC 2024.pdf",
    "physics/PHY101R01 MAY 2024.pdf",
]

for s in samples:
    fp = papers_dir / s
    if not fp.exists():
        print(f"MISSING: {s}")
        continue
    doc = pymupdf.open(fp)
    print(f"=== {fp.name} ({len(doc)} pages) ===")
    for i, page in enumerate(doc):
        text = page.get_text("text")
        images = page.get_images(full=True)
        print(f"  Page {i+1}: {len(text)} chars text, {len(images)} images")
        if i == 0:
            print(f"  First 500 chars:\n{text[:500]}")
            print("  ---")
    doc.close()
    print()
