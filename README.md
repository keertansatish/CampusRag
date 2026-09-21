# CampusRAG PDF extraction

This project turns college-policy PDFs into source-cited, RAG-ready content. It
preserves selectable text, exports vector tables as CSV plus Markdown, and OCRs
scanned pages such as `Anti_Ragging.pdf`.

## Run the extractor

```powershell
.\.venv\Scripts\python.exe main.py extract
```

The outputs are written to `data/extracted/`:

- `documents/<pdf-name>/page-XXX.md` - one retrieval document per source page.
- `tables/*.csv` - machine-readable copies of detected vector tables.
- `pages.jsonl` - page text, source filename, page number, OCR status, and table links.
- `ocr-images/` - 300-DPI renders retained for OCR auditability (only scanned pages).
- `ocr-layout/` - Tesseract TSV word coordinates for scanned tables and layouts.
- `summary.json` - counts and extraction status.

## OCR setup for scanned PDFs

Install the **Tesseract OCR** Windows application and ensure `tesseract.exe` is
on `PATH`. If it is installed elsewhere, supply the executable explicitly:

```powershell
.\.venv\Scripts\python.exe main.py extract --tesseract-cmd "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

The command exits non-zero if a scanned page could not be OCRed, so incomplete
collections cannot quietly enter the RAG index. Rerun it after fixing OCR.

## Use this in the RAG stage

Chunk `pages.jsonl` by `text` while carrying `source_file` and `page_number`
into each chunk's metadata. Retrieve the page and table CSVs as evidence, then
show citations such as `Examination_Regulations_2026_U.pdf, p. 4` in answers.

## Build the vector database

After extraction, create a local persistent Chroma database from every page in
`pages.jsonl`:

```powershell
.\.venv\Scripts\python.exe rag_pipeline\indexing.py --reset
```

The index is saved in `data/chroma/`, in the `campus_policy` collection. Each
chunk retains its source filename, original page number, chunk number, and
extraction method for citations at retrieval time. The first run downloads the
`all-MiniLM-L6-v2` embedding model. To validate chunking without writing the
database, use `--dry-run`.

## Retrieve the top three chunks

Pass the user's question as the positional argument:

```powershell
.\.venv\Scripts\python.exe rag_pipeline\retrieve.py "What is the penalty for ragging?"
```

The command returns the three closest chunks, along with their source PDF and
page number. Use their text as RAG context and their metadata for citations.

## Generate a grounded Groq answer

Create a `.env` file in the project root using `.env.example`, then add your
Groq key. The answerer loads this file automatically, retrieves the top three
chunks, sends only those chunks plus the question to the model, and prints the
response with source citations.

```text
GROQ_API_KEY=your-groq-api-key
LLM_MODEL=openai/gpt-oss-20b
```

Then run:

```powershell
.\.venv\Scripts\python.exe rag_pipeline\answer.py "What is the minimum attendance requirement?"
```

The key is loaded from `.env`, which is excluded from version control. If
`LLM_MODEL` is omitted, the app defaults to `openai/gpt-oss-20b`.
