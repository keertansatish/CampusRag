"""Chunk extracted pages and persist them in a local Chroma vector database."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping chunks, preferring to end at whitespace."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be at least zero and smaller than chunk_size")

    text = " ".join(text.split())
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            whitespace = text.rfind(" ", start, end)
            if whitespace > start:
                end = whitespace

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        # Move the overlap start to a word boundary so chunks never begin with
        # a truncated token. Prefer retaining the preceding whole word.
        next_start = end - chunk_overlap
        boundary = text.rfind(" ", start, next_start)
        if boundary >= start:
            next_start = boundary + 1
        elif next_start <= start:
            next_start = end
        start = next_start

    return chunks


def iter_chunks(manifest_path: Path, chunk_size: int, chunk_overlap: int) -> Iterator[tuple[str, str, dict[str, object]]]:
    """Yield deterministic chunk IDs, text, and source-citation metadata."""
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                page = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number} of {manifest_path}") from error

            source_file = str(page.get("source_file", ""))
            page_number = page.get("page_number")
            text = str(page.get("text", "")).strip()
            if not source_file or not isinstance(page_number, int) or not text:
                continue

            for chunk_number, chunk in enumerate(split_text(text, chunk_size, chunk_overlap), start=1):
                # The digest prevents unsafe characters and keeps IDs stable across runs.
                fingerprint = hashlib.sha256(f"{source_file}|{page_number}|{chunk_number}".encode()).hexdigest()
                metadata: dict[str, object] = {
                    "source_file": source_file,
                    "source_path": str(page.get("source_path", "")),
                    "page_number": page_number,
                    "chunk_number": chunk_number,
                    "extraction_method": str(page.get("extraction_method", "")),
                }
                yield fingerprint, chunk, metadata


def batched[T](items: Iterator[T], batch_size: int) -> Iterator[list[T]]:
    """Yield an iterator in bounded-size lists for efficient database writes."""
    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_index(
    manifest_path: Path,
    database_path: Path,
    collection_name: str,
    chunk_size: int = 1_000,
    chunk_overlap: int = 150,
    batch_size: int = 100,
    reset: bool = False,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> int:
    """Embed chunks from ``pages.jsonl`` and save them in a persistent Chroma collection."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    import chromadb
    from chromadb.errors import NotFoundError
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    database_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(database_path))
    if reset:
        try:
            client.delete_collection(collection_name)
        except NotFoundError:
            pass  # The collection does not exist yet.

    embeddings = SentenceTransformerEmbeddingFunction(model_name=embedding_model)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embeddings,
        metadata={"hnsw:space": "cosine"},
    )

    total = 0
    for batch in batched(iter_chunks(manifest_path, chunk_size, chunk_overlap), batch_size):
        ids, documents, metadatas = zip(*batch)
        collection.upsert(ids=list(ids), documents=list(documents), metadatas=list(metadatas))
        total += len(batch)
    return total


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chunk extracted CampusRAG pages into a persistent Chroma database.")
    parser.add_argument("--manifest", type=Path, default=Path("data/extracted/pages.jsonl"), help="extraction manifest to index")
    parser.add_argument("--database-dir", type=Path, default=Path("data/chroma"), help="directory for the local Chroma database")
    parser.add_argument("--collection", default="campus_policy", help="Chroma collection name")
    parser.add_argument("--chunk-size", type=int, default=1_000, help="maximum characters per chunk (default: 1000)")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="overlap between chunks in characters (default: 150)")
    parser.add_argument("--batch-size", type=int, default=100, help="chunks written per database call (default: 100)")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="Sentence Transformers model name")
    parser.add_argument("--reset", action="store_true", help="delete the existing collection before indexing")
    parser.add_argument("--dry-run", action="store_true", help="validate and count chunks without loading the model or writing a database")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.dry_run:
        count = sum(1 for _ in iter_chunks(args.manifest, args.chunk_size, args.chunk_overlap))
        print(f"Validated {count} chunks from {args.manifest}.")
        return 0

    count = build_index(
        manifest_path=args.manifest,
        database_path=args.database_dir,
        collection_name=args.collection,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
        reset=args.reset,
        embedding_model=args.embedding_model,
    )
    print(f"Stored {count} chunks in collection '{args.collection}' at {args.database_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
