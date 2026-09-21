"""Index extracted questions into a separate ChromaDB 'questions' collection.

Each question is stored as ONE vector document (not chunked).
The embedding text combines question text + options + image description.
All metadata is stored for filtering and retrieval.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_embedding_text(question: dict) -> str:
    """Combine question fields into a single string for embedding.

    Keeps the embedding focused on semantically useful information:
    question text, MCQ options, and image description.
    Does NOT include raw OCR dumps or noise.
    """
    parts: list[str] = []

    # Primary: question text
    q_text = question.get("question_text", "").strip()
    if q_text:
        parts.append(q_text)

    # MCQ options
    options = question.get("options")
    if options:
        parts.append("Options: " + " | ".join(options))

    # Image description (useful for semantic search)
    img_desc = question.get("image_description", "").strip()
    if img_desc:
        parts.append(f"Diagram: {img_desc}")

    return "\n".join(parts)


def build_metadata(question: dict) -> dict[str, object]:
    """Build ChromaDB-compatible metadata from a question record.

    ChromaDB metadata values must be str, int, float, or bool.
    Lists and None values need conversion.
    """
    meta: dict[str, object] = {}

    # String fields
    for key in [
        "question_id", "question_number", "question_text", "ocr_text",
        "full_text", "image_description", "source_file",
    ]:
        val = question.get(key, "")
        meta[key] = val if val is not None else ""

    # Nullable string fields
    for key in ["subject", "unit", "exam", "year", "question_type"]:
        val = question.get(key)
        meta[key] = val if val is not None else ""

    # Boolean
    meta["has_image"] = bool(question.get("has_image", False))

    # Integer
    marks = question.get("marks")
    meta["marks"] = marks if isinstance(marks, int) else -1  # -1 = unknown

    # Page number
    meta["page_number"] = question.get("page_number", 0)

    # Lists → JSON strings
    image_paths = question.get("image_paths", [])
    meta["image_paths"] = json.dumps(image_paths) if image_paths else "[]"

    options = question.get("options")
    meta["options"] = json.dumps(options) if options else "[]"

    return meta


def build_questions_index(
    manifest_path: Path,
    database_path: Path,
    collection_name: str = "questions",
    embedding_model: str = "all-MiniLM-L6-v2",
    batch_size: int = 100,
    reset: bool = False,
) -> int:
    """Read questions from the JSONL manifest and index them into ChromaDB.

    Uses upsert so re-running is idempotent — no duplicates.
    """
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
            print(f"Deleted existing collection '{collection_name}'.")
        except (NotFoundError, Exception):
            pass

    embeddings = SentenceTransformerEmbeddingFunction(model_name=embedding_model)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embeddings,
        metadata={"hnsw:space": "cosine"},
    )

    # Read all questions from manifest
    questions: list[dict] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                questions.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"WARNING: Invalid JSON on line {line_num}: {e}")

    if not questions:
        print("No questions found in manifest.")
        return 0

    print(f"Indexing {len(questions)} questions into '{collection_name}'...")

    # Batch upsert
    total = 0
    for i in range(0, len(questions), batch_size):
        batch = questions[i : i + batch_size]
        ids = [q["question_id"] for q in batch]
        documents = [build_embedding_text(q) for q in batch]
        metadatas = [build_metadata(q) for q in batch]

        # Filter out empty documents
        valid = [
            (id_, doc, meta)
            for id_, doc, meta in zip(ids, documents, metadatas)
            if doc.strip()
        ]
        if not valid:
            continue

        v_ids, v_docs, v_metas = zip(*valid)
        collection.upsert(
            ids=list(v_ids),
            documents=list(v_docs),
            metadatas=list(v_metas),
        )
        total += len(valid)

    print(f"Stored {total} questions in collection '{collection_name}' at {database_path}.")
    return total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Index extracted exam questions into the 'questions' ChromaDB collection."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/questions_extracted.jsonl"),
        help="path to the questions JSONL manifest (default: data/questions_extracted.jsonl)",
    )
    parser.add_argument(
        "--database-dir",
        type=Path,
        default=Path("data/chroma"),
        help="ChromaDB persistent directory (default: data/chroma)",
    )
    parser.add_argument(
        "--collection",
        default="questions",
        help="ChromaDB collection name (default: questions)",
    )
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformers model (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="batch size for database writes (default: 100)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="delete existing collection before indexing",
    )
    args = parser.parse_args()

    count = build_questions_index(
        manifest_path=args.manifest,
        database_path=args.database_dir,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
        reset=args.reset,
    )
    print(f"\nDone. {count} questions indexed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
