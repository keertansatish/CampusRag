"""Retrieve exam questions from the 'questions' ChromaDB collection.

Supports:
  - Semantic search (vector similarity)
  - Metadata filtering (subject, marks, year, exam, has_image)
  - Repeated question detection
  - Topic frequency analysis
  - Combined semantic + filter queries
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path


def _build_where_filter(
    subject: str | None = None,
    year: str | None = None,
    exam: str | None = None,
    marks: int | None = None,
    has_image: bool | None = None,
    question_type: str | None = None,
) -> dict | None:
    """Build a ChromaDB where-filter from optional criteria."""
    conditions: list[dict] = []

    if subject:
        conditions.append({"subject": {"$eq": subject}})
    if year:
        conditions.append({"year": {"$eq": year}})
    if exam:
        conditions.append({"exam": {"$eq": exam}})
    if marks is not None and marks > 0:
        conditions.append({"marks": {"$eq": marks}})
    if has_image is not None:
        conditions.append({"has_image": {"$eq": has_image}})
    if question_type:
        conditions.append({"question_type": {"$eq": question_type}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def retrieve_questions(
    prompt: str,
    database_path: Path = Path("data/chroma"),
    collection_name: str = "questions",
    embedding_model: str = "all-MiniLM-L6-v2",
    limit: int = 10,
    subject: str | None = None,
    year: str | None = None,
    exam: str | None = None,
    marks: int | None = None,
    has_image: bool | None = None,
    question_type: str | None = None,
) -> list[dict]:
    """Retrieve the most relevant questions for a query.

    Returns a list of dicts with keys: text, metadata, distance.
    """
    if not prompt.strip():
        raise ValueError("prompt must not be empty")

    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    client = chromadb.PersistentClient(path=str(database_path))
    embeddings = SentenceTransformerEmbeddingFunction(
        model_name=embedding_model, local_files_only=True,
    )
    collection = client.get_collection(
        name=collection_name, embedding_function=embeddings,
    )

    count = collection.count()
    if count == 0:
        return []

    where_filter = _build_where_filter(
        subject=subject, year=year, exam=exam,
        marks=marks, has_image=has_image, question_type=question_type,
    )

    query_kwargs: dict = {
        "query_texts": [prompt],
        "n_results": min(limit, count),
        "include": ["documents", "metadatas", "distances"],
    }
    if where_filter:
        query_kwargs["where"] = where_filter

    try:
        result = collection.query(**query_kwargs)
    except Exception:
        # If filter produces no results, retry without filter
        if where_filter:
            del query_kwargs["where"]
            result = collection.query(**query_kwargs)
        else:
            raise

    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    return [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]


def get_all_questions(
    database_path: Path = Path("data/chroma"),
    collection_name: str = "questions",
    embedding_model: str = "all-MiniLM-L6-v2",
) -> list[dict]:
    """Retrieve ALL questions from the collection (for analysis)."""
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    client = chromadb.PersistentClient(path=str(database_path))
    embeddings = SentenceTransformerEmbeddingFunction(
        model_name=embedding_model, local_files_only=True,
    )
    collection = client.get_collection(
        name=collection_name, embedding_function=embeddings,
    )

    count = collection.count()
    if count == 0:
        return []

    result = collection.get(
        include=["documents", "metadatas"],
        limit=count,
    )

    return [
        {"text": doc, "metadata": meta}
        for doc, meta in zip(result["documents"], result["metadatas"])
    ]


def find_repeated_questions(
    database_path: Path = Path("data/chroma"),
    collection_name: str = "questions",
    embedding_model: str = "all-MiniLM-L6-v2",
    similarity_threshold: float = 0.75,
) -> list[dict]:
    """Find questions that appear to be repeated across different papers.

    Uses text similarity (SequenceMatcher) to group similar questions.
    Returns groups of similar questions.
    """
    all_q = get_all_questions(database_path, collection_name, embedding_model)
    if not all_q:
        return []

    # Compare each pair
    groups: list[dict] = []
    used: set[int] = set()

    for i, q1 in enumerate(all_q):
        if i in used:
            continue

        text1 = q1["metadata"].get("question_text", "")
        if not text1 or len(text1) < 20:
            continue

        group = {"representative": text1, "questions": [q1]}

        for j, q2 in enumerate(all_q):
            if j <= i or j in used:
                continue

            text2 = q2["metadata"].get("question_text", "")
            if not text2 or len(text2) < 20:
                continue

            # Quick length check before expensive comparison
            if abs(len(text1) - len(text2)) > max(len(text1), len(text2)) * 0.5:
                continue

            ratio = SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
            if ratio >= similarity_threshold:
                group["questions"].append(q2)
                used.add(j)

        if len(group["questions"]) > 1:
            group["count"] = len(group["questions"])
            group["sources"] = [
                f"{q['metadata'].get('source_file', '?')} (p.{q['metadata'].get('page_number', '?')})"
                for q in group["questions"]
            ]
            groups.append(group)
            used.add(i)

    groups.sort(key=lambda g: g["count"], reverse=True)
    return groups


def analyze_topic_frequency(
    database_path: Path = Path("data/chroma"),
    collection_name: str = "questions",
    embedding_model: str = "all-MiniLM-L6-v2",
    subject: str | None = None,
) -> list[tuple[str, int]]:
    """Analyze which topics/keywords appear most frequently in questions.

    Returns a list of (keyword, count) tuples sorted by frequency.
    """
    all_q = get_all_questions(database_path, collection_name, embedding_model)
    if not all_q:
        return []

    # Filter by subject if specified
    if subject:
        all_q = [
            q for q in all_q
            if subject.lower() in (q["metadata"].get("subject", "") or "").lower()
        ]

    # Extract meaningful keywords (not stopwords)
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "under", "again",
        "further", "then", "once", "here", "there", "when", "where", "why",
        "how", "all", "both", "each", "few", "more", "most", "other", "some",
        "such", "no", "nor", "not", "only", "own", "same", "so", "than",
        "too", "very", "just", "because", "but", "and", "or", "if", "that",
        "this", "it", "its", "what", "which", "who", "whom", "these",
        "those", "i", "we", "you", "they", "he", "she", "me", "him", "her",
        "us", "them", "my", "your", "his", "our", "their", "following",
        "given", "find", "determine", "calculate", "show", "prove", "state",
        "define", "explain", "discuss", "describe", "write", "obtain",
        "derive", "using", "also", "about", "any", "answer", "question",
        "marks", "mark",
    }

    word_counter: Counter[str] = Counter()
    # Also count multi-word phrases
    phrase_counter: Counter[str] = Counter()

    for q in all_q:
        text = q["metadata"].get("question_text", "")
        if not text:
            continue

        # Single words
        words = re.findall(r"[a-zA-Z]{3,}", text.lower())
        meaningful = [w for w in words if w not in stopwords]
        word_counter.update(meaningful)

        # Bigrams
        for j in range(len(meaningful) - 1):
            phrase = f"{meaningful[j]} {meaningful[j+1]}"
            phrase_counter.update([phrase])

    # Combine: prefer phrases that appear frequently
    combined: Counter[str] = Counter()
    for phrase, count in phrase_counter.most_common(100):
        if count >= 2:
            combined[phrase] = count

    for word, count in word_counter.most_common(200):
        if count >= 3 and word not in combined:
            combined[word] = count

    return combined.most_common(50)


def format_question_result(match: dict, rank: int) -> str:
    """Format a single question result for display."""
    meta = match["metadata"]
    lines = [
        f"#{rank} | {meta.get('source_file', '?')}, p.{meta.get('page_number', '?')}",
    ]
    if meta.get("subject"):
        lines[0] += f" | {meta['subject']}"
    if meta.get("year"):
        lines[0] += f" ({meta['year']})"
    if meta.get("marks") and meta["marks"] > 0:
        lines[0] += f" | {meta['marks']} marks"
    if "distance" in match:
        lines[0] += f" | similarity: {1 - match['distance']:.3f}"

    lines.append(f"   Q{meta.get('question_number', '?')}: {meta.get('question_text', '')[:200]}")

    if meta.get("has_image"):
        image_paths = meta.get("image_paths", "[]")
        if isinstance(image_paths, str):
            try:
                image_paths = json.loads(image_paths)
            except json.JSONDecodeError:
                image_paths = []
        lines.append(f"   📷 Has image: {', '.join(image_paths) if image_paths else 'yes'}")

    if meta.get("image_description"):
        lines.append(f"   📝 Diagram: {meta['image_description'][:150]}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Retrieve exam questions from the 'questions' vector database."
    )
    parser.add_argument("prompt", help="search query")
    parser.add_argument("--limit", type=int, default=10, help="max results (default: 10)")
    parser.add_argument("--database-dir", type=Path, default=Path("data/chroma"))
    parser.add_argument("--collection", default="questions")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--subject", help="filter by subject")
    parser.add_argument("--year", help="filter by year")
    parser.add_argument("--exam", help="filter by exam type (e.g. 'CIA 2', 'End Semester')")
    parser.add_argument("--marks", type=int, help="filter by marks")
    parser.add_argument("--has-image", action="store_true", default=None, help="only questions with images")
    parser.add_argument("--repeated", action="store_true", help="find repeated questions instead of search")
    parser.add_argument("--topics", action="store_true", help="analyze topic frequency instead of search")
    args = parser.parse_args()

    if args.repeated:
        print("Analyzing repeated questions...")
        groups = find_repeated_questions(
            database_path=args.database_dir,
            collection_name=args.collection,
            embedding_model=args.embedding_model,
        )
        if not groups:
            print("No repeated questions found.")
            return 0
        print(f"\nFound {len(groups)} groups of repeated questions:\n")
        for i, group in enumerate(groups, start=1):
            print(f"Group {i} (appeared {group['count']} times):")
            print(f"  Question: {group['representative'][:200]}")
            print(f"  Sources: {', '.join(group['sources'])}")
            print()
        return 0

    if args.topics:
        print("Analyzing topic frequency...")
        topics = analyze_topic_frequency(
            database_path=args.database_dir,
            collection_name=args.collection,
            embedding_model=args.embedding_model,
            subject=args.subject,
        )
        if not topics:
            print("No topics found.")
            return 0
        print(f"\nTop topics/keywords:\n")
        for topic, count in topics:
            print(f"  {count:3d}x  {topic}")
        return 0

    matches = retrieve_questions(
        prompt=args.prompt,
        database_path=args.database_dir,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        limit=args.limit,
        subject=args.subject,
        year=args.year,
        exam=args.exam,
        marks=args.marks,
        has_image=args.has_image if args.has_image else None,
        question_type=None,
    )

    if not matches:
        print("No matching questions found.")
        return 0

    print(f"\nFound {len(matches)} matching questions:\n")
    for rank, match in enumerate(matches, start=1):
        print(format_question_result(match, rank))
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
