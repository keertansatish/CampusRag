"""Retrieve the most relevant indexed chunks for a user question."""

from __future__ import annotations

import argparse
from pathlib import Path


def retrieve_top_chunks(
    prompt: str,
    database_path: Path = Path("data/chroma"),
    collection_name: str = "campus_policy",
    limit: int = 3,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> list[dict[str, object]]:
    """Return the closest chunks and their citation metadata for ``prompt``."""
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if limit <= 0:
        raise ValueError("limit must be greater than zero")

    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    client = chromadb.PersistentClient(path=str(database_path))
    # The model is downloaded during indexing. Loading only from the local
    # cache keeps ordinary retrieval fast and avoids a network dependency.
    embeddings = SentenceTransformerEmbeddingFunction(model_name=embedding_model, local_files_only=True)
    collection = client.get_collection(name=collection_name, embedding_function=embeddings)
    result = collection.query(
        query_texts=[prompt],
        n_results=min(limit, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]
    return [
        {"text": document, "metadata": metadata, "distance": distance}
        for document, metadata, distance in zip(documents, metadatas, distances, strict=True)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieve the top matching chunks from the CampusRAG Chroma database.")
    parser.add_argument("prompt", help="the user's question")
    parser.add_argument("--limit", type=int, default=3, help="number of chunks to retrieve (default: 3)")
    parser.add_argument("--database-dir", type=Path, default=Path("data/chroma"), help="local Chroma database directory")
    parser.add_argument("--collection", default="campus_policy", help="Chroma collection name")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="Sentence Transformers model name")
    args = parser.parse_args()

    matches = retrieve_top_chunks(
        prompt=args.prompt,
        database_path=args.database_dir,
        collection_name=args.collection,
        limit=args.limit,
        embedding_model=args.embedding_model,
    )
    
    for rank, match in enumerate(matches, start=1):
        metadata = match["metadata"]
        print(f"#{rank} | {metadata['source_file']}, p. {metadata['page_number']} | distance: {match['distance']:.4f}")
        print(match["text"])
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
