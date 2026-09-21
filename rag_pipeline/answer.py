"""Generate a grounded RAG answer from retrieved CampusRAG chunks."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from question_retrieve import retrieve_questions
from retrieve import retrieve_top_chunks
from router import ROUTE_EXAM, route_query

DEFAULT_MODEL = "openai/gpt-oss-20b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Read credentials/configuration from the project file, not terminal arguments.
load_dotenv(Path(__file__).parent.parent / ".env")


def format_context(matches: list[dict[str, object]]) -> str:
    """Label retrieved chunks so the model can cite them in its answer."""
    sections: list[str] = []
    for number, match in enumerate(matches, start=1):
        metadata = match["metadata"]
        sections.append(
            f"[Source {number}: {metadata['source_file']}, page {metadata['page_number']}]\n"
            f"{match['text']}"
        )
    return "\n\n".join(sections)


def answer_question(prompt: str, model: str = DEFAULT_MODEL, limit: int = 3) -> tuple[str, list[dict[str, object]]]:
    """Retrieve relevant chunks, then ask an LLM to answer using only them."""
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("Set the GROQ_API_KEY environment variable before generating an answer.")

    route = route_query(prompt, model=model)
    if route == ROUTE_EXAM:
        matches = retrieve_questions(prompt, limit=limit)
    else:
        matches = retrieve_top_chunks(prompt, limit=limit)
    if not matches:
        return "I could not find relevant information in the indexed documents.", []

    from openai import OpenAI

    # Initialize the client with the proper Groq API endpoint
    client = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url=GROQ_BASE_URL)
    
    # FIX: Corrected endpoint mapping and structured the chat messages array properly
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a CampusRAG assistant. Answer the user's question using only the supplied "
                    "retrieved context. If the context does not support an answer, say so plainly. "
                    "Do not invent regulations or facts. Cite each factual claim with the applicable "
                    "source label, for example [Source 1]. Keep the answer concise."
                )
            },
            {
                "role": "user",
                "content": f"User question:\n{prompt}\n\nRetrieved context:\n{format_context(matches)}"
            }

        ],
        temperature=0.2,
    )
    
    # FIX: Extract response text via the correct object mapping path (.choices[0].message.content)
    return response.choices[0].message.content.strip(), matches


def main() -> int:
    parser = argparse.ArgumentParser(description="Answer a user question with retrieved CampusRAG context.")
    parser.add_argument("prompt", help="the user's question")
    parser.add_argument(
        "--model",
        default=os.getenv("LLM_MODEL", DEFAULT_MODEL),
        help=f"Groq model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--limit", type=int, default=3, help="number of retrieved chunks to provide (default: 3)")
    args = parser.parse_args()

    answer, matches = answer_question(args.prompt, model=args.model, limit=args.limit)
    print(answer)
    if matches:
        print("\nRetrieved sources:")
        for number, match in enumerate(matches, start=1):
            metadata = match["metadata"]
            print(f"[Source {number}] {metadata['source_file']}, p. {metadata['page_number']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
