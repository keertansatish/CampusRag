"""Route user queries to either the General College RAG or Exam Prep RAG.

Uses a fast keyword check first, with an LLM fallback for ambiguous cases.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# LLM config (reuses existing project settings)
DEFAULT_MODEL = "openai/gpt-oss-20b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Route types
ROUTE_GENERAL = "general"
ROUTE_EXAM = "exam_prep"

# ---------------------------------------------------------------------------
# Keyword-based fast routing
# ---------------------------------------------------------------------------

# _EXAM_KEYWORDS = [
#     # Direct exam references
#     r"\bcia\b", r"\bcia[\s\-]*[123]\b", r"\bend\s*sem(?:ester)?\b",
#     r"\bmid[\s\-]*sem(?:ester)?\b", r"\bexam\b", r"\bexamination\b",
#     r"\bquestion\s*paper\b", r"\bprevious\s*(?:year|paper|question)\b",
#     r"\bpast\s*(?:year|paper|question)\b",
#     # Study/prep related
#     r"\bstudy\b", r"\bprepare\b", r"\bpreparation\b", r"\brevise\b",
#     r"\brevision\b", r"\bimportant\s*(?:topic|question)\b",
#     r"\bfrequent(?:ly)?\b", r"\brepeated\b",
#     # Question-specific
#     r"\bprevious\s*question\b", r"\bquestion\s*(?:on|about|related)\b",
#     r"\b\d+[\s\-]*mark\b", r"\bmark\s*question\b",
#     r"\bunit[\s\-]*\d\b",
#     # Pattern/analysis
#     r"\btopic\s*(?:frequency|analysis|pattern|wise)\b",
#     r"\bmost\s*(?:asked|important|common|frequent)\b",
#     r"\bwhich\s*(?:topic|question|chapter)\b.*\b(?:come|appear|ask|repeat)\b",
#     # Direct question retrieval
#     r"\bshow\s*me\s*(?:previous|past|old)\b",
#     r"\bgive\s*me\s*(?:previous|past|old|important)\b",
#     r"\blist\s*(?:previous|past|old|important)\b",
#     r"\bfind\s*(?:question|similar)\b",
#     r"\bsimilar\s*question\b",
#     # Paper analysis
#     r"\banalyze?\s*(?:paper|question|cia)\b",
#     r"\bpaper\s*analysis\b",
#     r"\bquestion\s*bank\b",
#     r"\bmodel\s*(?:paper|question)\b",
#     r"\bsample\s*(?:paper|question)\b",
# ]

# _EXAM_COMPILED = [re.compile(p, re.IGNORECASE) for p in _EXAM_KEYWORDS]

# # General college keywords (strong signals for general RAG)
# _GENERAL_KEYWORDS = [
#     r"\bhostel\b", r"\bfee\b", r"\badmission\b", r"\battendance\b",
#     r"\bscholarship\b", r"\bplacement\b", r"\blibrary\b", r"\bcanteen\b",
#     r"\btransport\b", r"\bbus\b", r"\bragging\b", r"\banti[\s\-]*ragging\b",
#     r"\bregulation\b", r"\brule\b", r"\bdocument\b", r"\bsubmit\b",
#     r"\bregistration\b", r"\benrolment\b", r"\benrollment\b",
#     r"\bcertificate\b", r"\btranscript\b", r"\bgrade\b.*\bcard\b",
#     r"\bsyllabus\b", r"\bcurriculum\b", r"\bcalendar\b",
#     r"\bpenalty\b", r"\bdisciplin\b",
#     r"\bexam.*\breg(?:ulation|istry)\b",
# ]

# _GENERAL_COMPILED = [re.compile(p, re.IGNORECASE) for p in _GENERAL_KEYWORDS]


# def keyword_route(query: str) -> str | None:
#     """Attempt to route using keyword matching. Returns None if ambiguous."""
#     exam_score = sum(1 for p in _EXAM_COMPILED if p.search(query))
#     general_score = sum(1 for p in _GENERAL_COMPILED if p.search(query))

#     # Clear winner
#     if exam_score > 0 and general_score == 0:
#         return ROUTE_EXAM
#     if general_score > 0 and exam_score == 0:
#         return ROUTE_GENERAL
#     if exam_score > general_score + 1:
#         return ROUTE_EXAM
#     if general_score > exam_score + 1:
#         return ROUTE_GENERAL

#     # Ambiguous
#     return None


def llm_route(query: str, model: str = DEFAULT_MODEL) -> str:
    """Use the LLM to classify the query when keywords are ambiguous."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        # Fallback: default to general
        print("No GROQ API key found. Using default general route.")
        return ROUTE_GENERAL

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a query classifier for a college RAG system with two pipelines:\n"
                    "1. 'general' — answers about college policies, regulations, hostel, fees, "
                    "attendance, anti-ragging, admissions, library, transport, etc.\n"
                    "2. 'exam_prep' — answers about CIA exams, previous question papers, "
                    "important topics, study preparation, question analysis, repeated questions, "
                    "marks-based retrieval, subject-wise questions, etc.\n\n"
                    "your job is to classify whether the given prompt is 'general' or 'exam_prep'. "
                    "Respond with ONLY the word 'general' or 'exam_prep'. Nothing else."
                ),
            },
            {
                "role": "user",
                "content": query,
            },
        ],
        max_tokens=200,
        temperature=0,
        reasoning_effort="low"
    )

    answer = response.choices[0].message.content.strip().lower()
    print(f"LLM route classification: {answer}")
    if "exam" in answer:
        return ROUTE_EXAM
    return ROUTE_GENERAL


def route_query(query: str, model: str = DEFAULT_MODEL) -> str:
    """Determine which RAG pipeline should handle the query.

    Returns 'general' or 'exam_prep'.
    """
    # Fast keyword check first
    # result = keyword_route(query)
    # if result is not None:
    #     return result

    # LLM fallback for ambiguous queries
    return llm_route(query, model)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Route a query to the correct RAG pipeline.")
    parser.add_argument("query", help="the user's question")
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", DEFAULT_MODEL))
    args = parser.parse_args()

    route = route_query(args.query, model=args.model)
    print(f"Query: {args.query}")
    print(f"Route: {route}")

    # keyword_result = keyword_route(args.query)

    # print(f"Keyword match: {keyword_result or 'ambiguous (used LLM)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
