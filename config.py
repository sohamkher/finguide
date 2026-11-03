"""Central settings for FinGuide.

Secrets (the Groq API key) are read from an environment variable / .env file first and
from Streamlit secrets second, so the same code works locally and on Streamlit Community Cloud.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def get_secret(name: str, default: str | None = None) -> str | None:
    """Environment variable first, then Streamlit secrets (.streamlit/secrets.toml or Cloud)."""
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st

        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:  # no secrets file, or not running inside Streamlit
        pass
    return default


ROOT = Path(__file__).parent
KNOWLEDGE_BASE_DIR = ROOT / "knowledge_base"
CACHE_DIR = ROOT / ".cache"           # saved FAISS index, so restarts are fast

# Embeddings run locally (no API needed). "fastembed" is light (ONNX, no PyTorch) and suits
# Streamlit Cloud. "huggingface" uses sentence-transformers and needs: pip install langchain-huggingface sentence-transformers
EMBEDDING_BACKEND = (get_secret("EMBEDDING_BACKEND", "fastembed") or "fastembed").lower()
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Groq retired llama-3.3-70b-versatile and llama-3.1-8b-instant on 16 Aug 2026 (free/developer tiers).
# The sidebar loads the live model list from Groq; these are the fallbacks when it cannot.
DEFAULT_MODEL = get_secret("GROQ_MODEL", "openai/gpt-oss-120b")
FALLBACK_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]

PARAMS: dict = {
    "chunk_size": 700,        # characters per chunk
    "chunk_overlap": 120,     # characters shared between neighbouring chunks
    "top_k": 4,               # passages retrieved per question
    "min_relevance": float(get_secret("MIN_RELEVANCE_SCORE", "0.25")),  # cosine similarity gate
    "temperature": 0.3,
    "top_p": 0.9,
    "max_tokens": 1500,       # includes the model's hidden reasoning for gpt-oss models
    "reasoning_effort": "low",  # gpt-oss models only: low / medium / high
}

# Rewrite follow-up questions ("explain that more simply") into standalone questions
# using the chat history. Costs one short extra LLM call, and only when needed.
CONDENSE_FOLLOWUPS = True

# Protection for a public app that runs on YOUR Groq key. Visitors who paste their own key are not limited.
MAX_QUESTION_CHARS = int(get_secret("MAX_QUESTION_CHARS", "600"))
MAX_QUESTIONS_PER_SESSION = int(get_secret("MAX_QUESTIONS_PER_SESSION", "25"))

# PDF pages (1-based, as in the PDF viewer) that are NOT indexed, per knowledge-base file.
# They are front matter, contact lists, blank worksheets and personalised "Dear X" advice
# letters, which would add noise or conflict with FinGuide's "no personal advice" rule.
SKIP_PAGES: dict[str, str] = {
    "Financial_Education_Booklet_-_English.pdf": "1-3,67-73",
    "I_Can_Do_RBI.pdf": "1-7,31,39-45,56-61,82-83,88-95,108-111,128-129,143",
}
