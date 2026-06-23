"""
core/summarizer.py — Meeting summarisation & title generation
──────────────────────────────────────────────────────────────
Uses the centralised Mistral client (core/mistral_client.py) for:
  • API key validation before the first LLM call
  • Consistent model initialisation
  • Friendly error messages on auth / network failure
"""

from __future__ import annotations

import logging

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.mistral_client import MistralAuthError, classify_api_error, get_mistral_llm

logger = logging.getLogger(__name__)

# ─── Prompts ─────────────────────────────────────────────────────────────────────

_SUMMARY_PROMPT = PromptTemplate(
    input_variables=["transcript"],
    template="""You are an expert meeting analyst. Given the transcript below, write a
concise, well-structured summary (5–8 sentences) covering:
  • Main topics discussed
  • Key conclusions reached
  • Important context or background mentioned

Transcript:
{transcript}

Summary:""",
)

_TITLE_PROMPT = PromptTemplate(
    input_variables=["transcript"],
    template="""Generate a short, descriptive title (max 10 words) for this meeting
based on the transcript below. Return only the title, no quotes, no punctuation at the end.

Transcript (first 1000 chars):
{transcript}

Title:""",
)


# ─── Helpers ─────────────────────────────────────────────────────────────────────

def _build_chain(prompt: PromptTemplate, context: str):
    """
    Build a LangChain chain (prompt | llm | parser).

    Validates the API key before constructing the model so auth errors
    surface before any HTTP traffic is attempted.

    Raises:
        MistralAuthError: Key is missing / invalid.
        RuntimeError:     Wrapped LLM error.
    """
    llm   = get_mistral_llm(context=context)
    chain = prompt | llm | StrOutputParser()
    return chain


def _invoke_safe(chain, inputs: dict, context: str) -> str:
    """
    Invoke a LangChain chain and wrap errors in friendly messages.

    Raises:
        MistralAuthError: re-raised as-is (already friendly).
        RuntimeError:     Wrapped with classify_api_error message.
    """
    try:
        return chain.invoke(inputs)
    except MistralAuthError:
        raise  # already has a good message
    except Exception as exc:
        friendly = classify_api_error(exc)
        logger.error("[%s] LLM call failed: %s", context, exc)
        raise RuntimeError(friendly) from exc


# ─── Public API ──────────────────────────────────────────────────────────────────

def summarize(transcript: str) -> str:
    """
    Generate a concise meeting summary from *transcript*.

    Args:
        transcript: Full meeting transcript text.

    Returns:
        Summary string from Mistral.

    Raises:
        MistralAuthError: API key is missing or invalid.
        RuntimeError:     LLM call failed for another reason.
    """
    if not transcript or not transcript.strip():
        logger.warning("[summarize] Empty transcript — returning placeholder")
        return "No transcript content available to summarise."

    logger.info("[summarize] Summarising transcript (%d chars)", len(transcript))

    chain = _build_chain(_SUMMARY_PROMPT, context="summarizer")
    return _invoke_safe(chain, {"transcript": transcript}, context="summarize")


def generate_title(transcript: str) -> str:
    """
    Generate a short meeting title from *transcript*.

    Args:
        transcript: Full meeting transcript text.

    Returns:
        Title string (max ~10 words).

    Raises:
        MistralAuthError: API key is missing or invalid.
        RuntimeError:     LLM call failed for another reason.
    """
    if not transcript or not transcript.strip():
        logger.warning("[generate_title] Empty transcript — returning default title")
        return "Untitled Meeting"

    # Only use the first 1000 chars for the title prompt (cheaper, faster)
    snippet = transcript[:1000]
    logger.info("[generate_title] Generating title from snippet (%d chars)", len(snippet))

    chain = _build_chain(_TITLE_PROMPT, context="generate_title")
    title = _invoke_safe(chain, {"transcript": snippet}, context="generate_title")

    return title.strip().strip('"').strip("'") or "Untitled Meeting"
