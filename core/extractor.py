"""
core/extractor.py — Action items, key decisions, open questions
───────────────────────────────────────────────────────────────
Uses the centralised Mistral client (core/mistral_client.py) for:
  • API key validation before the first LLM call
  • Consistent model initialisation across all three extractors
  • Friendly error messages on auth / network failure
"""

from __future__ import annotations

import logging

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.mistral_client import MistralAuthError, classify_api_error, get_mistral_llm

logger = logging.getLogger(__name__)

# ─── Prompts ─────────────────────────────────────────────────────────────────────

_ACTION_ITEMS_PROMPT = PromptTemplate(
    input_variables=["transcript"],
    template="""You are an expert at extracting action items from meeting transcripts.
Analyse the transcript below and list ALL action items.

Format each item as:
• [Owner if mentioned] — Action description — [Deadline if mentioned]

If no action items are mentioned, write: "No action items identified."

Transcript:
{transcript}

Action Items:""",
)

_KEY_DECISIONS_PROMPT = PromptTemplate(
    input_variables=["transcript"],
    template="""You are an expert at identifying key decisions from meeting transcripts.
Analyse the transcript below and list ALL decisions that were made.

Format each decision as:
• Decision: [what was decided]
  Rationale: [why, if mentioned]

If no decisions were made, write: "No key decisions identified."

Transcript:
{transcript}

Key Decisions:""",
)

_OPEN_QUESTIONS_PROMPT = PromptTemplate(
    input_variables=["transcript"],
    template="""You are an expert at identifying unresolved questions from meeting transcripts.
Analyse the transcript below and list ALL questions that were raised but NOT answered.

Format each item as:
• [Question] — raised by [person if mentioned]

If all questions were resolved, write: "No open questions identified."

Transcript:
{transcript}

Open Questions:""",
)


# ─── Internal helpers ────────────────────────────────────────────────────────────

def _build_chain(prompt: PromptTemplate, context: str):
    """
    Build a LangChain chain (prompt | llm | parser).

    Validates the API key *before* constructing the model object so
    MistralAuthError fires immediately — not on the first HTTP call.
    """
    llm   = get_mistral_llm(context=context)
    chain = prompt | llm | StrOutputParser()
    return chain


def _invoke_safe(chain, inputs: dict, context: str) -> str:
    """
    Invoke *chain* and translate raw exceptions into friendly messages.

    Raises:
        MistralAuthError: re-raised unchanged (already has a good message).
        RuntimeError:     Wrapped with classify_api_error().
    """
    try:
        return chain.invoke(inputs)
    except MistralAuthError:
        raise
    except Exception as exc:
        friendly = classify_api_error(exc)
        logger.error("[%s] LLM call failed: %s", context, exc)
        raise RuntimeError(friendly) from exc


def _extract(prompt: PromptTemplate, transcript: str, context: str, empty_msg: str) -> str:
    """
    Generic extraction helper used by the three public functions below.

    Args:
        prompt:     The PromptTemplate to use.
        transcript: Full transcript text.
        context:    Label for logging / error messages.
        empty_msg:  Return value when transcript is empty.

    Returns:
        Extracted text string.
    """
    if not transcript or not transcript.strip():
        logger.warning("[%s] Empty transcript — returning placeholder", context)
        return empty_msg

    logger.info("[%s] Running extraction on %d-char transcript", context, len(transcript))
    chain = _build_chain(prompt, context=context)
    return _invoke_safe(chain, {"transcript": transcript}, context=context)


# ─── Public API ──────────────────────────────────────────────────────────────────

def extract_action_items(transcript: str) -> str:
    """
    Extract action items from *transcript*.

    Returns:
        Bullet-list string of action items, or a "none identified" message.

    Raises:
        MistralAuthError: API key is missing or invalid.
        RuntimeError:     LLM call failed.
    """
    return _extract(
        _ACTION_ITEMS_PROMPT,
        transcript,
        context="extract_action_items",
        empty_msg="No transcript content available to extract action items from.",
    )


def extract_key_decisions(transcript: str) -> str:
    """
    Extract key decisions from *transcript*.

    Returns:
        Bullet-list string of decisions, or a "none identified" message.

    Raises:
        MistralAuthError: API key is missing or invalid.
        RuntimeError:     LLM call failed.
    """
    return _extract(
        _KEY_DECISIONS_PROMPT,
        transcript,
        context="extract_key_decisions",
        empty_msg="No transcript content available to extract decisions from.",
    )


def extract_questions(transcript: str) -> str:
    """
    Extract unresolved open questions from *transcript*.

    Returns:
        Bullet-list string of open questions, or a "none identified" message.

    Raises:
        MistralAuthError: API key is missing or invalid.
        RuntimeError:     LLM call failed.
    """
    return _extract(
        _OPEN_QUESTIONS_PROMPT,
        transcript,
        context="extract_questions",
        empty_msg="No transcript content available to extract questions from.",
    )
