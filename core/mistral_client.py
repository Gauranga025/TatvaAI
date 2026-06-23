"""
core/mistral_client.py — Centralised Mistral AI client factory
───────────────────────────────────────────────────────────────
Single source of truth for:
  • API key resolution  (st.secrets → .env → os.environ)
  • API key validation  (missing / empty / placeholder check)
  • ChatMistralAI model instantiation (one reusable factory)
  • Friendly error classification for Streamlit UI

WHY THIS MODULE EXISTS
──────────────────────
Before this refactor every file (summarizer.py, extractor.py,
rag_engine.py) called os.getenv("MISTRAL_API_KEY") independently.
If the key was missing the first failing call would crash with a
generic RuntimeError deep inside LangChain/MistralAI with no
actionable user message.

Now every caller does:

    from core.mistral_client import get_mistral_llm, MistralAuthError

    try:
        llm = get_mistral_llm()
    except MistralAuthError as exc:
        # show st.error / re-raise / log — caller decides
        ...
"""

from __future__ import annotations

import logging
import os
from typing import Optional
print("MISTRAL CLIENT FILE:", __file__)
logger = logging.getLogger(__name__)

# ─── Sentinel / placeholder values that should be treated as "missing" ──────────
_PLACEHOLDER_MARKERS = (
    "your_mistral_api_key_here",
    "your-key-here",
    "replace_me",
    "xxxx",
    "<your",
)

# ─── Default model ───────────────────────────────────────────────────────────────
DEFAULT_MODEL      = "mistral-small-latest"
DEFAULT_TEMPERATURE = 0.3


# ─── Custom exception ────────────────────────────────────────────────────────────

class MistralAuthError(RuntimeError):
    """Raised when the Mistral API key is missing, empty, or a placeholder."""


# ─── Key resolution ──────────────────────────────────────────────────────────────

def _resolve_api_key() -> Optional[str]:
    """
    Try to load MISTRAL_API_KEY from multiple sources in priority order:

    1. st.secrets  (Streamlit Community Cloud secrets)
    2. os.environ  (populated from .env via load_dotenv() in app.py / main.py)

    Returns the raw string (may be empty/None) so the caller can validate it.
    """
    # 1 — Streamlit secrets (only available inside a running Streamlit app)
    try:
        import streamlit as st  # noqa: PLC0415  (lazy import — not always installed)
        key = st.secrets.get("MISTRAL_API_KEY", "")
        if key:
            logger.debug("MISTRAL_API_KEY resolved from st.secrets")
            return key
    except Exception:
        # Not running inside Streamlit, or streamlit not installed → fall through
        pass

    # 2 — Environment variable (set by load_dotenv or the OS)
    key = os.getenv("MISTRAL_API_KEY", "")
    if key:
        logger.debug("MISTRAL_API_KEY resolved from os.environ")
    return key or None


def _validate_key(key: Optional[str], *, context: str = "") -> str:
    """
    Validate that *key* looks like a real API key.

    Raises MistralAuthError with a clear, actionable message on failure.
    """
    print("KEY:", repr(key))
    print("LOWER:", repr(key.strip().lower()) if key else None)
    print("LEN:", len(key.strip()) if key else 0)
    prefix = f"[{context}] " if context else ""

    if not key:
        msg = (
            f"{prefix}MISTRAL_API_KEY is missing.\n"
            "• Local dev  → add MISTRAL_API_KEY=<your-key> to your .env file\n"
            "• Streamlit Cloud → add MISTRAL_API_KEY under Settings → Secrets"
        )
        logger.error(msg)
        raise MistralAuthError(msg)

    lower = key.strip().lower()
    if any(lower.startswith(p) for p in _PLACEHOLDER_MARKERS):
        msg = (
            f"{prefix}MISTRAL_API_KEY looks like a placeholder value.\n"
            "Replace it with your real Mistral API key from "
            "https://console.mistral.ai/api-keys/"
        )
        logger.error(msg)
        raise MistralAuthError(msg)

    if len(key.strip()) < 20:
        msg = (
            f"{prefix}MISTRAL_API_KEY appears too short to be valid "
            f"(length={len(key.strip())}).\n"
            "Double-check the value in your .env / Streamlit Secrets."
        )
        logger.error(msg)
        raise MistralAuthError(msg)

    return key.strip()


# ─── Public API ──────────────────────────────────────────────────────────────────

def get_api_key(*, context: str = "") -> str:
    """
    Resolve and validate the Mistral API key.

    Args:
        context: Short label inserted into error messages (e.g. "summarizer").

    Returns:
        The validated API key string.

    Raises:
        MistralAuthError: If the key is missing, empty, or a placeholder.
    """
    raw = _resolve_api_key()

    # Diagnostic log — safe because we only log bool, never the key itself
    print(f"Mistral Key Found: {bool(raw)}")          # explicit diagnostic
    logger.info("Mistral API key present: %s", bool(raw))

    return _validate_key(raw, context=context)


def get_mistral_llm(
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    context: str = "",
):
    """
    Build and return a ChatMistralAI instance.

    Validates the API key before constructing the object so failures are
    caught early with a clear message rather than on the first .invoke() call.

    Args:
        model:       Mistral model name.
        temperature: Sampling temperature (0 = deterministic, 1 = creative).
        context:     Label for error messages (e.g. "summarizer").

    Returns:
        langchain_mistralai.ChatMistralAI

    Raises:
        MistralAuthError:  API key is missing or invalid (auth not attempted yet).
        RuntimeError:      Wrapped 401/403/network error from the first API call.
    """
    from langchain_mistralai import ChatMistralAI  # noqa: PLC0415

    api_key = get_api_key(context=context or "get_mistral_llm")

    logger.info(
        "Instantiating ChatMistralAI | model=%s | temperature=%s | context=%s",
        model, temperature, context,
    )

    return ChatMistralAI(
        model=model,
        temperature=temperature,
        mistral_api_key=api_key,
    )


def classify_api_error(exc: Exception) -> str:
    """
    Map a raw Mistral / HTTP exception to a user-friendly string.

    Used by Streamlit's _friendly_error() and any caller that wants
    a clean message without a traceback.
    """
    msg  = str(exc)
    low  = msg.lower()

    if "401" in msg or "unauthorized" in low:
        return (
            "Mistral API key is invalid or has been revoked (HTTP 401).\n"
            "Generate a new key at https://console.mistral.ai/api-keys/ "
            "and update your .env / Streamlit Secrets."
        )
    if "403" in msg or "forbidden" in low:
        return (
            "Access denied by Mistral AI (HTTP 403).\n"
            "Check that your account has access to this model."
        )
    if "429" in msg or "rate limit" in low or "too many" in low:
        return (
            "Mistral AI rate limit exceeded (HTTP 429).\n"
            "Please wait a moment and try again."
        )
    if "timeout" in low or "timed out" in low:
        return "Mistral AI request timed out. Please try again."
    if "connection" in low or "network" in low:
        return "Could not reach Mistral AI. Check your internet connection."

    # Fallback — include raw message but no traceback
    return f"Mistral API call failed: {msg}"
