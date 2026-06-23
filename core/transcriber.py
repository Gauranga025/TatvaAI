"""
core/transcriber.py
────────────────────
Transcribes a list of WAV chunk file paths into a single text transcript.

Two backends are supported:
  • "english"  — OpenAI Whisper (local, CPU-friendly small model)
  • "hinglish"  — Sarvam AI transcription API (remote)

Production risks addressed
──────────────────────────
• Whisper model is loaded once per process via @st.cache_resource (in app.py)
  or lazily here with a module-level singleton — avoids reloading ~150 MB
  weights on every page interaction.
• Each chunk is transcribed independently so a single corrupt chunk does not
  abort the entire job; its slot is filled with a warning string instead.
• Sarvam API failures are caught with full error messages.
• No temp files are created here; all paths come from the caller.
• Memory: Whisper fp32 "small" peak ~1 GB RAM; acceptable on Streamlit Cloud
  (2 GB limit). Model is never loaded for the Sarvam path.
"""

import logging
import os
from typing import Literal

logger = logging.getLogger(__name__)

# ── Singleton Whisper model ──────────────────────────────────────────────────
_whisper_model = None
_whisper_model_name = "small"   # small: fast + accurate enough for meetings


def _get_whisper_model():
    """Load (and cache) the Whisper model exactly once per process."""
    global _whisper_model
    if _whisper_model is None:
        try:
            import whisper
        except ImportError as exc:
            raise RuntimeError(
                "openai-whisper is not installed. Add `openai-whisper` to requirements.txt."
            ) from exc
        logger.info("Loading Whisper model '%s' …", _whisper_model_name)
        _whisper_model = whisper.load_model(_whisper_model_name)
        logger.info("Whisper model loaded.")
    return _whisper_model


# ── Public API ───────────────────────────────────────────────────────────────

def transcribe_all(
    chunk_paths: list[str],
    language: Literal["english", "hinglish"] = "english",
) -> str:
    """
    Transcribe every chunk in *chunk_paths* and concatenate the results.

    Args:
        chunk_paths: Ordered list of WAV file paths (from audio_processor).
        language:    "english" uses Whisper; "hinglish" uses Sarvam AI.

    Returns:
        Full transcript as a single string.

    Raises:
        RuntimeError: If transcription fails completely (all chunks failed).
    """
    if not chunk_paths:
        raise ValueError("chunk_paths is empty — nothing to transcribe.")

    if language == "hinglish":
        return _transcribe_sarvam(chunk_paths)
    else:
        return _transcribe_whisper(chunk_paths)


# ── Whisper backend ──────────────────────────────────────────────────────────

def _transcribe_whisper(chunk_paths: list[str]) -> str:
    model = _get_whisper_model()
    parts: list[str] = []
    errors: list[str] = []

    for i, path in enumerate(chunk_paths, start=1):
        if not os.path.exists(path):
            logger.warning("Chunk %d not found: %s", i, path)
            errors.append(f"[Chunk {i}: file not found]")
            continue
        try:
            result = model.transcribe(
                path,
                language="en",
                fp16=False,          # CPU-safe; GPU users get fp32 still
                verbose=False,
            )
            text = result.get("text", "").strip()
            if text:
                parts.append(text)
            else:
                logger.warning("Chunk %d produced empty transcript.", i)
        except Exception as exc:
            logger.error("Whisper failed on chunk %d: %s", i, exc)
            errors.append(f"[Chunk {i}: transcription error — {exc}]")

    if not parts and errors:
        raise RuntimeError(
            "Transcription failed for all audio chunks.\n\n"
            + "\n".join(errors)
        )

    transcript = " ".join(parts)
    if errors:
        transcript += "\n\n" + "\n".join(errors)
    return transcript


# ── Sarvam AI backend ────────────────────────────────────────────────────────

def _transcribe_sarvam(chunk_paths: list[str]) -> str:
    """
    Transcribe using the Sarvam AI speech-to-text API.

    Requires SARVAM_API_KEY in the environment.
    API docs: https://docs.sarvam.ai/api-reference-docs/endpoints/speech-to-text
    """
    import requests  # already in requirements

    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise RuntimeError(
            "SARVAM_API_KEY is not set. "
            "Add it to your .env file (local) or Streamlit Cloud secrets."
        )

    url = "https://api.sarvam.ai/speech-to-text"
    headers = {"api-subscription-key": api_key}

    parts: list[str] = []
    errors: list[str] = []

    for i, path in enumerate(chunk_paths, start=1):
        if not os.path.exists(path):
            errors.append(f"[Chunk {i}: file not found]")
            continue
        try:
            with open(path, "rb") as f:
                response = requests.post(
                    url,
                    headers=headers,
                    files={"file": (os.path.basename(path), f, "audio/wav")},
                    data={
                        "language_code": "hi-IN",
                        "model": "saarika:v2",
                        "with_timestamps": "false",
                    },
                    timeout=120,
                )
            if response.status_code == 200:
                data = response.json()
                text = data.get("transcript", "").strip()
                if text:
                    parts.append(text)
            elif response.status_code == 401:
                raise RuntimeError(
                    "Sarvam API authentication failed. Check your SARVAM_API_KEY."
                )
            elif response.status_code == 429:
                raise RuntimeError(
                    "Sarvam API rate limit exceeded. Please wait and try again."
                )
            else:
                errors.append(
                    f"[Chunk {i}: Sarvam API error {response.status_code} — {response.text[:200]}]"
                )
        except RuntimeError:
            raise
        except Exception as exc:
            logger.error("Sarvam transcription failed on chunk %d: %s", i, exc)
            errors.append(f"[Chunk {i}: error — {exc}]")

    if not parts and errors:
        raise RuntimeError(
            "Sarvam transcription failed for all chunks.\n\n" + "\n".join(errors)
        )

    transcript = " ".join(parts)
    if errors:
        transcript += "\n\n" + "\n".join(errors)
    return transcript
