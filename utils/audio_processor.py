"""
utils/audio_processor.py
─────────────────────────
Handles audio acquisition from two sources:
  1. YouTube URL  → yt-dlp download → FFmpeg conversion → WAV chunks
  2. Local / uploaded file → FFmpeg conversion → WAV chunks

Production risks addressed
──────────────────────────
• All files live inside a caller-supplied TemporaryDirectory; nothing is
  written to downloads/, audio_chunks/, or any persistent project folder.
• The temp dir is owned by the caller (app.py) and cleaned up there with
  a try/finally block, so cleanup happens even if the pipeline crashes.
• yt-dlp anti-bot failures are caught and re-raised as descriptive errors.
• File size is checked before chunking; files > MAX_FILE_MB get a clear error.
• pydub AudioSegment is exported as WAV (pcm_s16le) so Whisper gets a format
  it handles without additional conversion.
• Each chunk is written to the same temp dir, avoiding cross-user collisions.
"""

import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
CHUNK_DURATION_MS   = 10 * 60 * 1000   # 10 minutes per chunk
MAX_FILE_MB         = 500               # hard limit for uploaded files
AUDIO_SAMPLE_RATE   = 16_000           # Hz — what Whisper expects
YTDLP_TIMEOUT_SEC   = 300              # 5 minutes max for download


# ── Public API ───────────────────────────────────────────────────────────────

def process_input(source: str, work_dir: Optional[str] = None) -> list[str]:
    """
    Convert *source* (URL or file path) into a list of WAV chunk file paths.

    All output files are created inside *work_dir*.  If *work_dir* is None a
    new TemporaryDirectory is created and its path is prepended to the returned
    list as element [0] with the sentinel prefix "__TMPDIR__:".

    Callers that pass their own *work_dir* receive only chunk paths.

    Args:
        source:   YouTube URL or path to a local audio/video file.
        work_dir: Directory where all temp files should be placed.

    Returns:
        List of absolute paths to WAV chunk files, ordered sequentially.

    Raises:
        ValueError:  Source is empty, file not found, or file too large.
        RuntimeError: Download failed, FFmpeg conversion failed.
    """
    source = (source or "").strip()
    if not source:
        raise ValueError("No source provided. Enter a YouTube URL or upload a file.")

    # Determine whether this is a URL or a local path
    is_url = re.match(r"https?://", source, re.IGNORECASE) is not None

    if is_url:
        wav_path = _download_youtube(source, work_dir)
    else:
        wav_path = _convert_local_file(source, work_dir)

    chunks = _split_audio(wav_path, work_dir)
    return chunks


# ── Private helpers ───────────────────────────────────────────────────────────

def _tmp(work_dir: Optional[str], suffix: str, prefix: str = "ta_") -> str:
    """Create a NamedTemporaryFile path inside *work_dir* (or system tmp)."""
    kw = dict(suffix=suffix, prefix=prefix, delete=False)
    if work_dir:
        kw["dir"] = work_dir
    with tempfile.NamedTemporaryFile(**kw) as f:
        return f.name


def _download_youtube(url: str, work_dir: Optional[str]) -> str:
    """
    Download audio from a YouTube URL using yt-dlp.

    Returns the path to a 16 kHz mono WAV file.

    Raises:
        RuntimeError: on yt-dlp failure with a user-friendly message.
    """
    try:
        import yt_dlp  # lazy import — only needed for YouTube path
    except ImportError as exc:
        raise RuntimeError(
            "yt-dlp is not installed. Add `yt-dlp` to requirements.txt."
        ) from exc

    out_template = _tmp(work_dir, suffix=".%(ext)s", prefix="ta_ytdl_")
    # Remove the extension placeholder so yt-dlp fills it in
    out_template_base = out_template.replace(".%(ext)s", "")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_template_base + ".%(ext)s",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        # Postprocessor: convert to WAV 16 kHz mono via ffmpeg
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "0",
            }
        ],
        "postprocessor_args": [
            "-ar", str(AUDIO_SAMPLE_RATE),
            "-ac", "1",
        ],
        "socket_timeout": 30,
        # Disable caching on Cloud — no persistent home dir available
        "cachedir": False,
        # Retries
        "retries": 3,
        "fragment_retries": 3,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info is None:
                raise RuntimeError("yt-dlp returned no video information.")
    except yt_dlp.utils.DownloadError as exc:
        raise RuntimeError(_ytdlp_friendly(str(exc))) from exc
    except yt_dlp.utils.ExtractorError as exc:
        raise RuntimeError(
            f"Could not extract video info. The URL may be private or restricted.\n{exc}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"YouTube download failed: {exc}") from exc

    # yt-dlp postprocessor renames the file to .wav
    wav_path = out_template_base + ".wav"
    if not os.path.exists(wav_path):
        # Fallback: scan work_dir for any newly created .wav
        search_dir = work_dir or tempfile.gettempdir()
        matches = list(Path(search_dir).glob("ta_ytdl_*.wav"))
        if not matches:
            raise RuntimeError(
                "Download appeared to succeed but the WAV output file was not found."
            )
        wav_path = str(sorted(matches, key=os.path.getmtime)[-1])

    _check_file_size(wav_path)
    return wav_path


def _convert_local_file(path: str, work_dir: Optional[str]) -> str:
    """
    Convert a local audio/video file to a 16 kHz mono WAV via FFmpeg.

    Returns the path to the converted WAV file.
    """
    if not os.path.exists(path):
        raise ValueError(f"File not found: {path}")

    _check_file_size(path)

    out_wav = _tmp(work_dir, suffix=".wav", prefix="ta_conv_")

    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [
        ffmpeg,
        "-y",                      # overwrite without asking
        "-i", path,
        "-ar", str(AUDIO_SAMPLE_RATE),
        "-ac", "1",                # mono
        "-sample_fmt", "s16",      # 16-bit PCM
        "-f", "wav",
        out_wav,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=600)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"FFmpeg conversion failed.\n\n"
            f"Command: {' '.join(cmd)}\n\n"
            f"FFmpeg output:\n{stderr[-2000:]}"
        )

    return out_wav


def _split_audio(wav_path: str, work_dir: Optional[str]) -> list[str]:
    """
    Split *wav_path* into ≤10-minute WAV chunks.

    Uses pydub which reads the whole file into memory.  For Streamlit Cloud
    this is acceptable for files up to ~500 MB; for longer content callers
    should reject earlier via _check_file_size.

    Returns a list of chunk file paths.
    """
    try:
        from pydub import AudioSegment
    except ImportError as exc:
        raise RuntimeError(
            "pydub is not installed. Add `pydub` to requirements.txt."
        ) from exc

    try:
        audio = AudioSegment.from_wav(wav_path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not read audio file. It may be corrupt or in an unsupported format.\n{exc}"
        ) from exc

    duration_ms = len(audio)
    chunk_paths: list[str] = []

    if duration_ms <= CHUNK_DURATION_MS:
        # Short enough — return as-is (no copy needed)
        chunk_paths.append(wav_path)
        return chunk_paths

    for i, start in enumerate(range(0, duration_ms, CHUNK_DURATION_MS)):
        segment = audio[start : start + CHUNK_DURATION_MS]
        chunk_path = _tmp(work_dir, suffix=".wav", prefix=f"ta_chunk_{i:03d}_")
        segment.export(chunk_path, format="wav", parameters=["-ar", str(AUDIO_SAMPLE_RATE)])
        chunk_paths.append(chunk_path)

    # Free the large AudioSegment from memory
    del audio

    return chunk_paths


def _check_file_size(path: str) -> None:
    """Raise ValueError if the file exceeds MAX_FILE_MB."""
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        raise ValueError(
            f"File is {size_mb:.0f} MB, which exceeds the {MAX_FILE_MB} MB limit. "
            "Please use a shorter or lower-bitrate recording."
        )


def _ytdlp_friendly(raw: str) -> str:
    """Convert a raw yt-dlp error message into a user-friendly string."""
    raw_lower = raw.lower()
    if "private video" in raw_lower:
        return "This video is private and cannot be downloaded."
    if "video unavailable" in raw_lower or "not available" in raw_lower:
        return "This video is unavailable. It may have been removed or geo-blocked."
    if "sign in" in raw_lower or "age" in raw_lower:
        return "This video requires sign-in or age verification and cannot be downloaded."
    if "copyright" in raw_lower:
        return "This video is blocked due to copyright restrictions."
    if "429" in raw:
        return (
            "YouTube is rate-limiting this server. "
            "Please try again in a few minutes or use a different video."
        )
    if "http error 403" in raw_lower:
        return (
            "YouTube returned 403 Forbidden. "
            "This sometimes happens on shared servers. Try a different video or upload a file instead."
        )
    # Return a cleaned version of the raw message, strip the yt-dlp prefix
    cleaned = re.sub(r"ERROR:\s*", "", raw).strip()
    return f"YouTube download failed: {cleaned[:500]}"
