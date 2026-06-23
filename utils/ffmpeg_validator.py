"""
utils/ffmpeg_validator.py
─────────────────────────
Validates that FFmpeg and ffprobe are available on the system PATH.

Production risk: Streamlit Cloud requires `ffmpeg` in packages.txt.
Without it, yt-dlp and pydub will raise cryptic errors deep in the pipeline.
We surface this at startup so users see a clear message, not a traceback.
"""

import shutil
import subprocess
from typing import Optional


def get_ffmpeg_path() -> Optional[str]:
    """Return the resolved path to ffmpeg, or None if not found."""
    return shutil.which("ffmpeg")


def get_ffprobe_path() -> Optional[str]:
    """Return the resolved path to ffprobe, or None if not found."""
    return shutil.which("ffprobe")


def validate_ffmpeg(raise_on_missing: bool = False) -> bool:
    """
    Check that both ffmpeg and ffprobe are available and executable.

    Args:
        raise_on_missing: If True, raise RuntimeError with install instructions
                          when validation fails.

    Returns:
        True if both tools are found and working, False otherwise.

    Raises:
        RuntimeError: If raise_on_missing=True and tools are missing/broken.
    """
    ffmpeg_path  = get_ffmpeg_path()
    ffprobe_path = get_ffprobe_path()

    missing = []
    if not ffmpeg_path:
        missing.append("ffmpeg")
    if not ffprobe_path:
        missing.append("ffprobe")

    if missing:
        instructions = _install_instructions(missing)
        if raise_on_missing:
            raise RuntimeError(instructions)
        return False

    # Smoke-test: confirm they actually run
    for tool, path in [("ffmpeg", ffmpeg_path), ("ffprobe", ffprobe_path)]:
        try:
            result = subprocess.run(
                [path, "-version"],
                capture_output=True,
                timeout=10,
            )
            if result.returncode != 0:
                msg = f"{tool} found at {path} but returned non-zero exit code."
                if raise_on_missing:
                    raise RuntimeError(msg)
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            msg = f"{tool} could not be executed: {exc}"
            if raise_on_missing:
                raise RuntimeError(msg) from exc
            return False

    return True


def _install_instructions(missing: list[str]) -> str:
    tools = " and ".join(missing)
    return (
        f"{tools} not found on PATH.\n\n"
        "── Streamlit Cloud ──────────────────────────────────────\n"
        "Add this line to packages.txt in your repository root:\n\n"
        "    ffmpeg\n\n"
        "── Local (macOS) ────────────────────────────────────────\n"
        "    brew install ffmpeg\n\n"
        "── Local (Ubuntu / Debian) ──────────────────────────────\n"
        "    sudo apt-get update && sudo apt-get install -y ffmpeg\n\n"
        "── Local (Windows) ──────────────────────────────────────\n"
        "    Download from https://ffmpeg.org/download.html\n"
        "    and add the bin/ folder to your system PATH.\n"
    )
