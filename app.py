"""
app.py — TatvaAI  |  Streamlit AI Meeting Assistant
─────────────────────────────────────────────────────
Production-ready for Streamlit Community Cloud.

Key production changes vs. the original
────────────────────────────────────────
1.  Temp-dir lifecycle  — A TemporaryDirectory is created per analysis run and
    stored in session_state.  It is explicitly cleaned up in a finally block
    so no audio/WAV/chunk files linger after the pipeline finishes or crashes.

2.  Ephemeral ChromaDB — build_rag_chain() uses chromadb.EphemeralClient()
    (in-memory).  No vector_db/ folder is created; no cross-user collisions.

3.  Singleton models   — Whisper and the sentence-transformer embedding model
    are loaded once via @st.cache_resource so they survive page reruns without
    reloading 150–300 MB of weights each time.

4.  User-facing errors — Every exception is mapped to a friendly message.
    No raw Python tracebacks are ever shown to the user.

5.  Concurrent safety  — Every user's audio, chunks, and Chroma collection
    live in their own session_state.  No shared mutable global paths.

6.  File size guard    — upload_file_size is checked in audio_processor before
    any processing starts.

7.  Lazy heavy imports — ML libraries are imported only when the user clicks
    Analyse, keeping the initial page load fast.
"""

import logging
import os
import tempfile
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Load .env for local development — must run before any core/ imports
# so os.environ is populated when mistral_client.py resolves keys.
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TatvaAI — Meeting Intelligence",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --blue:        #2563eb;
    --blue-light:  #eff6ff;
    --blue-mid:    #dbeafe;
    --blue-dark:   #1d4ed8;
    --text-900:    #0f172a;
    --text-600:    #475569;
    --text-400:    #94a3b8;
    --border:      #e2e8f0;
    --bg:          #f8fafc;
    --surface:     #ffffff;
    --red:         #dc2626;
    --red-light:   #fef2f2;
    --green:       #16a34a;
    --green-light: #f0fdf4;
    --radius:      10px;
    --shadow-sm:   0 1px 3px rgba(0,0,0,.06), 0 1px 2px rgba(0,0,0,.04);
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: var(--bg) !important;
    color: var(--text-900) !important;
    -webkit-font-smoothing: antialiased;
}
.stApp { background: var(--bg) !important; }
[data-testid="stSidebar"] { display: none !important; }

/* Typography */
.ta-eyebrow { font-size:11px; font-weight:600; letter-spacing:.08em;
               text-transform:uppercase; color:var(--blue); }
.ta-heading  { font-size:26px; font-weight:700; color:var(--text-900);
               line-height:1.25; margin:0 0 4px; }
.ta-subheading { font-size:14px; color:var(--text-600); margin:0; line-height:1.6; }

/* Cards */
.ta-card { background:var(--surface); border:1px solid var(--border);
           border-radius:var(--radius); padding:24px; box-shadow:var(--shadow-sm); }
.ta-card-label { font-size:11px; font-weight:600; letter-spacing:.07em;
                 text-transform:uppercase; color:var(--text-400);
                 margin-bottom:10px; display:flex; align-items:center; gap:6px; }
.ta-card-body  { font-size:14px; line-height:1.75; color:var(--text-600); white-space:pre-wrap; }
.ta-card-title { font-size:18px; font-weight:600; color:var(--text-900); margin:0; }

/* Pills */
.ta-pill { display:inline-flex; align-items:center; gap:5px; font-size:12px;
           font-weight:500; padding:3px 10px; border-radius:999px; }
.ta-pill-blue  { background:var(--blue-mid);   color:var(--blue-dark); }
.ta-pill-green { background:var(--green-light); color:var(--green); }
.ta-pill-red   { background:var(--red-light);   color:var(--red); }

/* Inputs */
.stTextInput > div > div > input {
    border:1px solid var(--border) !important; background:var(--surface) !important;
    color:var(--text-900) !important; border-radius:8px !important;
    padding:10px 14px !important; font-family:'Inter',sans-serif !important;
    font-size:14px !important; box-shadow:var(--shadow-sm) !important;
    transition:border-color .15s !important;
}
.stTextInput > div > div > input:focus {
    border-color:var(--blue) !important; outline:none !important;
    box-shadow:0 0 0 3px rgba(37,99,235,.12) !important;
}

/* Primary button */
.stButton > button:not([kind="secondary"]) {
    background:var(--blue) !important; color:#fff !important;
    border:none !important; border-radius:8px !important;
    font-family:'Inter',sans-serif !important; font-size:14px !important;
    font-weight:600 !important; padding:10px 20px !important;
    text-transform:none !important; letter-spacing:0 !important;
    box-shadow:0 1px 2px rgba(37,99,235,.3) !important;
    transition:background .15s, box-shadow .15s !important;
}
.stButton > button:not([kind="secondary"]):hover {
    background:var(--blue-dark) !important;
    box-shadow:0 4px 12px rgba(37,99,235,.3) !important;
}
.stButton > button[kind="secondary"] {
    background:var(--surface) !important; color:var(--text-600) !important;
    border:1px solid var(--border) !important; border-radius:8px !important;
    font-family:'Inter',sans-serif !important; font-size:13px !important;
    font-weight:500 !important; padding:8px 16px !important;
    text-transform:none !important; letter-spacing:0 !important;
    box-shadow:var(--shadow-sm) !important;
}

/* Error card */
.ta-error { background:var(--red-light); border:1px solid #fecaca;
            border-radius:var(--radius); padding:16px 20px; margin:12px 0; }
.ta-error-title { font-size:14px; font-weight:600; color:var(--red); margin-bottom:6px; }
.ta-error-body  { font-size:13px; color:#7f1d1d; line-height:1.6; white-space:pre-wrap; }

/* Progress steps */
.ta-step { display:flex; align-items:center; gap:10px; padding:8px 12px;
           border-radius:8px; font-size:13px; font-weight:500; color:var(--text-600);
           background:var(--bg); border:1px solid var(--border); margin-bottom:6px; }
.ta-step-dot { width:8px; height:8px; border-radius:50%; flex-shrink:0; }
.step-pending { background:var(--border); }
.step-active  { background:var(--blue); box-shadow:0 0 0 3px var(--blue-mid);
                animation:ta-pulse 1.2s ease-in-out infinite; }
.step-done    { background:var(--green); }
@keyframes ta-pulse { 0%,100%{opacity:1} 50%{opacity:.5} }

/* Transcript */
.ta-transcript { font-size:13px; line-height:1.85; color:var(--text-600);
                 background:var(--bg); border:1px solid var(--border); border-radius:8px;
                 padding:16px; max-height:360px; overflow-y:auto;
                 white-space:pre-wrap; word-break:break-word; }

/* Tabs */
[data-baseweb="tab-list"] { gap:4px !important; border-bottom:1px solid var(--border) !important;
                             background:transparent !important; padding-bottom:0 !important; }
[data-baseweb="tab"] { font-family:'Inter',sans-serif !important; font-size:13px !important;
                       font-weight:500 !important; color:var(--text-600) !important;
                       padding:8px 14px !important; border-radius:6px 6px 0 0 !important;
                       background:transparent !important; border:none !important; }
[aria-selected="true"][data-baseweb="tab"] { color:var(--blue) !important;
    border-bottom:2px solid var(--blue) !important; font-weight:600 !important; }

/* Chat */
[data-testid="stChatMessage"] { background:var(--surface) !important;
    border:1px solid var(--border) !important; border-radius:var(--radius) !important;
    padding:14px 18px !important; box-shadow:var(--shadow-sm) !important; margin-bottom:8px !important; }
[data-testid="stChatInput"] > div { border:1px solid var(--border) !important;
    border-radius:8px !important; background:var(--surface) !important;
    box-shadow:var(--shadow-sm) !important; }
[data-testid="stChatInput"] > div:focus-within {
    border-color:var(--blue) !important;
    box-shadow:0 0 0 3px rgba(37,99,235,.12) !important; }

hr { border:none !important; border-top:1px solid var(--border) !important; margin:28px 0 !important; }

[data-testid="stMarkdownContainer"] p { color:var(--text-600) !important;
    font-size:14px !important; line-height:1.7 !important; }
label { color:var(--text-600) !important; font-size:13px !important; font-weight:500 !important; }
.stProgress > div > div > div { background:var(--blue) !important; }
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:4px; }
::-webkit-scrollbar-thumb:hover { background:var(--text-400); }
</style>
""", unsafe_allow_html=True)


# ─── Cached heavy resources (survive reruns; loaded once per server process) ────

@st.cache_resource(show_spinner=False)
def _check_ffmpeg() -> tuple[bool, str]:
    """Validate FFmpeg once per server process."""
    from utils.ffmpeg_validator import validate_ffmpeg, get_ffmpeg_path
    try:
        validate_ffmpeg(raise_on_missing=True)
        return True, get_ffmpeg_path() or "on PATH"
    except RuntimeError as exc:
        return False, str(exc)


@st.cache_resource(show_spinner=False)
def _load_whisper_model():
    """Load Whisper model once. Prevents re-loading 150 MB on every rerun."""
    from core.transcriber import _get_whisper_model
    return _get_whisper_model()


@st.cache_resource(show_spinner=False)
def _load_embedding_model():
    """Load sentence-transformer once. Prevents re-loading on every rerun."""
    from core.rag_engine import _get_embeddings
    return _get_embeddings()


# ─── FFmpeg gate ─────────────────────────────────────────────────────────────────
ffmpeg_ok, ffmpeg_info = _check_ffmpeg()

if not ffmpeg_ok:
    st.markdown(f"""
    <div class="ta-error">
        <div class="ta-error-title">⚠️ FFmpeg not found — audio processing unavailable</div>
        <div class="ta-error-body">{ffmpeg_info}</div>
    </div>
    """, unsafe_allow_html=True)
    with st.expander("Installation instructions", expanded=True):
        st.code(ffmpeg_info, language=None)
    st.stop()


# ─── Mistral API key startup check ───────────────────────────────────────────────
# Resolve the key now so we show a clear, actionable error immediately
# rather than crashing deep inside summarizer / extractor / rag_engine.

def _resolve_mistral_key_for_ui() -> tuple[bool, str]:
    """
    Try MISTRAL_API_KEY from st.secrets first, then os.environ.
    Returns (is_ok, diagnostic_message).
    """
    _PLACEHOLDERS = ("your_mistral_api_key_here", "your-key-here", "replace_me", "<your")

    # Priority 1 — Streamlit Cloud secrets
    try:
        key = st.secrets.get("MISTRAL_API_KEY", "")
        if key and key.strip() and not any(key.strip().lower().startswith(p) for p in _PLACEHOLDERS):
            if len(key.strip()) >= 20:
                print("Mistral Key Found: True (source: st.secrets)")   # diagnostic log
                logger.info("MISTRAL_API_KEY resolved from st.secrets")
                return True, "st.secrets"
    except Exception:
        pass  # secrets not available outside Streamlit Cloud

    # Priority 2 — Environment variable (populated by load_dotenv)
    key = st.secrets.get("MISTRAL_API_KEY") or os.getenv("MISTRAL_API_KEY")
    print(f"Mistral Key Found: {bool(key and key.strip())}")             # diagnostic log
    logger.info("MISTRAL_API_KEY present in env: %s", bool(key and key.strip()))

    if not key or not key.strip():
        return False, (
            "MISTRAL_API_KEY is not set.\n\n"
            "• Local development → add MISTRAL_API_KEY=<your-key> to your .env file\n"
            "• Streamlit Cloud   → add MISTRAL_API_KEY under App Settings → Secrets\n\n"
            "Get your key at https://console.mistral.ai/api-keys/"
        )
    if any(key.strip().lower().startswith(p) for p in _PLACEHOLDERS):
        return False, (
            "MISTRAL_API_KEY is still a placeholder value.\n"
            "Replace it with your real key from https://console.mistral.ai/api-keys/"
        )
    if len(key.strip()) < 20:
        return False, (
            f"MISTRAL_API_KEY looks too short (length={len(key.strip())}).\n"
            "Double-check the value in your .env or Streamlit Secrets."
        )
    return True, "os.environ / .env"


_mistral_ok, _mistral_info = _resolve_mistral_key_for_ui()

if not _mistral_ok:
    st.markdown(f"""
    <div class="ta-error">
        <div class="ta-error-title">🔑 Mistral API key not configured</div>
        <div class="ta-error-body">{_mistral_info}</div>
    </div>
    """, unsafe_allow_html=True)
    st.info(
        "Once you've added the key, click ☰ → Rerun, or redeploy on Streamlit Cloud.",
        icon="ℹ️",
    )
    st.stop()


# ─── Session State Init ──────────────────────────────────────────────────────────
_DEFAULTS = {
    "result":          None,
    "chat_history":    [],
    "pipeline_done":   False,
    "pipeline_steps":  {},
    "uploaded_tmp":    None,   # path of last saved upload (cleaned on next upload)
    "work_dir":        None,   # TemporaryDirectory path for current analysis run
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── Helpers ─────────────────────────────────────────────────────────────────────

def _cleanup_work_dir() -> None:
    """Delete the current analysis temp dir if it exists."""
    wd = st.session_state.get("work_dir")
    if wd and os.path.isdir(wd):
        try:
            import shutil
            shutil.rmtree(wd, ignore_errors=True)
        except Exception:
            pass
    st.session_state.work_dir = None


def _step_dot(key: str) -> str:
    s = st.session_state.pipeline_steps.get(key, "pending")
    return {"active": "step-active", "done": "step-done"}.get(s, "step-pending")


def _render_step(label: str, key: str, icon: str) -> None:
    css = _step_dot(key)
    st.markdown(
        f'<div class="ta-step">'
        f'<div class="ta-step-dot {css}"></div>{icon}&nbsp;{label}</div>',
        unsafe_allow_html=True,
    )


def _save_upload(uploaded_file) -> str:
    """Save UploadedFile bytes to a NamedTemporaryFile and return its path."""
    # Clean up the previous upload if user changed their file
    prev = st.session_state.uploaded_tmp
    if prev:
        try:
            os.unlink(prev)
        except OSError:
            pass
    suffix = Path(uploaded_file.name).suffix or ".tmp"
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=suffix, prefix="ta_upload_"
    ) as f:
        f.write(uploaded_file.getbuffer())
        st.session_state.uploaded_tmp = f.name
        return f.name


def _friendly_error(exc: Exception) -> str:
    """
    Map any exception to a user-readable string. Never exposes raw tracebacks.

    Auth errors from Mistral are handled first via MistralAuthError /
    classify_api_error so the message is always specific and actionable.
    """
    # Import here (lazy) so startup is unaffected if core/ has issues
    try:
        from core.mistral_client import MistralAuthError, classify_api_error
        if isinstance(exc, MistralAuthError):
            return (
                "Unable to connect to Mistral AI — API key issue.\n\n"
                + str(exc)
                + "\n\nVerify your MISTRAL_API_KEY in .env or Streamlit Secrets."
            )
    except ImportError:
        pass

    msg = str(exc)
    low = msg.lower()

    # ── Mistral / LLM errors ─────────────────────────────────────────────────
    if "401" in msg or "unauthorized" in low:
        return (
            "Mistral API returned 401 Unauthorized.\n"
            "Your MISTRAL_API_KEY is invalid or has been revoked.\n"
            "Generate a new key at https://console.mistral.ai/api-keys/ "
            "and update your .env / Streamlit Secrets."
        )
    if "mistral" in low or "mistral_api_key" in low:
        try:
            from core.mistral_client import classify_api_error
            return classify_api_error(exc)
        except ImportError:
            pass
        return (
            "Mistral AI call failed.\n"
            "Check that MISTRAL_API_KEY is set correctly in .env or Streamlit Secrets."
        )

    # ── FFmpeg / audio errors ─────────────────────────────────────────────────
    if "ffprobe" in low or "ffmpeg" in low:
        return (
            "FFmpeg is not available on this server.\n"
            "Add 'ffmpeg' to packages.txt and redeploy."
        )

    # ── YouTube errors ────────────────────────────────────────────────────────
    if "private video" in low:
        return "This YouTube video is private and cannot be downloaded."
    if "unavailable" in low or "not available" in low:
        return "This video is unavailable. It may be geo-blocked or removed."
    if "sign in" in low or "age" in low:
        return "This video requires sign-in or age verification."
    if "429" in msg or "rate limit" in low:
        return "YouTube is rate-limiting this server. Please try again in a few minutes."
    if "403" in msg and "mistral" not in low:
        return (
            "YouTube returned 403 Forbidden — common on shared cloud servers.\n"
            "Try uploading the audio file directly instead of using a URL."
        )

    # ── Other errors ──────────────────────────────────────────────────────────
    if "max_file" in low or "exceeds" in low or "too large" in low:
        return msg  # already user-friendly from audio_processor
    if "api key" in low:
        return (
            "An API key is missing or invalid.\n"
            "Check that MISTRAL_API_KEY (and SARVAM_API_KEY if using Hinglish) "
            "are set in Streamlit Cloud secrets."
        )
    if "sarvam" in low:
        return f"Sarvam AI transcription failed: {msg}"
    if "file not found" in low or "no such file" in low:
        return "A required file could not be found. Please try again."
    if "empty" in low or "no audio" in low:
        return "The audio file appears to be empty or silent."

    # Generic fallback — show the message, never the full traceback
    return f"Processing failed: {msg}"


# ─── Header ──────────────────────────────────────────────────────────────────────
hdr_left, hdr_right = st.columns([1, 1])
with hdr_left:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding:8px 0 24px">
        <div style="width:32px;height:32px;background:#2563eb;border-radius:8px;
                    display:flex;align-items:center;justify-content:center;font-size:16px">🎙️</div>
        <div>
            <div style="font-size:16px;font-weight:700;color:#0f172a;line-height:1">TatvaAI</div>
            <div style="font-size:11px;color:#94a3b8;letter-spacing:.04em">Meeting Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
with hdr_right:
    st.markdown(
        '<div style="display:flex;justify-content:flex-end;padding-top:12px">'
        '<span class="ta-pill ta-pill-green">● FFmpeg ready</span></div>',
        unsafe_allow_html=True,
    )

# ─── Hero ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:12px 0 32px">
    <p class="ta-eyebrow">AI-powered analysis</p>
    <h1 class="ta-heading">Understand any meeting in seconds</h1>
    <p class="ta-subheading" style="max-width:520px">
        Paste a YouTube link or upload a recording. TatvaAI transcribes, summarises,
        extracts decisions and action items, then lets you chat with the transcript.
    </p>
</div>
""", unsafe_allow_html=True)

# ─── Input Panel ──────────────────────────────────────────────────────────────────
with st.container():
    st.markdown('<div class="ta-card">', unsafe_allow_html=True)

    inp_col, lang_col = st.columns([3, 1], gap="medium")
    with inp_col:
        input_mode = st.radio(
            "Source",
            ["YouTube URL", "Upload file"],
            horizontal=True,
            label_visibility="collapsed",
        )
    with lang_col:
        language = st.selectbox("Language", ["english", "hinglish"], index=0)

    source: str = ""
    if input_mode == "YouTube URL":
        source = st.text_input(
            "YouTube URL",
            placeholder="https://youtube.com/watch?v=…",
            label_visibility="collapsed",
        )
    else:
        uploaded = st.file_uploader(
            "Audio or video file",
            type=["mp3", "mp4", "wav", "m4a", "ogg", "flac", "webm", "mkv", "mov"],
            label_visibility="collapsed",
        )
        if uploaded:
            source = _save_upload(uploaded)
            st.markdown(
                f'<span class="ta-pill ta-pill-blue">📎 {uploaded.name} '
                f'({uploaded.size // 1024} KB)</span>',
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)

btn_col, _ = st.columns([2, 5])
with btn_col:
    run_btn = st.button("Analyse meeting", use_container_width=True)

st.markdown("---")

# ─── Pipeline ─────────────────────────────────────────────────────────────────────
if run_btn:
    if not (source or "").strip():
        st.error("Please enter a YouTube URL or upload a file before analysing.")
    else:
        # ── Lazy imports — keep initial page load fast ────────────────────────
        from utils.audio_processor import process_input
        from core.transcriber    import transcribe_all
        from core.summarizer     import summarize, generate_title
        from core.extractor      import (
            extract_action_items,
            extract_key_decisions,
            extract_questions,
        )
        from core.rag_engine     import build_rag_chain

        # Trigger model pre-loading (no-ops if already cached)
        _load_whisper_model()
        _load_embedding_model()

        # ── Reset state from any previous run ────────────────────────────────
        _cleanup_work_dir()
        st.session_state.pipeline_done  = False
        st.session_state.result         = None
        st.session_state.chat_history   = []
        st.session_state.pipeline_steps = {}

        progress_ph = st.empty()
        error_ph    = st.empty()

        STEPS = [
            ("audio",      "🔊", "Processing audio"),
            ("transcript", "📝", "Transcribing"),
            ("title",      "🏷️",  "Generating title"),
            ("summary",    "📋", "Summarising"),
            ("extract",    "🔍", "Extracting insights"),
            ("rag",        "🧠", "Building RAG index"),
        ]

        def _set_step(key: str, state: str) -> None:
            st.session_state.pipeline_steps[key] = state

        def _render_progress() -> None:
            with progress_ph.container():
                st.markdown('<div class="ta-card" style="padding:20px 24px">', unsafe_allow_html=True)
                st.markdown(
                    '<div style="font-size:13px;font-weight:600;color:#0f172a;margin-bottom:14px">'
                    'Running analysis…</div>',
                    unsafe_allow_html=True,
                )
                for key, icon, label in STEPS:
                    _render_step(label, key, icon)
                st.markdown("</div>", unsafe_allow_html=True)

        # ── Create a per-run temp dir that auto-cleans on any exit path ──────
        work_tmp = tempfile.mkdtemp(prefix="tatvaai_run_")
        st.session_state.work_dir = work_tmp

        try:
            _render_progress()

            _set_step("audio", "active")
            chunks = process_input(source, work_dir=work_tmp)
            _set_step("audio", "done")
            _render_progress()

            _set_step("transcript", "active")
            transcript = transcribe_all(chunks, language)
            _set_step("transcript", "done")
            _render_progress()

            # Audio chunks are no longer needed — free disk space now
            # (the WAV files inside work_tmp will be removed later in finally,
            #  but Python won't reclaim pydub memory until del)
            del chunks

            _set_step("title", "active")
            title = generate_title(transcript)
            _set_step("title", "done")
            _render_progress()

            _set_step("summary", "active")
            summary = summarize(transcript)
            _set_step("summary", "done")
            _render_progress()

            _set_step("extract", "active")
            action_items = extract_action_items(transcript)
            decisions    = extract_key_decisions(transcript)
            questions    = extract_questions(transcript)
            _set_step("extract", "done")
            _render_progress()

            _set_step("rag", "active")
            rag_chain = build_rag_chain(transcript)
            _set_step("rag", "done")
            _render_progress()

            st.session_state.result = {
                "title":          title,
                "transcript":     transcript,
                "summary":        summary,
                "action_items":   action_items,
                "key_decisions":  decisions,
                "open_questions": questions,
                "rag_chain":      rag_chain,
            }
            st.session_state.pipeline_done = True

        except Exception as exc:
            logger.exception("Pipeline failed")
            # Mark active step as pending so the UI doesn't show a stuck spinner
            for k, _, _ in STEPS:
                if st.session_state.pipeline_steps.get(k) == "active":
                    st.session_state.pipeline_steps[k] = "pending"

            # Distinguish Mistral auth failures from everything else
            try:
                from core.mistral_client import MistralAuthError
                is_auth_error = isinstance(exc, MistralAuthError)
            except ImportError:
                is_auth_error = False

            msg = _friendly_error(exc)

            if is_auth_error:
                error_ph.markdown(f"""
                <div class="ta-error">
                    <div class="ta-error-title">🔑 Mistral Authentication Failed</div>
                    <div class="ta-error-body">{msg}</div>
                </div>
                """, unsafe_allow_html=True)
                error_ph.info(
                    "Fix the API key in your .env or Streamlit Secrets, then click "
                    "**Analyse meeting** again to retry.",
                    icon="ℹ️",
                )
            else:
                error_ph.markdown(f"""
                <div class="ta-error">
                    <div class="ta-error-title">Analysis failed</div>
                    <div class="ta-error-body">{msg}</div>
                </div>
                """, unsafe_allow_html=True)

        finally:
            # Always remove audio/WAV/chunk files from disk
            _cleanup_work_dir()

        if st.session_state.pipeline_done:
            progress_ph.empty()
            st.rerun()


# ─── Results ──────────────────────────────────────────────────────────────────────
if st.session_state.result:
    r = st.session_state.result

    # Meeting title banner
    st.markdown(f"""
    <div class="ta-card" style="margin-bottom:20px">
        <div class="ta-card-label">📌 Meeting</div>
        <div class="ta-card-title">{r['title']}</div>
    </div>
    """, unsafe_allow_html=True)

    # Tabbed result sections
    tab_summary, tab_actions, tab_decisions, tab_questions, tab_transcript = st.tabs([
        "Summary", "Action items", "Key decisions", "Open questions", "Transcript",
    ])

    with tab_summary:
        st.markdown(f"""
        <div class="ta-card" style="margin-top:16px">
            <div class="ta-card-label">📋 Summary</div>
            <div class="ta-card-body">{r['summary']}</div>
        </div>""", unsafe_allow_html=True)

    with tab_actions:
        st.markdown(f"""
        <div class="ta-card" style="margin-top:16px">
            <div class="ta-card-label">✅ Action items</div>
            <div class="ta-card-body">{r['action_items']}</div>
        </div>""", unsafe_allow_html=True)

    with tab_decisions:
        st.markdown(f"""
        <div class="ta-card" style="margin-top:16px">
            <div class="ta-card-label">🔑 Key decisions</div>
            <div class="ta-card-body">{r['key_decisions']}</div>
        </div>""", unsafe_allow_html=True)

    with tab_questions:
        st.markdown(f"""
        <div class="ta-card" style="margin-top:16px">
            <div class="ta-card-label">❓ Open questions</div>
            <div class="ta-card-body">{r['open_questions']}</div>
        </div>""", unsafe_allow_html=True)

    with tab_transcript:
        st.markdown(
            f'<div class="ta-transcript" style="margin-top:16px">{r["transcript"]}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Chat ──────────────────────────────────────────────────────────────────
    st.markdown(
        '<div style="font-size:16px;font-weight:600;color:#0f172a;margin-bottom:4px">'
        'Ask about this meeting</div>'
        '<div style="font-size:13px;color:#64748b;margin-bottom:16px">'
        'The assistant has full context from the transcript.</div>',
        unsafe_allow_html=True,
    )

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("What were the main decisions? Who owns which action items?")
    if user_input:
        from core.rag_engine import ask_question
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    answer = ask_question(r["rag_chain"], user_input)
                except Exception as exc:
                    logger.error("ask_question failed: %s", exc)
                    answer = f"Sorry, I could not generate an answer: {_friendly_error(exc)}"
            st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    if st.session_state.chat_history:
        if st.button("Clear conversation", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

else:
    # ── Empty state ───────────────────────────────────────────────────────────
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;
                text-align:center;padding:60px 24px 80px">
        <div style="width:56px;height:56px;background:#eff6ff;border-radius:14px;
                    display:flex;align-items:center;justify-content:center;
                    font-size:26px;margin-bottom:16px">🎙️</div>
        <div style="font-size:18px;font-weight:600;color:#0f172a;margin-bottom:6px">
            Ready to analyse
        </div>
        <div style="font-size:14px;color:#64748b;max-width:360px;line-height:1.7">
            Paste a YouTube URL or upload a recording above, choose a language,
            and click <strong>Analyse meeting</strong>.
        </div>
        <div style="display:flex;gap:8px;margin-top:20px;flex-wrap:wrap;justify-content:center">
            <span class="ta-pill ta-pill-blue">Transcription</span>
            <span class="ta-pill ta-pill-blue">Summarisation</span>
            <span class="ta-pill ta-pill-blue">Chat</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
