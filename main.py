"""
main.py — TatvaAI CLI entry point
───────────────────────────────────
Production-ready: uses a TemporaryDirectory for all intermediate files.
The temp dir is cleaned up automatically on exit (even on crash).
"""

import tempfile
import shutil
from dotenv import load_dotenv

load_dotenv()

from utils.audio_processor import process_input
from core.transcriber      import transcribe_all
from core.summarizer       import summarize, generate_title
from core.extractor        import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine       import build_rag_chain, ask_question


def run_pipeline(source: str, language: str = "english") -> dict:
    """
    Run the full TatvaAI pipeline.

    All intermediate audio/WAV/chunk files are stored in a TemporaryDirectory
    that is deleted when this function returns, regardless of success or failure.

    Args:
        source:   YouTube URL or local file path.
        language: "english" (Whisper) or "hinglish" (Sarvam AI).

    Returns:
        dict with keys: title, transcript, summary, action_items,
                        key_decisions, open_questions, rag_chain.
    """
    print("Starting TatvaAI pipeline…")

    work_dir = tempfile.mkdtemp(prefix="tatvaai_cli_")
    try:
        print(f"  Work directory: {work_dir}")

        print("  [1/6] Processing audio…")
        chunks = process_input(source, work_dir=work_dir)
        print(f"        → {len(chunks)} chunk(s) created")

        print("  [2/6] Transcribing…")
        transcript = transcribe_all(chunks, language)
        print(f"        → {len(transcript.split())} words transcribed")
        print(f"        Preview: {transcript[:200]}…")

        # Free chunk memory / disk before LLM calls
        del chunks

        print("  [3/6] Generating title…")
        title = generate_title(transcript)
        print(f"        → {title}")

        print("  [4/6] Summarising…")
        summary = summarize(transcript)

        print("  [5/6] Extracting action items, decisions, questions…")
        action_items = extract_action_items(transcript)
        decisions    = extract_key_decisions(transcript)
        questions    = extract_questions(transcript)

        print("  [6/6] Building RAG index…")
        rag_chain = build_rag_chain(transcript)
        print("  ✅ Pipeline complete.")

        return {
            "title":          title,
            "transcript":     transcript,
            "summary":        summary,
            "action_items":   action_items,
            "key_decisions":  decisions,
            "open_questions": questions,
            "rag_chain":      rag_chain,
        }

    finally:
        # Always clean up — no WAV/chunk files left behind
        shutil.rmtree(work_dir, ignore_errors=True)
        print(f"  Temp dir removed: {work_dir}")


if __name__ == "__main__":
    source   = input("Enter YouTube URL or local file path: ").strip()
    language = input("Language (english/hinglish) [english]: ").strip() or "english"

    result = run_pipeline(source, language)

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"📌 Title: {result['title']}")
    print(f"\n📋 Summary:\n{result['summary']}")
    print(f"\n✅ Action Items:\n{result['action_items']}")
    print(f"\n🔑 Key Decisions:\n{result['key_decisions']}")
    print(f"\n❓ Open Questions:\n{result['open_questions']}")
    print(sep)

    print("\n💬 Chat with your meeting (type 'exit' to quit)\n")
    rag_chain = result["rag_chain"]
    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit", "q", ""):
            if not question:
                continue
            print("👋 Goodbye!")
            break
        answer = ask_question(rag_chain, question)
        print(f"\n🤖 Assistant: {answer}\n")
