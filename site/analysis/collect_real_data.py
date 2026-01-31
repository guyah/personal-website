"""Collect small, real datasets for blog plots (no synthetic).

Writes CSVs under site/analysis/data/.

This is intentionally small-sample to keep cost low.

Run:
  source .venv/bin/activate
  python site/analysis/collect_real_data.py

Env:
  OPENAI_API_KEY (required)
  MODEL_TEXT (default: gpt-5.2)
  MODEL_TTS  (default: gpt-4o-mini-tts)
"""

from __future__ import annotations

import csv
import os
import time
import wave
from pathlib import Path

# Allow running as a script (no package install)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # site/

from analysis.openai_http import responses_create, tts_speech


ROOT = Path(__file__).resolve().parents[1]  # site/
DATA = ROOT / "analysis" / "data"

MODEL_TEXT = os.environ.get("MODEL_TEXT", "gpt-5.2")
MODEL_TTS = os.environ.get("MODEL_TTS", "gpt-4o-mini-tts")
TTS_VOICE = os.environ.get("TTS_VOICE", "alloy")


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def collect_voice_latency(n: int = 25) -> None:
    """Measure LLM response latency (time-to-first-byte-ish at HTTP level).

    This isn't token-level first-token latency (we're not streaming), but it's real,
    reproducible, and maps to end-user perceived delay in many server setups.
    """

    prompt = """You are a concise assistant. Answer in one sentence: What is latency in a voice agent?"""

    rows: list[list[object]] = []
    settings = [
        ("t0.2", 0.2, 1.0, 64),
        ("t0.8", 0.8, 1.0, 128),
        ("t1.2", 1.2, 1.0, 128),
    ]

    for label, temp, top_p, max_out in settings:
        for i in range(n):
            r = responses_create(
                model=MODEL_TEXT,
                input_text=prompt,
                temperature=temp,
                top_p=top_p,
                max_output_tokens=max_out,
                timeout_s=60,
            )
            if not r.ok:
                rows.append([label, i, temp, top_p, max_out, "err", r.status, r.took_ms, "", ""])
                continue
            usage = (r.json or {}).get("usage", {})
            rows.append(
                [
                    label,
                    i,
                    temp,
                    top_p,
                    max_out,
                    "ok",
                    r.status,
                    round(r.took_ms, 2),
                    usage.get("input_tokens", ""),
                    usage.get("output_tokens", ""),
                ]
            )
            time.sleep(0.15)

    _write_csv(
        DATA / "voice_latency_samples.csv",
        ["setting", "trial", "temperature", "top_p", "max_output_tokens", "ok", "status", "took_ms", "input_tokens", "output_tokens"],
        rows,
    )


def _wav_duration_s(wav_bytes: bytes) -> float:
    """Robust duration extraction via ffprobe.

    Some TTS WAVs can have placeholder RIFF sizes; `wave` then misreports.
    ffprobe is more reliable.
    """

    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as f:
        f.write(wav_bytes)
        f.flush()
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                f.name,
            ],
            text=True,
        ).strip()
        return float(out)


def collect_tts_durations() -> None:
    """Generate real TTS WAVs and measure duration in seconds."""

    texts = [
        "One short sentence.",
        "This is a slightly longer sentence to measure audio duration more reliably.",
        "Here is a paragraph-length sample. It should take a few seconds to speak, and it helps estimate how duration scales with text length.",
        "Arabic sample: مرحبا! كيفك اليوم؟ هذا مثال قصير لقياس مدة تحويل النص إلى كلام.",
    ]

    rows: list[list[object]] = []
    for idx, text in enumerate(texts):
        # Use MP3 for robust duration measurement
        r = tts_speech(model=MODEL_TTS, voice=TTS_VOICE, text=text, response_format="mp3", timeout_s=60)
        if not r.ok or not r.bytes:
            rows.append([idx, len(text), text, "err", r.status, r.took_ms, ""])
            continue
        dur = _wav_duration_s(r.bytes)
        rows.append([idx, len(text), text, "ok", r.status, round(r.took_ms, 2), round(dur, 3)])
        time.sleep(0.2)

    _write_csv(
        DATA / "tts_durations.csv",
        ["sample_id", "chars", "text", "ok", "status", "took_ms", "duration_s"],
        rows,
    )


def collect_token_budgets(n: int = 40) -> None:
    """Estimate chars-per-token using real prompt tokenization via API usage.

    We send texts and read back `usage.input_tokens`.
    """

    samples = [
        "Hello world.",
        "Write a concise explanation of retrieval-augmented generation.",
        """Here is a longer piece of text intended to be tokenized by a modern LLM. It includes punctuation, numbers (12345), and some code: `print('hi')`.""",
        "Arabic: هذا نص عربي لاختبار عدد المحارف لكل توكن.",
    ]

    # Add real site content snippets (blog titles + descriptions) as additional real-world prompts.
    try:
        import glob

        paths = sorted(glob.glob(str((ROOT / "content" / "blog" / "*.md").as_posix())))
        for p in paths[:20]:
            try:
                txt = Path(p).read_text(encoding="utf-8")
                samples.append(txt[:600])
            except Exception:
                pass
    except Exception:
        pass

    rows: list[list[object]] = []
    for i, text in enumerate(samples[:n]):
        r = responses_create(
            model=MODEL_TEXT,
            input_text=text,
            temperature=0,
            top_p=1,
            max_output_tokens=16,
            timeout_s=60,
        )
        if not r.ok:
            rows.append([i, len(text), "err", r.status, r.took_ms, ""])
            continue
        usage = (r.json or {}).get("usage", {})
        in_toks = usage.get("input_tokens", "")
        ratio = (len(text) / in_toks) if isinstance(in_toks, (int, float)) and in_toks else ""
        rows.append([i, len(text), "ok", r.status, round(r.took_ms, 2), in_toks, "" if ratio == "" else round(ratio, 3)])
        time.sleep(0.15)

    _write_csv(
        DATA / "chars_per_token.csv",
        ["sample_id", "chars", "ok", "status", "took_ms", "input_tokens", "chars_per_token"],
        rows,
    )


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    print("[collect] voice latency...")
    collect_voice_latency()
    print("[collect] tts durations...")
    collect_tts_durations()
    print("[collect] chars/token...")
    collect_token_budgets()
    print(f"[collect] done. wrote CSVs under {DATA}")


if __name__ == "__main__":
    main()
