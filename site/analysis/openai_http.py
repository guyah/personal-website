"""Tiny OpenAI HTTP client (no external deps).

We avoid the official SDK to keep this repo lightweight and compatible.

Requires env:
- OPENAI_API_KEY

Uses:
- Responses API for text generation (timing + usage)
- Audio speech API for TTS WAV generation
"""

from __future__ import annotations

import json
import os
import time
import subprocess
from dataclasses import dataclass


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")


@dataclass
class HttpResult:
    ok: bool
    status: int
    took_ms: float
    json: dict | None = None
    bytes: bytes | None = None


def _request(url: str, payload: dict, *, timeout_s: float = 60.0) -> HttpResult:
    """POST JSON using curl.

    Python SSL cert validation is flaky on this machine (observed CERT_VERIFY_FAILED),
    while curl works reliably.
    """

    data = json.dumps(payload)
    t0 = time.perf_counter()

    # We ask curl to print the HTTP status code on stderr-ish marker, then split.
    cmd = [
        "curl",
        "-sS",
        "-L",
        "--max-time",
        str(int(timeout_s)),
        "-H",
        f"Authorization: Bearer {OPENAI_API_KEY}",
        "-H",
        "Content-Type: application/json",
        "-d",
        data,
        "-w",
        "\n__STATUS__:%{http_code}\n",
        url,
    ]

    out = subprocess.check_output(cmd, text=False)
    took = (time.perf_counter() - t0) * 1000

    raw, status_line = out.rsplit(b"\n__STATUS__:", 1)
    status = int(status_line.strip().splitlines()[0])

    ok = 200 <= status < 300

    # Try JSON first; otherwise return bytes
    try:
        j = json.loads(raw.decode("utf-8"))
        return HttpResult(ok, status, took, json=j)
    except Exception:
        return HttpResult(ok, status, took, bytes=raw)


def responses_create(
    *,
    model: str,
    input_text: str,
    temperature: float | None = None,
    top_p: float | None = None,
    max_output_tokens: int | None = None,
    timeout_s: float = 60.0,
) -> HttpResult:
    payload: dict = {
        "model": model,
        "input": input_text,
    }
    if temperature is not None:
        payload["temperature"] = float(temperature)
    if top_p is not None:
        payload["top_p"] = float(top_p)
    if max_output_tokens is not None:
        payload["max_output_tokens"] = int(max_output_tokens)

    return _request("https://api.openai.com/v1/responses", payload, timeout_s=timeout_s)


def tts_speech(
    *,
    model: str,
    voice: str,
    text: str,
    response_format: str = "mp3",
    timeout_s: float = 60.0,
) -> HttpResult:
    """TTS request.

    Note: WAV responses can include placeholder RIFF sizes which break naive duration
    estimation. MP3 is easier to measure reliably via ffprobe.
    """

    payload = {
        "model": model,
        "voice": voice,
        "input": text,
        "response_format": response_format,
    }
    return _request("https://api.openai.com/v1/audio/speech", payload, timeout_s=timeout_s)
