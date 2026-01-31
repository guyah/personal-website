from __future__ import annotations

import io
import wave


def pcm16_mono_to_wav_bytes(pcm: bytes, *, sample_rate_hz: int) -> bytes:
	"""Wrap raw PCM16 mono audio in a WAV container."""
	buf = io.BytesIO()
	with wave.open(buf, "wb") as wf:
		wf.setnchannels(1)
		wf.setsampwidth(2)  # int16
		wf.setframerate(sample_rate_hz)
		wf.writeframes(pcm)
	return buf.getvalue()
