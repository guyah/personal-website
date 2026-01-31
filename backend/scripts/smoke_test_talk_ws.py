#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import base64
import json
import os
import struct

import websockets


def gen_silence_pcm16(*, seconds: float = 1.0, sample_rate_hz: int = 16000) -> bytes:
	n = int(seconds * sample_rate_hz)
	# int16 zeros
	return struct.pack("<%dh" % n, *([0] * n))


async def main() -> None:
	url = os.environ.get("TALK_WS_URL", "ws://localhost:8000/ws/talk")
	print(f"connecting to {url}")

	pcm = gen_silence_pcm16(seconds=0.8)
	chunk = base64.b64encode(pcm).decode("ascii")

	async with websockets.connect(url, max_size=50 * 1024 * 1024) as ws:
		await ws.send(json.dumps({"type": "start", "sample_rate_hz": 16000}))
		await ws.send(json.dumps({"type": "audio_chunk", "data": chunk}))
		await ws.send(json.dumps({"type": "end"}))

		reply_audio_b64 = None
		# Read messages until we get the audio.
		for _ in range(200):
			msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=180))
			print("<-", msg.get("type"), msg.get("message") or "")
			if msg.get("type") == "audio":
				reply_audio_b64 = msg.get("data")
				break

		assert reply_audio_b64, "no audio received (timeout waiting for audio)"
		wav = base64.b64decode(reply_audio_b64)
		out = "talk_reply.wav"
		with open(out, "wb") as f:
			f.write(wav)
		print(f"wrote {out} ({len(wav)} bytes)")


if __name__ == "__main__":
	asyncio.run(main())
