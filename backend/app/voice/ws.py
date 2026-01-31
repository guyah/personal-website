from __future__ import annotations

import base64
import json
from dataclasses import dataclass

from fastapi import WebSocket

from .openai_pipecat import OpenAIPipecatLLMProvider, OpenAIPipecatSTTProvider, OpenAIPipecatTTSProvider
from .providers import LLMProvider, STTProvider, TTSProvider
from .system_prompt import SYSTEM_PROMPT


@dataclass
class TalkSessionConfig:
	input_sample_rate_hz: int = 16000


class TalkSession:
	def __init__(
		self,
		*,
		stt: STTProvider,
		llm: LLMProvider,
		tts: TTSProvider,
		cfg: TalkSessionConfig,
	):
		self.stt = stt
		self.llm = llm
		self.tts = tts
		self.cfg = cfg
		self._pcm_parts: list[bytes] = []

	def add_pcm16_chunk(self, chunk: bytes) -> None:
		self._pcm_parts.append(chunk)

	def clear(self) -> None:
		self._pcm_parts = []

	async def finalize_once(self) -> tuple[str, str, bytes]:
		"""Returns (transcript, reply_text, reply_wav_bytes)."""
		pcm = b"".join(self._pcm_parts)
		# wrap pcm as wav (OpenAISTTService expects wav bytes)
		from .wav import pcm16_mono_to_wav_bytes

		wav_in = pcm16_mono_to_wav_bytes(pcm, sample_rate_hz=self.cfg.input_sample_rate_hz)
		stt = await self.stt.transcribe_wav(wav_in)
		user_text = stt.text or ""
		if not user_text.strip():
			user_text = "(I couldn't catch the audio. Ask the user to try again.)"

		reply_text = await self.llm.respond(system_prompt=SYSTEM_PROMPT, user_text=user_text)
		if not reply_text.strip():
			reply_text = "Sorry — I didn't get that. Can you try again?"

		tts = await self.tts.synthesize_wav(reply_text)
		return (stt.text, reply_text, tts.wav_bytes)


def build_default_session(cfg: TalkSessionConfig) -> TalkSession:
	stt = OpenAIPipecatSTTProvider(sample_rate_hz=cfg.input_sample_rate_hz)
	llm = OpenAIPipecatLLMProvider()
	tts = OpenAIPipecatTTSProvider()
	return TalkSession(stt=stt, llm=llm, tts=tts, cfg=cfg)


async def talk_websocket_handler(websocket: WebSocket) -> None:
	"""WebSocket protocol (JSON messages):

	Client -> Server:
	- {"type":"start","sample_rate_hz":16000}
	- {"type":"audio_chunk","data":"<base64 pcm16 mono>"}
	- {"type":"end"}

	Server -> Client:
	- {"type":"status","message":"..."}
	- {"type":"transcript","text":"..."}
	- {"type":"reply","text":"..."}
	- {"type":"audio","mime":"audio/wav","data":"<base64 wav>"}
	"""
	await websocket.accept()

	cfg = TalkSessionConfig()
	session = build_default_session(cfg)

	await websocket.send_text(json.dumps({"type": "status", "message": "connected"}))

	try:
		while True:
			msg = await websocket.receive_text()
			payload = json.loads(msg)
			mtype = payload.get("type")

			if mtype == "start":
				session.clear()
				if payload.get("sample_rate_hz"):
					cfg.input_sample_rate_hz = int(payload["sample_rate_hz"])
				await websocket.send_text(json.dumps({"type": "status", "message": "recording"}))

			elif mtype == "audio_chunk":
				data_b64 = payload.get("data")
				if not data_b64:
					continue
				session.add_pcm16_chunk(base64.b64decode(data_b64))

			elif mtype == "end":
				await websocket.send_text(json.dumps({"type": "status", "message": "processing"}))
				transcript, reply_text, wav_out = await session.finalize_once()
				await websocket.send_text(json.dumps({"type": "transcript", "text": transcript}))
				await websocket.send_text(json.dumps({"type": "reply", "text": reply_text}))
				await websocket.send_text(
					json.dumps(
						{
							"type": "audio",
							"mime": "audio/wav",
							"data": base64.b64encode(wav_out).decode("ascii"),
						}
					)
				)
				await websocket.send_text(json.dumps({"type": "status", "message": "ready"}))

			else:
				await websocket.send_text(
					json.dumps({"type": "status", "message": f"unknown message type: {mtype}"})
				)
	except Exception as e:
		# Don’t fully swallow server-side errors; they make debugging impossible.
		try:
			await websocket.send_text(json.dumps({"type": "status", "message": f"error: {e}"}))
		except Exception:
			pass
		raise
