from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass
from time import perf_counter
from typing import Optional

from fastapi import WebSocket

from .openai_pipecat import OpenAIPipecatLLMProvider, OpenAIPipecatSTTProvider, OpenAIPipecatTTSProvider
from .providers import LLMProvider, STTProvider, TTSProvider
from .system_prompt import SYSTEM_PROMPT
from .wav import pcm16_mono_to_wav_bytes


@dataclass
class TalkSessionConfig:
	input_sample_rate_hz: int = 16000
	stt_chunk_seconds: float = 1.2


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
		self._partial_pcm: bytearray = bytearray()
		self._partial_text_parts: list[str] = []
		self._chunk_bytes = int(self.cfg.input_sample_rate_hz * 2 * self.cfg.stt_chunk_seconds)

	def reset(self) -> None:
		self._pcm_parts = []
		self._partial_pcm = bytearray()
		self._partial_text_parts = []

	def add_pcm16_chunk(self, chunk: bytes) -> None:
		self._pcm_parts.append(chunk)
		self._partial_pcm.extend(chunk)

	def pop_partial_chunk(self) -> Optional[bytes]:
		if len(self._partial_pcm) < self._chunk_bytes:
			return None
		chunk = bytes(self._partial_pcm[: self._chunk_bytes])
		del self._partial_pcm[: self._chunk_bytes]
		return chunk

	@property
	def partial_text(self) -> str:
		return " ".join(self._partial_text_parts).strip()

	async def finalize_once(self) -> tuple[str, str, bytes]:
		"""Returns (transcript, reply_text, reply_wav_bytes)."""
		pcm = b"".join(self._pcm_parts)

		# wrap pcm as wav (OpenAISTTService expects wav bytes)
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
	- {"type":"end"} (or {"type":"stop"})

	Server -> Client:
	- {"type":"status","message":"..."}
	- {"type":"transcript_partial","text":"..."}
	- {"type":"transcript","text":"..."} (final)
	- {"type":"reply","text":"..."}
	- {"type":"audio_start","format":"pcm16","sample_rate_hz":24000,"channels":1}
	- {"type":"audio_chunk","data":"<base64 pcm16 mono>"}
	- {"type":"audio_end"}
	- {"type":"latency","stage":"stt_final|llm|tts_first|tts_total","ms":123}
	"""
	await websocket.accept()

	cfg = TalkSessionConfig()
	session = build_default_session(cfg)

	async def send_json(payload: dict) -> None:
		await websocket.send_text(json.dumps(payload))

	await send_json({"type": "status", "message": "connected"})

	partial_queue: Optional[asyncio.Queue[Optional[bytes]]] = None
	partial_task: Optional[asyncio.Task] = None
	is_recording = False

	async def stop_partial_worker() -> None:
		nonlocal partial_queue, partial_task
		if not partial_queue or not partial_task:
			return
		await partial_queue.put(None)
		try:
			await partial_task
		finally:
			partial_queue = None
			partial_task = None

	async def partial_worker(queue: asyncio.Queue[Optional[bytes]]) -> None:
		while True:
			chunk = await queue.get()
			if chunk is None:
				break
			wav_in = pcm16_mono_to_wav_bytes(chunk, sample_rate_hz=session.cfg.input_sample_rate_hz)
			stt_start = perf_counter()
			stt = await session.stt.transcribe_wav(wav_in)
			stt_ms = int((perf_counter() - stt_start) * 1000)
			text = (stt.text or "").strip()
			if text:
				session._partial_text_parts.append(text)
				await send_json({"type": "transcript_partial", "text": session.partial_text})
				await send_json({"type": "latency", "stage": "stt_partial", "ms": stt_ms})

	try:
		while True:
			msg = await websocket.receive_text()
			payload = json.loads(msg)
			mtype = payload.get("type")

			if mtype == "start":
				session.reset()
				if payload.get("sample_rate_hz"):
					cfg.input_sample_rate_hz = int(payload["sample_rate_hz"])
				session._chunk_bytes = int(cfg.input_sample_rate_hz * 2 * cfg.stt_chunk_seconds)
				partial_queue = asyncio.Queue()
				partial_task = asyncio.create_task(partial_worker(partial_queue))
				is_recording = True
				await send_json({"type": "status", "message": "recording"})

			elif mtype == "audio_chunk":
				data_b64 = payload.get("data")
				if not data_b64:
					continue
				session.add_pcm16_chunk(base64.b64decode(data_b64))
				if partial_queue:
					while True:
						chunk = session.pop_partial_chunk()
						if not chunk:
							break
						await partial_queue.put(chunk)

			elif mtype in ("end", "stop"):
				if not is_recording:
					await send_json({"type": "status", "message": "idle"})
					continue
				is_recording = False
				await send_json({"type": "status", "message": "processing"})
				await stop_partial_worker()

				pcm = b"".join(session._pcm_parts)
				wav_in = pcm16_mono_to_wav_bytes(pcm, sample_rate_hz=cfg.input_sample_rate_hz)

				stt_start = perf_counter()
				stt = await session.stt.transcribe_wav(wav_in)
				stt_ms = int((perf_counter() - stt_start) * 1000)
				await send_json({"type": "latency", "stage": "stt_final", "ms": stt_ms})

				user_text = (stt.text or "").strip()
				if not user_text:
					user_text = "(I couldn't catch the audio. Ask the user to try again.)"
				await send_json({"type": "transcript", "text": user_text})

				llm_start = perf_counter()
				reply_text = await session.llm.respond(system_prompt=SYSTEM_PROMPT, user_text=user_text)
				llm_ms = int((perf_counter() - llm_start) * 1000)
				await send_json({"type": "latency", "stage": "llm", "ms": llm_ms})

				if not reply_text.strip():
					reply_text = "Sorry — I didn't get that. Can you try again?"
				await send_json({"type": "reply", "text": reply_text})

				tts_start = perf_counter()
				first_audio_sent = False
				if hasattr(session.tts, "stream_pcm"):
					await send_json(
						{
							"type": "audio_start",
							"format": "pcm16",
							"sample_rate_hz": getattr(session.tts, "output_sample_rate_hz", 24000),
							"channels": 1,
						}
					)
					async for chunk in session.tts.stream_pcm(reply_text):  # type: ignore[attr-defined]
						if not first_audio_sent:
							first_audio_sent = True
							await send_json(
								{
									"type": "latency",
									"stage": "tts_first",
									"ms": int((perf_counter() - tts_start) * 1000),
								}
							)
						await send_json(
							{
								"type": "audio_chunk",
								"data": base64.b64encode(chunk).decode("ascii"),
							}
						)
				else:
					tts = await session.tts.synthesize_wav(reply_text)
					await send_json(
						{
							"type": "audio",
							"mime": tts.mime,
							"data": base64.b64encode(tts.wav_bytes).decode("ascii"),
						}
					)
				await send_json(
					{
						"type": "latency",
						"stage": "tts_total",
						"ms": int((perf_counter() - tts_start) * 1000),
					}
				)
				await send_json({"type": "audio_end"})
				await send_json({"type": "status", "message": "ready"})

			else:
				await send_json({"type": "status", "message": f"unknown message type: {mtype}"})
	except Exception as e:
		# Don’t fully swallow server-side errors; they make debugging impossible.
		try:
			await send_json({"type": "status", "message": f"error: {e}"})
		except Exception:
			pass
		try:
			await stop_partial_worker()
		except Exception:
			pass
		raise
