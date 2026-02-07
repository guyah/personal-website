from __future__ import annotations

import os
from typing import Optional

from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.tts import OpenAITTSService

from .providers import LLMProvider, STTProvider, STTResult, TTSProvider, TTSResult
from .wav import pcm16_mono_to_wav_bytes


class OpenAIPipecatSTTProvider(STTProvider):
	def __init__(
		self,
		*,
		api_key: Optional[str] = None,
		model: str = "gpt-4o-mini-transcribe",
		sample_rate_hz: int = 16000,
	):
		self.sample_rate_hz = sample_rate_hz
		self._svc = OpenAISTTService(api_key=api_key or os.environ.get("OPENAI_API_KEY"), model=model)

	async def transcribe_wav(self, wav_bytes: bytes) -> STTResult:
		# Pipecat's OpenAISTTService exposes the actual transcription call as a private method.
		# We keep this behind our provider interface so it can be swapped later.
		result = await self._svc._transcribe(wav_bytes)  # noqa: SLF001
		text = getattr(result, "text", "") or ""
		return STTResult(text=text.strip())


class OpenAIPipecatLLMProvider(LLMProvider):
	def __init__(self, *, api_key: Optional[str] = None, model: str = "gpt-4.1-mini"):
		self._svc = OpenAILLMService(model=model, params=None)
		# OpenAILLMService uses OPENAI_API_KEY env var via openai client under the hood.
		# api_key is kept for parity / future extension.
		if api_key:
			os.environ["OPENAI_API_KEY"] = api_key

	async def respond(self, *, system_prompt: str, user_text: str) -> str:
		ctx = OpenAILLMContext(
			messages=[
				{"role": "system", "content": system_prompt},
				{"role": "user", "content": user_text},
			]
		)
		out = await self._svc.run_inference(ctx)
		return (out or "").strip()


class OpenAIPipecatTTSProvider(TTSProvider):
	def __init__(
		self,
		*,
		api_key: Optional[str] = None,
		model: str = "gpt-4o-mini-tts",
		voice: str = "alloy",
		output_sample_rate_hz: int = 24000,
	):
		self.output_sample_rate_hz = output_sample_rate_hz
		self._svc = OpenAITTSService(
			api_key=api_key or os.environ.get("OPENAI_API_KEY"),
			model=model,
			voice=voice,
			sample_rate=output_sample_rate_hz,
		)
		# When using OpenAITTSService out-of-pipeline, Pipecat doesn't always initialize
		# the runtime sample_rate. Set it explicitly so chunk_size > 0.
		self._svc._sample_rate = output_sample_rate_hz  # noqa: SLF001
		self._setup_done = False

	async def _ensure_setup(self) -> None:
		if self._setup_done:
			return
		from pipecat.processors.frame_processor import FrameProcessorSetup
		from pipecat.clocks.system_clock import SystemClock
		import asyncio
		from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams

		tm = TaskManager()
		tm.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
		await self._svc.setup(FrameProcessorSetup(clock=SystemClock(), task_manager=tm))
		self._setup_done = True

	async def synthesize_wav(self, text: str) -> TTSResult:
		# OpenAITTSService yields raw PCM frames. We collect and wrap as WAV.
		await self._ensure_setup()

		pcm_parts: list[bytes] = []
		async for frame in self._svc.run_tts(text):
			data = self._frame_to_pcm(frame)
			if data:
				pcm_parts.append(data)
		pcm = b"".join(pcm_parts)
		wav = pcm16_mono_to_wav_bytes(pcm, sample_rate_hz=self.output_sample_rate_hz)
		return TTSResult(wav_bytes=wav, mime="audio/wav")

	async def stream_pcm(self, text: str):
		"""Yield PCM16 mono chunks for low-latency playback."""
		await self._ensure_setup()
		async for frame in self._svc.run_tts(text):
			data = self._frame_to_pcm(frame)
			if data:
				yield data

	@staticmethod
	def _frame_to_pcm(frame) -> Optional[bytes]:
		# Depending on Pipecat frame type, the PCM payload can be exposed under
		# different attribute names.
		return (
			getattr(frame, "audio", None)
			or getattr(frame, "data", None)
			or getattr(frame, "chunk", None)
		)
