from __future__ import annotations

import abc
from dataclasses import dataclass


@dataclass
class STTResult:
	text: str


class STTProvider(abc.ABC):
	@abc.abstractmethod
	async def transcribe_wav(self, wav_bytes: bytes) -> STTResult: ...


class LLMProvider(abc.ABC):
	@abc.abstractmethod
	async def respond(self, *, system_prompt: str, user_text: str) -> str: ...


@dataclass
class TTSResult:
	wav_bytes: bytes
	mime: str = "audio/wav"


class TTSProvider(abc.ABC):
	@abc.abstractmethod
	async def synthesize_wav(self, text: str) -> TTSResult: ...
