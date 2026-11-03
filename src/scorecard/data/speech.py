"""Speech round trip for noise calibration: TTS, telephony degradation, then real STT.

Azure voice names and Deepgram isiZulu coverage must be confirmed against live docs before a paid
run. Deepgram's `language=multi` handling of isiZulu is unverified.
"""

from __future__ import annotations

import io
import wave
from typing import Any, Protocol
from xml.sax.saxutils import escape

import httpx
import numpy as np

from scorecard.data.codeswitch import switch_stats

_VOICES = {"en": "en-ZA-LeahNeural", "zu": "zu-ZA-ThandoNeural", "af": "af-ZA-AdriNeural"}
_MU = 255.0


def dominant_language(text: str) -> str:
    stats = switch_stats(text)
    if stats.zulu_share >= 0.3:
        return "zu"
    if stats.afrikaans_share >= 0.3:
        return "af"
    return "en"


class Synthesizer(Protocol):
    def synthesize(self, text: str, language: str) -> bytes: ...


class Transcriber(Protocol):
    def transcribe(self, wav: bytes) -> str: ...


class AzureSynthesizer:
    def __init__(self, key: str, region: str, transport: httpx.BaseTransport | None = None) -> None:
        self._client = httpx.Client(
            base_url=f"https://{region}.tts.speech.microsoft.com",
            headers={
                "Ocp-Apim-Subscription-Key": key,
                "Content-Type": "application/ssml+xml",
                "X-Microsoft-OutputFormat": "riff-16khz-16bit-mono-pcm",
            },
            timeout=60,
            transport=transport,
        )

    def synthesize(self, text: str, language: str) -> bytes:
        voice = _VOICES[language]
        locale = voice.rsplit("-", 1)[0]
        ssml = (
            f"<speak version='1.0' xml:lang='{locale}'><voice name='{voice}'>"
            f"{escape(text)}</voice></speak>"
        )
        reply = self._client.post("/cognitiveservices/v1", content=ssml.encode("utf-8"))
        reply.raise_for_status()
        return reply.content


class DeepgramTranscriber:
    def __init__(
        self,
        key: str,
        model: str = "nova-3",
        language: str = "multi",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._params = {"model": model, "language": language, "punctuate": "false"}
        self._client = httpx.Client(
            base_url="https://api.deepgram.com",
            headers={"Authorization": f"Token {key}", "Content-Type": "audio/wav"},
            timeout=300,
            transport=transport,
        )

    def transcribe(self, wav: bytes) -> str:
        reply = self._client.post("/v1/listen", params=self._params, content=wav)
        reply.raise_for_status()
        channel = reply.json()["results"]["channels"][0]
        return str(channel["alternatives"][0]["transcript"])

    def transcribe_utterances(self, wav: bytes) -> list[dict[str, Any]]:
        """Diarized utterances, each with a speaker number and a transcript."""
        params = {**self._params, "diarize": "true", "utterances": "true", "punctuate": "true"}
        reply = self._client.post("/v1/listen", params=params, content=wav)
        reply.raise_for_status()
        return list(reply.json()["results"]["utterances"])


def _read_pcm16(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(wav_bytes)) as reader:
        if reader.getsampwidth() != 2 or reader.getnchannels() != 1:
            raise ValueError("expected mono 16-bit PCM")
        rate = reader.getframerate()
        frames = reader.readframes(reader.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float64) / 32768.0, rate


def degrade_for_telephony(wav_bytes: bytes, seed: int = 0, snr_db: float = 20.0) -> bytes:
    """Resample to 8kHz, add noise and pass through mu-law, as a phone line would."""
    samples, rate = _read_pcm16(wav_bytes)
    target = 8000
    if rate != target:
        duration = len(samples) / rate
        count = max(1, int(duration * target))
        # Averaging over each output sample acts as a crude anti-aliasing filter.
        window = max(1, rate // target)
        smoothed = np.convolve(samples, np.ones(window) / window, mode="same")
        positions = np.linspace(0, len(samples) - 1, count)
        samples = np.interp(positions, np.arange(len(samples)), smoothed)
    rng = np.random.default_rng(seed)
    signal_power = float(np.mean(samples**2)) or 1e-12
    noise_power = signal_power / (10 ** (snr_db / 10))
    noisy = np.clip(samples + rng.normal(0, np.sqrt(noise_power), len(samples)), -1, 1)
    encoded = np.sign(noisy) * np.log1p(_MU * np.abs(noisy)) / np.log1p(_MU)
    quantized = np.round((encoded + 1) / 2 * _MU) / _MU * 2 - 1
    decoded = np.sign(quantized) * (np.power(1 + _MU, np.abs(quantized)) - 1) / _MU
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(target)
        writer.writeframes((decoded * 32767).astype(np.int16).tobytes())
    return output.getvalue()


def round_trip_turn(
    text: str, synthesizer: Synthesizer, transcriber: Transcriber, seed: int = 0
) -> str:
    """One turn through TTS, the phone line and STT. Returns the STT hypothesis."""
    audio = synthesizer.synthesize(text, dominant_language(text))
    return transcriber.transcribe(degrade_for_telephony(audio, seed))
