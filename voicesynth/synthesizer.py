"""Core TTS synthesizer — text normalization, phoneme conversion, audio generation."""

import re
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from voicesynth.audio import AudioProcessor, AudioConfig
from voicesynth.emotions import EmotionController, Emotion
from voicesynth.models import BaseTTSModel, TTSOutput, get_model


ABBREVIATIONS = {
    "mr.": "mister", "mrs.": "missus", "dr.": "doctor", "st.": "saint",
    "jr.": "junior", "sr.": "senior", "prof.": "professor", "vs.": "versus",
    "etc.": "etcetera", "approx.": "approximately", "dept.": "department",
    "govt.": "government", "inc.": "incorporated", "ltd.": "limited",
}

ONES = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def number_to_words(n: int) -> str:
    if n == 0:
        return "zero"
    if n < 0:
        return "negative " + number_to_words(-n)
    parts = []
    if n >= 1000:
        parts.append(number_to_words(n // 1000) + " thousand")
        n %= 1000
    if n >= 100:
        parts.append(ONES[n // 100] + " hundred")
        n %= 100
    if n >= 20:
        word = TENS[n // 10]
        if n % 10:
            word += " " + ONES[n % 10]
        parts.append(word)
    elif n > 0:
        parts.append(ONES[n])
    return " ".join(parts)


def normalize_text(text: str) -> str:
    text = text.strip()
    for abbr, expansion in ABBREVIATIONS.items():
        text = re.sub(re.escape(abbr), expansion, text, flags=re.IGNORECASE)

    def replace_number(match):
        num = int(match.group())
        if num > 999999:
            return match.group()
        return number_to_words(num)

    text = re.sub(r"\b\d+\b", replace_number, text)
    symbol_map = {"&": " and ", "%": " percent ", "$": " dollars ", "#": " number ",
                  "@": " at ", "+": " plus ", "=": " equals "}
    for sym, word in symbol_map.items():
        text = text.replace(sym, word)
    text = re.sub(r"\s+", " ", text).strip()
    return text


G2P_RULES = {
    "a": "AE", "b": "B", "c": "K", "d": "D", "e": "EH", "f": "F",
    "g": "G", "h": "HH", "i": "IH", "j": "JH", "k": "K", "l": "L",
    "m": "M", "n": "N", "o": "AA", "p": "P", "q": "K", "r": "R",
    "s": "S", "t": "T", "u": "AH", "v": "V", "w": "W", "x": "K S",
    "y": "Y", "z": "Z",
}

DIGRAPH_RULES = {
    "th": "TH", "sh": "SH", "ch": "CH", "ph": "F", "wh": "W",
    "ng": "NG", "ck": "K", "ee": "IY", "oo": "UW", "ea": "IY",
    "ou": "AW", "ai": "EY", "ey": "EY", "ow": "OW", "oi": "OY",
    "au": "AO", "ar": "AA R", "er": "ER", "ir": "ER", "or": "AO R",
    "ur": "ER",
}


def text_to_phonemes(text: str) -> list:
    text = text.lower()
    words = re.findall(r"[a-z]+|[.,!?;:]", text)
    phonemes = []
    for word in words:
        if word in ".,!?;:":
            phonemes.append("SIL")
            continue
        i = 0
        while i < len(word):
            if i + 1 < len(word):
                digraph = word[i:i + 2]
                if digraph in DIGRAPH_RULES:
                    phonemes.extend(DIGRAPH_RULES[digraph].split())
                    i += 2
                    continue
            ch = word[i]
            if ch in G2P_RULES:
                phonemes.extend(G2P_RULES[ch].split())
            i += 1
        phonemes.append("SIL")
    return phonemes


@dataclass
class SynthesisConfig:
    model_name: str = "vits"
    sample_rate: int = 22050
    emotion: str = "neutral"
    emotion_intensity: float = 1.0
    speed: float = 1.0
    normalize_audio: bool = True
    trim_silence: bool = True


class VoiceSynthesizer:
    """End-to-end text-to-speech synthesizer.

    Pipeline: text normalization -> phoneme conversion -> model inference
    -> emotion modification -> audio post-processing.
    """

    def __init__(self, config=None):
        self.config = config or SynthesisConfig()
        self._model = None
        self._emotion_controller = EmotionController(self.config.sample_rate)
        self._audio_processor = AudioProcessor(AudioConfig(sample_rate=self.config.sample_rate))
        self._voices = {}

    def load_model(self, model_name=None):
        name = model_name or self.config.model_name
        self._model = get_model(name)
        self._model.load()

    def register_voice(self, name, speaker_embedding):
        self._voices[name] = speaker_embedding

    def get_voices(self):
        return list(self._voices.keys())

    def synthesize(self, text, voice=None, emotion=None, emotion_intensity=None, speed=None):
        if self._model is None:
            self.load_model()
        normalized = normalize_text(text)
        phonemes = text_to_phonemes(normalized)
        speaker_emb = self._voices.get(voice) if voice else None
        output = self._model.synthesize(phonemes, speaker_embedding=speaker_emb)
        audio = output.audio

        effective_speed = speed or self.config.speed
        if abs(effective_speed - 1.0) > 0.01:
            audio = self._emotion_controller.change_speed(audio, effective_speed)

        emo = emotion or self.config.emotion
        intensity = emotion_intensity if emotion_intensity is not None else self.config.emotion_intensity
        if emo != "neutral":
            audio = self._emotion_controller.apply_emotion(audio, emo, intensity)

        if self.config.trim_silence:
            audio = self._audio_processor.trim_silence(audio)
        if self.config.normalize_audio:
            audio = self._audio_processor.normalize_audio(audio)

        return TTSOutput(
            audio=audio,
            sample_rate=output.sample_rate,
            duration_seconds=len(audio) / output.sample_rate,
            mel_spectrogram=output.mel_spectrogram,
            alignment=output.alignment,
            metadata={**output.metadata, "text": text, "normalized": normalized,
                      "n_phonemes": len(phonemes), "emotion": emo, "voice": voice},
        )

    def synthesize_to_wav(self, text, **kwargs):
        output = self.synthesize(text, **kwargs)
        return self._audio_processor.to_wav_bytes(output.audio, output.sample_rate)
