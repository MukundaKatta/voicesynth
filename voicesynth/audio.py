"""Audio processing utilities for VoiceSynth."""

import io
import wave
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class AudioConfig:
    sample_rate: int = 22050
    channels: int = 1
    bit_depth: int = 16
    normalize: bool = True


class AudioProcessor:
    def __init__(self, config=None):
        self.config = config or AudioConfig()

    def resample(self, audio, orig_sr, target_sr):
        if orig_sr == target_sr:
            return audio
        duration = len(audio) / orig_sr
        target_length = int(duration * target_sr)
        indices = np.linspace(0, len(audio) - 1, target_length)
        idx_floor = np.floor(indices).astype(int)
        idx_ceil = np.minimum(idx_floor + 1, len(audio) - 1)
        frac = indices - idx_floor
        resampled = audio[idx_floor] * (1 - frac) + audio[idx_ceil] * frac
        return resampled.astype(audio.dtype)

    def normalize_audio(self, audio, target_db=-3.0):
        if len(audio) == 0:
            return audio
        peak = np.max(np.abs(audio))
        if peak == 0:
            return audio
        target_amplitude = 10 ** (target_db / 20.0)
        return audio * (target_amplitude / peak)

    def reduce_noise(self, audio, noise_floor=0.01, smoothing_window=5):
        if len(audio) == 0:
            return audio
        amplitude = np.abs(audio)
        if smoothing_window > 1 and len(amplitude) >= smoothing_window:
            kernel = np.ones(smoothing_window) / smoothing_window
            smoothed = np.convolve(amplitude, kernel, mode="same")
        else:
            smoothed = amplitude
        peak = np.max(smoothed)
        if peak == 0:
            return audio
        threshold = peak * noise_floor
        mask = smoothed > threshold
        return audio * mask.astype(audio.dtype)

    def trim_silence(self, audio, threshold_db=-40.0, frame_length=1024):
        if len(audio) == 0:
            return audio
        threshold_amp = 10 ** (threshold_db / 20.0)
        n_frames = max(1, len(audio) // frame_length)
        frame_energies = []
        for i in range(n_frames):
            start = i * frame_length
            end = min(start + frame_length, len(audio))
            frame_energies.append(np.max(np.abs(audio[start:end])))
        start_frame = 0
        for i, energy in enumerate(frame_energies):
            if energy > threshold_amp:
                start_frame = i
                break
        end_frame = n_frames
        for i in range(len(frame_energies) - 1, -1, -1):
            if frame_energies[i] > threshold_amp:
                end_frame = i + 1
                break
        return audio[start_frame * frame_length: min(end_frame * frame_length, len(audio))]

    def to_wav_bytes(self, audio, sample_rate=None):
        sr = sample_rate or self.config.sample_rate
        if audio.dtype in (np.float32, np.float64):
            audio_int = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        else:
            audio_int = audio.astype(np.int16)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(self.config.channels)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(audio_int.tobytes())
        return buffer.getvalue()

    def from_wav_bytes(self, wav_bytes):
        buffer = io.BytesIO(wav_bytes)
        with wave.open(buffer, "rb") as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
            sampwidth = wf.getsampwidth()
        if sampwidth == 2:
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
        elif sampwidth == 1:
            audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128.0
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth}")
        return audio, sr

    def convert_format(self, audio, source_sr, target_sr=None, normalize=True, reduce_noise=False, trim=False):
        result = audio.copy()
        if trim:
            result = self.trim_silence(result)
        if reduce_noise:
            result = self.reduce_noise(result)
        target = target_sr or self.config.sample_rate
        if source_sr != target:
            result = self.resample(result, source_sr, target)
        if normalize:
            result = self.normalize_audio(result)
        return result
