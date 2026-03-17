"""Emotion control for speech synthesis."""

from dataclasses import dataclass
from enum import Enum

import numpy as np


class Emotion(str, Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    CALM = "calm"
    EXCITED = "excited"
    FEARFUL = "fearful"
    SURPRISED = "surprised"


@dataclass
class EmotionProfile:
    pitch_shift: float = 0.0
    speed_factor: float = 1.0
    energy_factor: float = 1.0
    pitch_variance: float = 1.0
    tremolo_depth: float = 0.0
    breathiness: float = 0.0


EMOTION_PROFILES = {
    Emotion.NEUTRAL: EmotionProfile(),
    Emotion.HAPPY: EmotionProfile(pitch_shift=2.5, speed_factor=1.15, energy_factor=1.2, pitch_variance=1.4, tremolo_depth=0.05, breathiness=0.1),
    Emotion.SAD: EmotionProfile(pitch_shift=-2.0, speed_factor=0.82, energy_factor=0.7, pitch_variance=0.6, tremolo_depth=0.0, breathiness=0.25),
    Emotion.ANGRY: EmotionProfile(pitch_shift=1.5, speed_factor=1.1, energy_factor=1.5, pitch_variance=1.6),
    Emotion.CALM: EmotionProfile(pitch_shift=-0.5, speed_factor=0.9, energy_factor=0.8, pitch_variance=0.5, breathiness=0.15),
    Emotion.EXCITED: EmotionProfile(pitch_shift=4.0, speed_factor=1.25, energy_factor=1.3, pitch_variance=1.8, tremolo_depth=0.1),
    Emotion.FEARFUL: EmotionProfile(pitch_shift=3.0, speed_factor=1.2, energy_factor=0.9, pitch_variance=1.5, tremolo_depth=0.15, breathiness=0.3),
    Emotion.SURPRISED: EmotionProfile(pitch_shift=5.0, speed_factor=1.05, energy_factor=1.1, pitch_variance=2.0, breathiness=0.1),
}


class EmotionController:
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate

    def get_profile(self, emotion, intensity=1.0):
        if isinstance(emotion, str):
            emotion = Emotion(emotion.lower())
        base = EMOTION_PROFILES.get(emotion, EMOTION_PROFILES[Emotion.NEUTRAL])
        return EmotionProfile(
            pitch_shift=base.pitch_shift * intensity,
            speed_factor=1.0 + (base.speed_factor - 1.0) * intensity,
            energy_factor=1.0 + (base.energy_factor - 1.0) * intensity,
            pitch_variance=1.0 + (base.pitch_variance - 1.0) * intensity,
            tremolo_depth=base.tremolo_depth * intensity,
            breathiness=base.breathiness * intensity,
        )

    def shift_pitch(self, audio, semitones):
        if abs(semitones) < 0.01:
            return audio
        factor = 2 ** (semitones / 12.0)
        indices = np.arange(0, len(audio), factor)
        indices = indices[indices < len(audio) - 1].astype(int)
        pitched = audio[indices]
        x_old = np.linspace(0, 1, len(pitched))
        x_new = np.linspace(0, 1, len(audio))
        return np.interp(x_new, x_old, pitched).astype(audio.dtype)

    def change_speed(self, audio, factor):
        if abs(factor - 1.0) < 0.01:
            return audio
        new_length = int(len(audio) / factor)
        x_old = np.linspace(0, 1, len(audio))
        x_new = np.linspace(0, 1, new_length)
        return np.interp(x_new, x_old, audio).astype(audio.dtype)

    def apply_tremolo(self, audio, depth, rate_hz=5.0):
        if depth < 0.001:
            return audio
        t = np.arange(len(audio)) / self.sample_rate
        modulation = 1.0 - depth * (0.5 * (1 + np.sin(2 * np.pi * rate_hz * t)))
        return (audio * modulation).astype(audio.dtype)

    def add_breathiness(self, audio, amount):
        if amount < 0.001:
            return audio
        noise = np.random.randn(len(audio)).astype(np.float32) * 0.02 * amount
        envelope = np.abs(audio)
        window = min(1024, len(envelope))
        if window > 1:
            kernel = np.ones(window) / window
            envelope = np.convolve(envelope, kernel, mode="same")
        return audio + noise * envelope

    def modify_pitch_variance(self, audio, factor):
        if abs(factor - 1.0) < 0.05:
            return audio
        frame_size = 2048
        n_frames = len(audio) // frame_size
        if n_frames < 2:
            return audio
        result = audio.copy()
        energies = []
        for i in range(n_frames):
            frame = audio[i * frame_size:(i + 1) * frame_size]
            energies.append(np.sqrt(np.mean(frame ** 2)))
        mean_energy = np.mean(energies) if energies else 1.0
        if mean_energy == 0:
            return audio
        for i in range(n_frames):
            start = i * frame_size
            end = start + frame_size
            deviation = (energies[i] - mean_energy) / mean_energy
            pitch_mod = np.clip(deviation * (factor - 1.0) * 2.0, -4, 4)
            if abs(pitch_mod) > 0.05:
                result[start:end] = self.shift_pitch(audio[start:end], pitch_mod)
        return result

    def apply_emotion(self, audio, emotion, intensity=1.0):
        profile = self.get_profile(emotion, intensity)
        result = audio.copy().astype(np.float32)
        result = self.shift_pitch(result, profile.pitch_shift)
        result = self.change_speed(result, profile.speed_factor)
        result = self.modify_pitch_variance(result, profile.pitch_variance)
        result = self.apply_tremolo(result, profile.tremolo_depth)
        result = self.add_breathiness(result, profile.breathiness)
        result = result * profile.energy_factor
        return np.clip(result, -1.0, 1.0)

    def blend_emotions(self, audio, emotions):
        total_weight = sum(emotions.values())
        if total_weight == 0:
            return audio
        result = np.zeros_like(audio, dtype=np.float32)
        for emotion_name, weight in emotions.items():
            modified = self.apply_emotion(audio, emotion_name, intensity=1.0)
            if len(modified) < len(result):
                padded = np.zeros_like(result)
                padded[:len(modified)] = modified
                modified = padded
            elif len(modified) > len(result):
                modified = modified[:len(result)]
            result += modified * (weight / total_weight)
        return np.clip(result, -1.0, 1.0)
