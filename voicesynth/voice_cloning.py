"""Voice cloning from short audio samples using speaker embeddings."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from voicesynth.audio import AudioProcessor


@dataclass
class SpeakerProfile:
    embedding: np.ndarray
    fundamental_freq: float
    spectral_centroid: float
    speaking_rate: float
    energy_profile: np.ndarray
    name: Optional[str] = None


class VoiceCloner:
    EMBEDDING_DIM = 256
    N_MEL_BANDS = 80
    FRAME_SIZE = 1024
    HOP_SIZE = 256

    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        self._audio_processor = AudioProcessor()
        self._profiles = {}

    def _compute_stft(self, audio):
        n_frames = (len(audio) - self.FRAME_SIZE) // self.HOP_SIZE + 1
        if n_frames <= 0:
            return np.zeros((self.FRAME_SIZE // 2 + 1, 1))
        window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(self.FRAME_SIZE) / self.FRAME_SIZE))
        spectrogram = np.zeros((self.FRAME_SIZE // 2 + 1, n_frames))
        for i in range(n_frames):
            start = i * self.HOP_SIZE
            frame = audio[start:start + self.FRAME_SIZE] * window
            spectrogram[:, i] = np.abs(np.fft.rfft(frame))
        return spectrogram

    def _mel_filterbank(self, n_fft_bins):
        fmin, fmax = 80.0, self.sample_rate / 2
        mel_min = 2595 * np.log10(1 + fmin / 700)
        mel_max = 2595 * np.log10(1 + fmax / 700)
        mel_points = np.linspace(mel_min, mel_max, self.N_MEL_BANDS + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        bin_points = np.floor((n_fft_bins * 2 - 1) * hz_points / self.sample_rate).astype(int)
        filterbank = np.zeros((self.N_MEL_BANDS, n_fft_bins))
        for i in range(self.N_MEL_BANDS):
            left, center, right = bin_points[i], bin_points[i + 1], bin_points[i + 2]
            for j in range(left, center):
                if center != left:
                    filterbank[i, j] = (j - left) / (center - left)
            for j in range(center, right):
                if right != center:
                    filterbank[i, j] = (right - j) / (right - center)
        return filterbank

    def _compute_mel_spectrogram(self, audio):
        stft = self._compute_stft(audio)
        filterbank = self._mel_filterbank(stft.shape[0])
        mel = filterbank @ stft
        return np.log(np.maximum(mel, 1e-8))

    def _estimate_f0(self, audio):
        mid = len(audio) // 2
        frame_len = min(4096, len(audio))
        start = max(0, mid - frame_len // 2)
        frame = audio[start:start + frame_len]
        if len(frame) < 256:
            return 150.0
        corr = np.correlate(frame, frame, mode="full")
        corr = corr[len(corr) // 2:]
        min_lag = int(self.sample_rate / 500)
        max_lag = min(int(self.sample_rate / 60), len(corr) - 1)
        if min_lag >= max_lag:
            return 150.0
        search = corr[min_lag:max_lag]
        if len(search) == 0:
            return 150.0
        peak_idx = np.argmax(search) + min_lag
        return self.sample_rate / peak_idx if peak_idx > 0 else 150.0

    def _compute_spectral_centroid(self, spectrogram):
        freqs = np.linspace(0, self.sample_rate / 2, spectrogram.shape[0])
        total_energy = np.maximum(np.sum(spectrogram, axis=0), 1e-8)
        centroids = np.sum(spectrogram * freqs[:, np.newaxis], axis=0) / total_energy
        return float(np.mean(centroids))

    def _estimate_speaking_rate(self, audio):
        frame_len = int(self.sample_rate * 0.025)
        hop = int(self.sample_rate * 0.010)
        n_frames = (len(audio) - frame_len) // hop
        if n_frames <= 0:
            return 4.0
        energies = np.array([np.sqrt(np.mean(audio[i * hop:i * hop + frame_len] ** 2)) for i in range(n_frames)])
        above = energies > np.mean(energies) * 0.5
        transitions = np.diff(above.astype(int))
        n_syllables = np.sum(transitions > 0)
        duration_sec = len(audio) / self.sample_rate
        return max(1.0, n_syllables / duration_sec) if duration_sec > 0 else 4.0

    def extract_embedding(self, audio):
        mel = self._compute_mel_spectrogram(audio)
        mel_mean = np.mean(mel, axis=1)
        mel_std = np.std(mel, axis=1)
        mel_std_safe = np.maximum(mel_std, 1e-8)
        centered = mel - mel_mean[:, np.newaxis]
        mel_skew = np.mean((centered / mel_std_safe[:, np.newaxis]) ** 3, axis=1)
        if mel.shape[1] > 1:
            delta = np.diff(mel, axis=1)
            delta_mean = np.mean(delta, axis=1)
            delta_std = np.std(delta, axis=1)
        else:
            delta_mean = np.zeros(self.N_MEL_BANDS)
            delta_std = np.zeros(self.N_MEL_BANDS)
        features = np.concatenate([mel_mean, mel_std, mel_skew, delta_mean, delta_std])
        np.random.seed(42)
        projection = np.random.randn(self.EMBEDDING_DIM, len(features)).astype(np.float32) * 0.01
        embedding = projection @ features
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.astype(np.float32)

    def clone_voice(self, audio, name=None, preprocess=True):
        if preprocess:
            audio = self._audio_processor.normalize_audio(audio.astype(np.float32))
            audio = self._audio_processor.trim_silence(audio)
            audio = self._audio_processor.reduce_noise(audio)
        embedding = self.extract_embedding(audio)
        stft = self._compute_stft(audio)
        mel = self._compute_mel_spectrogram(audio)
        profile = SpeakerProfile(
            embedding=embedding, fundamental_freq=self._estimate_f0(audio),
            spectral_centroid=self._compute_spectral_centroid(stft),
            speaking_rate=self._estimate_speaking_rate(audio),
            energy_profile=np.mean(mel, axis=1), name=name,
        )
        if name:
            self._profiles[name] = profile
        return profile

    def get_profile(self, name):
        return self._profiles.get(name)

    def list_profiles(self):
        return list(self._profiles.keys())

    def interpolate_voices(self, profile_a, profile_b, alpha=0.5):
        embedding = (1 - alpha) * profile_a.embedding + alpha * profile_b.embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return SpeakerProfile(
            embedding=embedding,
            fundamental_freq=(1 - alpha) * profile_a.fundamental_freq + alpha * profile_b.fundamental_freq,
            spectral_centroid=(1 - alpha) * profile_a.spectral_centroid + alpha * profile_b.spectral_centroid,
            speaking_rate=(1 - alpha) * profile_a.speaking_rate + alpha * profile_b.speaking_rate,
            energy_profile=(1 - alpha) * profile_a.energy_profile + alpha * profile_b.energy_profile,
            name=f"interpolated_{alpha:.2f}",
        )
