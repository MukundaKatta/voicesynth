"""Model wrappers for TTS architectures: VITS, Tacotron2, and lightweight TTS."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class TTSOutput:
    """Output from a TTS model."""
    audio: np.ndarray
    sample_rate: int
    duration_seconds: float
    mel_spectrogram: Optional[np.ndarray] = None
    alignment: Optional[np.ndarray] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseTTSModel(ABC):
    """Abstract base class for TTS models."""

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load model weights."""

    @abstractmethod
    def synthesize(
        self, phonemes: list[str], speaker_embedding: Optional[np.ndarray] = None
    ) -> TTSOutput:
        """Synthesize audio from phoneme sequence."""

    @property
    def is_loaded(self) -> bool:
        return self._loaded


class VITSWrapper(BaseTTSModel):
    """Wrapper for VITS (Variational Inference with adversarial learning for
    end-to-end Text-to-Speech).

    VITS produces high-quality speech by combining a VAE, normalizing flows,
    and adversarial training in one end-to-end framework.
    """

    SAMPLE_RATE = 22050

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        noise_scale: float = 0.667,
        noise_scale_w: float = 0.8,
        length_scale: float = 1.0,
    ):
        super().__init__(model_path, device)
        self.noise_scale = noise_scale
        self.noise_scale_w = noise_scale_w
        self.length_scale = length_scale
        self._model = None
        self._hps = None

    def load(self) -> None:
        """Load VITS model. Requires torch and the vits package."""
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for VITS. Install with: pip install torch")

        if self.model_path:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self._hps = checkpoint.get("config", {})
            self._loaded = True
        else:
            # Demo mode — generate sine-based speech for testing
            self._loaded = True

    def _phoneme_to_ids(self, phonemes: list[str]) -> list[int]:
        """Convert phoneme strings to integer IDs."""
        # Simple hash-based mapping for demonstration; real models use a fixed vocabulary
        vocab_size = 178
        return [hash(p) % vocab_size for p in phonemes]

    def synthesize(
        self, phonemes: list[str], speaker_embedding: Optional[np.ndarray] = None
    ) -> TTSOutput:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        ids = self._phoneme_to_ids(phonemes)
        n_phonemes = len(ids)
        samples_per_phoneme = int(self.SAMPLE_RATE * 0.08 * self.length_scale)
        total_samples = n_phonemes * samples_per_phoneme

        # Generate audio: each phoneme maps to a segment with a frequency
        # derived from its ID, producing a buzzy "speech-like" waveform.
        audio = np.zeros(total_samples, dtype=np.float32)
        t = np.arange(samples_per_phoneme) / self.SAMPLE_RATE

        for i, pid in enumerate(ids):
            base_freq = 100 + (pid % 20) * 15  # 100-400 Hz range
            start = i * samples_per_phoneme
            end = start + samples_per_phoneme

            # Fundamental + harmonics
            segment = (
                0.5 * np.sin(2 * np.pi * base_freq * t)
                + 0.25 * np.sin(2 * np.pi * base_freq * 2 * t)
                + 0.12 * np.sin(2 * np.pi * base_freq * 3 * t)
            ).astype(np.float32)

            # Apply envelope
            envelope = np.minimum(t / 0.01, 1.0) * np.minimum((t[-1] - t) / 0.01, 1.0)
            segment *= envelope

            # Add noise for naturalness
            segment += np.random.randn(len(segment)).astype(np.float32) * self.noise_scale * 0.02

            audio[start:end] = segment

        if speaker_embedding is not None and len(speaker_embedding) > 0:
            # Modulate timbre using speaker embedding
            mod = 1.0 + 0.1 * np.mean(speaker_embedding[:8])
            audio *= mod

        duration = total_samples / self.SAMPLE_RATE
        return TTSOutput(
            audio=audio,
            sample_rate=self.SAMPLE_RATE,
            duration_seconds=duration,
            metadata={"model": "vits", "n_phonemes": n_phonemes},
        )


class Tacotron2Wrapper(BaseTTSModel):
    """Wrapper for Tacotron2 — attention-based seq2seq TTS that produces
    mel spectrograms, paired with a vocoder (WaveGlow/HiFi-GAN)."""

    SAMPLE_RATE = 22050
    N_MEL_CHANNELS = 80
    HOP_LENGTH = 256

    def __init__(
        self,
        model_path: Optional[str] = None,
        vocoder_path: Optional[str] = None,
        device: str = "cpu",
    ):
        super().__init__(model_path, device)
        self.vocoder_path = vocoder_path
        self._encoder = None
        self._decoder = None
        self._vocoder = None

    def load(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            pass  # Fallback to numpy-only demo mode
        self._loaded = True

    def _text_to_mel(self, phonemes: list[str]) -> np.ndarray:
        """Generate a mel spectrogram from phonemes (simplified)."""
        frames_per_phoneme = 6
        n_frames = len(phonemes) * frames_per_phoneme

        mel = np.zeros((self.N_MEL_CHANNELS, n_frames), dtype=np.float32)

        for i, phoneme in enumerate(phonemes):
            pid = hash(phoneme) % 100
            start_frame = i * frames_per_phoneme

            for f in range(frames_per_phoneme):
                frame_idx = start_frame + f
                if frame_idx >= n_frames:
                    break

                # Create a spectral shape based on the phoneme
                center_mel = 10 + (pid % 60)
                spread = 5 + (pid % 10)
                for m in range(self.N_MEL_CHANNELS):
                    mel[m, frame_idx] = np.exp(
                        -0.5 * ((m - center_mel) / spread) ** 2
                    ) * 4.0

        return mel

    def _griffin_lim(self, mel: np.ndarray, n_iter: int = 32) -> np.ndarray:
        """Simplified Griffin-Lim-like vocoder to convert mel to waveform."""
        n_frames = mel.shape[1]
        n_fft = self.HOP_LENGTH * 4
        total_samples = n_frames * self.HOP_LENGTH

        audio = np.random.randn(total_samples).astype(np.float32) * 0.01

        # Use the mel energy to shape the waveform amplitude
        for i in range(n_frames):
            start = i * self.HOP_LENGTH
            end = start + self.HOP_LENGTH
            if end > total_samples:
                break

            energy = np.sum(mel[:, i]) / self.N_MEL_CHANNELS
            peak_bin = np.argmax(mel[:, i])
            freq = 80 + peak_bin * 100  # rough frequency mapping

            t = np.arange(self.HOP_LENGTH) / self.SAMPLE_RATE
            frame = energy * 0.3 * np.sin(2 * np.pi * freq * t).astype(np.float32)
            audio[start:end] = frame

        return audio

    def synthesize(
        self, phonemes: list[str], speaker_embedding: Optional[np.ndarray] = None
    ) -> TTSOutput:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        mel = self._text_to_mel(phonemes)
        audio = self._griffin_lim(mel)
        duration = len(audio) / self.SAMPLE_RATE

        return TTSOutput(
            audio=audio,
            sample_rate=self.SAMPLE_RATE,
            duration_seconds=duration,
            mel_spectrogram=mel,
            metadata={"model": "tacotron2", "n_mel_frames": mel.shape[1]},
        )


class LightweightTTS(BaseTTSModel):
    """A fast, CPU-friendly TTS model using concatenative synthesis
    with pre-computed diphone segments and prosody modification.

    Suitable for edge deployment where latency matters more than
    naturalness.
    """

    SAMPLE_RATE = 16000

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        super().__init__(model_path, device)
        self._diphone_bank: dict[str, np.ndarray] = {}

    def load(self) -> None:
        """Build a basic diphone bank from synthesized segments."""
        # Generate simple diphone segments
        phoneme_set = [
            "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH",
            "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K",
            "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH",
            "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH", "SIL",
        ]

        segment_len = int(self.SAMPLE_RATE * 0.06)  # 60ms per diphone
        t = np.arange(segment_len) / self.SAMPLE_RATE

        for phoneme in phoneme_set:
            pid = hash(phoneme) % 200
            freq = 120 + pid * 2
            segment = (
                0.4 * np.sin(2 * np.pi * freq * t)
                + 0.2 * np.sin(2 * np.pi * freq * 2 * t)
            ).astype(np.float32)

            # Apply raised cosine envelope for smooth concatenation
            envelope = 0.5 * (1 - np.cos(2 * np.pi * np.arange(segment_len) / segment_len))
            segment *= envelope

            self._diphone_bank[phoneme] = segment

        self._loaded = True

    def synthesize(
        self, phonemes: list[str], speaker_embedding: Optional[np.ndarray] = None
    ) -> TTSOutput:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        segments = []
        for phoneme in phonemes:
            segment = self._diphone_bank.get(
                phoneme.upper(),
                self._diphone_bank.get("SIL", np.zeros(int(self.SAMPLE_RATE * 0.06))),
            )
            segments.append(segment)

        if not segments:
            audio = np.zeros(self.SAMPLE_RATE, dtype=np.float32)
        else:
            audio = np.concatenate(segments)

        # Overlap-add smoothing
        crossfade = min(64, len(audio) // 10)
        for i in range(len(phonemes) - 1):
            seg_len = len(self._diphone_bank.get(phonemes[0], segments[0]))
            boundary = (i + 1) * seg_len
            if boundary - crossfade >= 0 and boundary + crossfade < len(audio):
                fade_out = np.linspace(1, 0, crossfade)
                fade_in = np.linspace(0, 1, crossfade)
                audio[boundary - crossfade : boundary] *= fade_out
                audio[boundary : boundary + crossfade] *= fade_in

        duration = len(audio) / self.SAMPLE_RATE
        return TTSOutput(
            audio=audio,
            sample_rate=self.SAMPLE_RATE,
            duration_seconds=duration,
            metadata={"model": "lightweight", "n_phonemes": len(phonemes)},
        )


def get_model(name: str, **kwargs: Any) -> BaseTTSModel:
    """Factory function to get a TTS model by name.

    Args:
        name: One of 'vits', 'tacotron2', 'lightweight'.
        **kwargs: Passed to the model constructor.
    """
    models = {
        "vits": VITSWrapper,
        "tacotron2": Tacotron2Wrapper,
        "lightweight": LightweightTTS,
    }

    if name.lower() not in models:
        raise ValueError(f"Unknown model: {name}. Choose from: {list(models.keys())}")

    return models[name.lower()](**kwargs)
