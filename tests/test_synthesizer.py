"""Tests for VoiceSynth synthesizer and supporting modules."""

import numpy as np
import pytest

from voicesynth.synthesizer import (
    VoiceSynthesizer,
    SynthesisConfig,
    normalize_text,
    text_to_phonemes,
    number_to_words,
)
from voicesynth.audio import AudioProcessor, AudioConfig
from voicesynth.emotions import EmotionController, Emotion
from voicesynth.models import get_model, VITSWrapper, Tacotron2Wrapper, LightweightTTS
from voicesynth.voice_cloning import VoiceCloner


# ---------- Text normalization ----------

class TestNormalization:
    def test_abbreviations(self):
        assert "doctor" in normalize_text("Dr. Smith")
        assert "mister" in normalize_text("Mr. Jones")

    def test_numbers(self):
        result = normalize_text("I have 42 apples")
        assert "forty two" in result

    def test_symbols(self):
        result = normalize_text("100% done & ready")
        assert "percent" in result
        assert "and" in result

    def test_whitespace(self):
        result = normalize_text("  too   many    spaces  ")
        assert "  " not in result

    def test_number_to_words(self):
        assert number_to_words(0) == "zero"
        assert number_to_words(1) == "one"
        assert number_to_words(13) == "thirteen"
        assert number_to_words(42) == "forty two"
        assert number_to_words(100) == "one hundred"
        assert number_to_words(1000) == "one thousand"


class TestPhonemizer:
    def test_basic_word(self):
        phonemes = text_to_phonemes("hello")
        assert len(phonemes) > 0
        assert "SIL" in phonemes  # At least one silence

    def test_digraphs(self):
        phonemes = text_to_phonemes("the shop")
        assert "TH" in phonemes
        assert "SH" in phonemes

    def test_punctuation(self):
        phonemes = text_to_phonemes("hello, world!")
        sil_count = phonemes.count("SIL")
        assert sil_count >= 2


# ---------- Audio processor ----------

class TestAudioProcessor:
    def setup_method(self):
        self.proc = AudioProcessor(AudioConfig(sample_rate=16000))

    def test_normalize(self):
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        normalized = self.proc.normalize_audio(audio, target_db=-3.0)
        peak = np.max(np.abs(normalized))
        expected = 10 ** (-3.0 / 20.0)
        assert abs(peak - expected) < 0.01

    def test_resample(self):
        audio = np.sin(np.linspace(0, 10, 16000)).astype(np.float32)
        resampled = self.proc.resample(audio, 16000, 8000)
        assert len(resampled) == 8000

    def test_resample_same_rate(self):
        audio = np.ones(100, dtype=np.float32)
        result = self.proc.resample(audio, 16000, 16000)
        np.testing.assert_array_equal(audio, result)

    def test_wav_roundtrip(self):
        audio = np.sin(np.linspace(0, 10, 16000)).astype(np.float32) * 0.5
        wav = self.proc.to_wav_bytes(audio, 16000)
        loaded, sr = self.proc.from_wav_bytes(wav)
        assert sr == 16000
        assert abs(len(loaded) - len(audio)) <= 1

    def test_trim_silence(self):
        silence = np.zeros(4000, dtype=np.float32)
        signal = np.random.randn(8000).astype(np.float32) * 0.5
        audio = np.concatenate([silence, signal, silence])
        trimmed = self.proc.trim_silence(audio)
        assert len(trimmed) < len(audio)

    def test_noise_reduction(self):
        clean = np.sin(np.linspace(0, 20, 8000)).astype(np.float32)
        noisy = clean + np.random.randn(8000).astype(np.float32) * 0.001
        reduced = self.proc.reduce_noise(noisy)
        assert len(reduced) == len(noisy)


# ---------- Emotion controller ----------

class TestEmotionController:
    def setup_method(self):
        self.ctrl = EmotionController(sample_rate=16000)
        self.audio = np.sin(np.linspace(0, 100, 16000)).astype(np.float32) * 0.5

    def test_neutral_passthrough(self):
        result = self.ctrl.apply_emotion(self.audio, "neutral")
        # Neutral should be close to original
        assert len(result) > 0

    def test_happy_faster(self):
        result = self.ctrl.apply_emotion(self.audio, "happy")
        # Happy speech should be shorter (faster)
        assert len(result) < len(self.audio)

    def test_sad_slower(self):
        result = self.ctrl.apply_emotion(self.audio, "sad")
        assert len(result) > len(self.audio)

    def test_intensity_scaling(self):
        low = self.ctrl.apply_emotion(self.audio, "angry", intensity=0.1)
        high = self.ctrl.apply_emotion(self.audio, "angry", intensity=1.0)
        # High intensity should have more energy
        assert np.mean(np.abs(high)) > np.mean(np.abs(low)) * 0.8

    def test_blend_emotions(self):
        result = self.ctrl.blend_emotions(self.audio, {"happy": 0.5, "sad": 0.5})
        assert len(result) == len(self.audio)

    def test_tremolo(self):
        result = self.ctrl.apply_tremolo(self.audio, depth=0.5, rate_hz=5.0)
        assert len(result) == len(self.audio)


# ---------- Models ----------

class TestModels:
    def test_get_model_factory(self):
        model = get_model("vits")
        assert isinstance(model, VITSWrapper)

        model = get_model("tacotron2")
        assert isinstance(model, Tacotron2Wrapper)

        model = get_model("lightweight")
        assert isinstance(model, LightweightTTS)

    def test_invalid_model(self):
        with pytest.raises(ValueError):
            get_model("nonexistent")

    def test_vits_synthesize(self):
        model = get_model("vits")
        model.load()
        output = model.synthesize(["HH", "EH", "L", "OW"])
        assert len(output.audio) > 0
        assert output.sample_rate == 22050

    def test_tacotron2_synthesize(self):
        model = get_model("tacotron2")
        model.load()
        output = model.synthesize(["HH", "EH", "L", "OW"])
        assert output.mel_spectrogram is not None
        assert len(output.audio) > 0

    def test_lightweight_synthesize(self):
        model = get_model("lightweight")
        model.load()
        output = model.synthesize(["HH", "EH", "L", "OW"])
        assert output.sample_rate == 16000
        assert len(output.audio) > 0


# ---------- Voice cloning ----------

class TestVoiceCloner:
    def setup_method(self):
        self.cloner = VoiceCloner(sample_rate=16000)
        t = np.arange(16000 * 5) / 16000  # 5 seconds
        self.audio = (np.sin(2 * np.pi * 200 * t) * 0.5).astype(np.float32)

    def test_extract_embedding(self):
        emb = self.cloner.extract_embedding(self.audio)
        assert emb.shape == (256,)
        # Should be L2 normalized
        assert abs(np.linalg.norm(emb) - 1.0) < 0.01

    def test_clone_voice(self):
        profile = self.cloner.clone_voice(self.audio, name="test")
        assert profile.name == "test"
        assert profile.fundamental_freq > 0
        assert profile.embedding.shape == (256,)

    def test_profile_storage(self):
        self.cloner.clone_voice(self.audio, name="speaker_a")
        assert "speaker_a" in self.cloner.list_profiles()
        assert self.cloner.get_profile("speaker_a") is not None

    def test_interpolate(self):
        p1 = self.cloner.clone_voice(self.audio, name="a")
        audio2 = (np.sin(2 * np.pi * 300 * np.arange(80000) / 16000) * 0.5).astype(np.float32)
        p2 = self.cloner.clone_voice(audio2, name="b")
        blended = self.cloner.interpolate_voices(p1, p2, alpha=0.5)
        assert blended.embedding.shape == (256,)


# ---------- Full synthesizer ----------

class TestVoiceSynthesizer:
    def setup_method(self):
        config = SynthesisConfig(model_name="lightweight")
        self.synth = VoiceSynthesizer(config)
        self.synth.load_model()

    def test_synthesize_basic(self):
        output = self.synth.synthesize("Hello world")
        assert len(output.audio) > 0
        assert output.duration_seconds > 0

    def test_synthesize_with_emotion(self):
        output = self.synth.synthesize("I am happy", emotion="happy")
        assert output.metadata.get("emotion") == "happy"

    def test_synthesize_to_wav(self):
        wav = self.synth.synthesize_to_wav("Test")
        assert wav[:4] == b"RIFF"  # WAV header

    def test_register_and_use_voice(self):
        embedding = np.random.randn(256).astype(np.float32)
        self.synth.register_voice("test_voice", embedding)
        assert "test_voice" in self.synth.get_voices()
        output = self.synth.synthesize("Hello", voice="test_voice")
        assert output.metadata.get("voice") == "test_voice"
