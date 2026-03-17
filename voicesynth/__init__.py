"""VoiceSynth - State-of-the-art Text-to-Speech System."""
__version__ = "0.1.0"

from voicesynth.synthesizer import VoiceSynthesizer
from voicesynth.audio import AudioProcessor
from voicesynth.emotions import EmotionController

__all__ = ["VoiceSynthesizer", "AudioProcessor", "EmotionController"]
