# VoiceSynth

State-of-the-art Text-to-Speech system with voice cloning and emotion control.

## Features

- **Multiple TTS Models**: VITS, Tacotron2, and a lightweight CPU-friendly model
- **Voice Cloning**: Clone any voice from a 30-second audio sample
- **Emotion Control**: Synthesize speech with emotions (happy, sad, angry, calm, etc.)
- **Text Normalization**: Automatic expansion of abbreviations, numbers, and symbols
- **Audio Processing**: Sample rate conversion, noise reduction, silence trimming
- **FastAPI Server**: Production-ready REST API

## Quick Start

```python
from voicesynth import VoiceSynthesizer

synth = VoiceSynthesizer()
synth.load_model()
output = synth.synthesize("Hello, world!")
wav = synth.synthesize_to_wav("Hello, world!", emotion="happy")
```

## API Server

```bash
uvicorn voicesynth.api:app --host 0.0.0.0 --port 8000
```

## Installation

```bash
pip install -e ".[full]"
```

## Testing

```bash
pytest tests/
```

## License

MIT
