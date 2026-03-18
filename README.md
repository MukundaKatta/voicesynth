# voicesynth

**State-of-the-art open source text-to-speech with emotion control and voice cloning**

![Build](https://img.shields.io/badge/build-passing-brightgreen) ![License](https://img.shields.io/badge/license-proprietary-red)

## Install
```bash
pip install -e ".[dev]"
```

## Quick Start
```python
from src.core import Voicesynth
 instance = Voicesynth()
r = instance.synthesize(input="test")
```

## CLI
```bash
python -m src status
python -m src run --input "data"
```

## API
| Method | Description |
|--------|-------------|
| `synthesize()` | Synthesize |
| `clone_voice()` | Clone voice |
| `set_emotion()` | Set emotion |
| `adjust_speed()` | Adjust speed |
| `list_voices()` | List voices |
| `export_audio()` | Export audio |
| `get_stats()` | Get stats |
| `reset()` | Reset |

## Test
```bash
pytest tests/ -v
```

## License
(c) 2026 Officethree Technologies. All Rights Reserved.
