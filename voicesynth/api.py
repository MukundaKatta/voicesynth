"""FastAPI server for VoiceSynth TTS service."""

import base64
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import Response
from pydantic import BaseModel, Field

from voicesynth.synthesizer import VoiceSynthesizer, SynthesisConfig
from voicesynth.voice_cloning import VoiceCloner
from voicesynth.audio import AudioProcessor

app = FastAPI(title="VoiceSynth API", version="0.1.0")
synthesizer = None
cloner = None
audio_processor = None


class SynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    voice: Optional[str] = None
    emotion: str = "neutral"
    emotion_intensity: float = Field(1.0, ge=0.0, le=2.0)
    speed: float = Field(1.0, ge=0.25, le=4.0)
    model: str = "vits"
    format: str = "wav"


class CloneRequest(BaseModel):
    audio_base64: str
    name: str = Field(..., min_length=1, max_length=64)


@app.on_event("startup")
async def startup():
    global synthesizer, cloner, audio_processor
    config = SynthesisConfig()
    synthesizer = VoiceSynthesizer(config)
    synthesizer.load_model()
    cloner = VoiceCloner(sample_rate=config.sample_rate)
    audio_processor = AudioProcessor()


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": synthesizer is not None}


@app.post("/synthesize")
async def synthesize(request: SynthesizeRequest):
    if synthesizer is None:
        raise HTTPException(503, "Not initialized")
    output = synthesizer.synthesize(text=request.text, voice=request.voice, emotion=request.emotion,
                                     emotion_intensity=request.emotion_intensity, speed=request.speed)
    wav_bytes = audio_processor.to_wav_bytes(output.audio, output.sample_rate)
    return {"audio_base64": base64.b64encode(wav_bytes).decode(), "sample_rate": output.sample_rate,
            "duration_seconds": output.duration_seconds, "format": request.format, "metadata": output.metadata}


@app.post("/synthesize/stream")
async def synthesize_stream(request: SynthesizeRequest):
    if synthesizer is None:
        raise HTTPException(503, "Not initialized")
    output = synthesizer.synthesize(text=request.text, voice=request.voice, emotion=request.emotion,
                                     emotion_intensity=request.emotion_intensity, speed=request.speed)
    wav_bytes = audio_processor.to_wav_bytes(output.audio, output.sample_rate)
    return Response(content=wav_bytes, media_type="audio/wav",
                    headers={"X-Duration-Seconds": str(output.duration_seconds)})


@app.post("/clone")
async def clone_voice(request: CloneRequest):
    if cloner is None:
        raise HTTPException(503, "Not initialized")
    try:
        wav_bytes = base64.b64decode(request.audio_base64)
        audio, sr = audio_processor.from_wav_bytes(wav_bytes)
    except Exception as e:
        raise HTTPException(400, f"Invalid audio: {e}")
    if sr != cloner.sample_rate:
        audio = audio_processor.resample(audio, sr, cloner.sample_rate)
    if len(audio) / cloner.sample_rate < 5.0:
        raise HTTPException(400, "Audio too short, need at least 5 seconds")
    profile = cloner.clone_voice(audio, name=request.name)
    synthesizer.register_voice(request.name, profile.embedding)
    return {"name": request.name, "fundamental_freq": float(profile.fundamental_freq),
            "embedding_dim": len(profile.embedding)}


@app.get("/voices")
async def list_voices():
    return [{"name": n} for n in synthesizer.get_voices()]


@app.get("/models")
async def list_models():
    return {"models": [
        {"name": "vits", "description": "VITS end-to-end TTS", "sample_rate": 22050},
        {"name": "tacotron2", "description": "Tacotron2 + vocoder", "sample_rate": 22050},
        {"name": "lightweight", "description": "Fast concatenative TTS", "sample_rate": 16000},
    ]}


@app.get("/emotions")
async def list_emotions():
    return {"emotions": ["neutral", "happy", "sad", "angry", "calm", "excited", "fearful", "surprised"]}
