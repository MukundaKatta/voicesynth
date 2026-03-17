"""Basic text-to-speech example using VoiceSynth."""

from voicesynth.synthesizer import VoiceSynthesizer, SynthesisConfig


def main():
    # Configure the synthesizer
    config = SynthesisConfig(
        model_name="vits",
        sample_rate=22050,
        emotion="neutral",
        speed=1.0,
    )

    synth = VoiceSynthesizer(config)
    synth.load_model()

    # Basic synthesis
    text = "Hello! Welcome to VoiceSynth. This is a demonstration of text to speech synthesis."
    output = synth.synthesize(text)

    print(f"Synthesized {output.duration_seconds:.2f}s of audio at {output.sample_rate}Hz")
    print(f"Phonemes: {output.metadata.get('n_phonemes', 'N/A')}")

    # Save to WAV
    wav_bytes = synth.synthesize_to_wav(text)
    with open("output_basic.wav", "wb") as f:
        f.write(wav_bytes)
    print("Saved to output_basic.wav")

    # Synthesize with different emotions
    emotions = ["happy", "sad", "angry", "calm", "excited"]
    for emotion in emotions:
        output = synth.synthesize(text, emotion=emotion, emotion_intensity=0.8)
        wav_bytes = synth.synthesize_to_wav(text, emotion=emotion, emotion_intensity=0.8)
        with open(f"output_{emotion}.wav", "wb") as f:
            f.write(wav_bytes)
        print(f"  [{emotion}] duration={output.duration_seconds:.2f}s")

    # Different speeds
    for speed in [0.75, 1.0, 1.5]:
        output = synth.synthesize("Speed test.", speed=speed)
        print(f"  speed={speed}x -> {output.duration_seconds:.2f}s")

    # Try different models
    for model in ["vits", "tacotron2", "lightweight"]:
        synth.load_model(model)
        output = synth.synthesize("Testing model output.")
        print(f"  [{model}] sr={output.sample_rate}, duration={output.duration_seconds:.2f}s")


if __name__ == "__main__":
    main()
