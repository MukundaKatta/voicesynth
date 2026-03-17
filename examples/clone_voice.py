"""Voice cloning example — clone a voice from an audio sample and synthesize."""

import numpy as np

from voicesynth.synthesizer import VoiceSynthesizer, SynthesisConfig
from voicesynth.voice_cloning import VoiceCloner
from voicesynth.audio import AudioProcessor


def main():
    sample_rate = 22050
    processor = AudioProcessor()
    cloner = VoiceCloner(sample_rate=sample_rate)

    # For this example, we generate a synthetic "reference" voice signal.
    # In production, you would load a real WAV file:
    #   audio, sr = processor.from_wav_bytes(open("speaker.wav", "rb").read())
    duration_sec = 30
    t = np.arange(int(sample_rate * duration_sec)) / sample_rate
    reference_audio = (
        0.3 * np.sin(2 * np.pi * 180 * t)   # fundamental
        + 0.15 * np.sin(2 * np.pi * 360 * t)  # 2nd harmonic
        + 0.08 * np.sin(2 * np.pi * 540 * t)  # 3rd harmonic
        + 0.02 * np.random.randn(len(t))       # noise
    ).astype(np.float32)

    # Amplitude modulation to simulate speech rhythm
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)  # ~4 Hz modulation
    reference_audio *= envelope

    # Clone the voice
    print("Extracting speaker profile...")
    profile = cloner.clone_voice(reference_audio, name="demo_speaker")
    print(f"  F0: {profile.fundamental_freq:.1f} Hz")
    print(f"  Spectral centroid: {profile.spectral_centroid:.1f} Hz")
    print(f"  Speaking rate: {profile.speaking_rate:.1f} syl/s")
    print(f"  Embedding shape: {profile.embedding.shape}")

    # Use the cloned voice with the synthesizer
    config = SynthesisConfig(model_name="vits", sample_rate=sample_rate)
    synth = VoiceSynthesizer(config)
    synth.load_model()
    synth.register_voice("demo_speaker", profile.embedding)

    text = "This sentence is synthesized using a cloned voice."
    output = synth.synthesize(text, voice="demo_speaker")
    print(f"\nSynthesized: {output.duration_seconds:.2f}s with cloned voice")

    # Save output
    wav = processor.to_wav_bytes(output.audio, output.sample_rate)
    with open("output_cloned.wav", "wb") as f:
        f.write(wav)
    print("Saved to output_cloned.wav")

    # Interpolate between two voices
    print("\nCreating a second voice profile...")
    reference2 = (
        0.3 * np.sin(2 * np.pi * 250 * t)
        + 0.1 * np.sin(2 * np.pi * 500 * t)
        + 0.02 * np.random.randn(len(t))
    ).astype(np.float32) * envelope

    profile2 = cloner.clone_voice(reference2, name="speaker_2")

    blended = cloner.interpolate_voices(profile, profile2, alpha=0.5)
    synth.register_voice("blended", blended.embedding)

    output_blended = synth.synthesize(text, voice="blended")
    print(f"Blended voice synthesis: {output_blended.duration_seconds:.2f}s")

    print(f"\nRegistered voices: {synth.get_voices()}")


if __name__ == "__main__":
    main()
