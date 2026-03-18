"""CLI for voicesynth."""
import sys, json, argparse
from .core import Voicesynth

def main():
    parser = argparse.ArgumentParser(description="State-of-the-art open source text-to-speech with emotion control and voice cloning")
    parser.add_argument("command", nargs="?", default="status", choices=["status", "run", "info"])
    parser.add_argument("--input", "-i", default="")
    args = parser.parse_args()
    instance = Voicesynth()
    if args.command == "status":
        print(json.dumps(instance.get_stats(), indent=2))
    elif args.command == "run":
        print(json.dumps(instance.synthesize(input=args.input or "test"), indent=2, default=str))
    elif args.command == "info":
        print(f"voicesynth v0.1.0 — State-of-the-art open source text-to-speech with emotion control and voice cloning")

if __name__ == "__main__":
    main()
