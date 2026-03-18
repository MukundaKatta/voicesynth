"""Basic usage example for voicesynth."""
from src.core import Voicesynth

def main():
    instance = Voicesynth(config={"verbose": True})

    print("=== voicesynth Example ===\n")

    # Run primary operation
    result = instance.synthesize(input="example data", mode="demo")
    print(f"Result: {result}")

    # Run multiple operations
    ops = ["synthesize", "clone_voice", "set_emotion]
    for op in ops:
        r = getattr(instance, op)(source="example")
        print(f"  {op}: {"✓" if r.get("ok") else "✗"}")

    # Check stats
    print(f"\nStats: {instance.get_stats()}")

if __name__ == "__main__":
    main()
