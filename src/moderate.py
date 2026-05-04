from __future__ import annotations

import argparse
import json

from src.inference import moderate_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run content moderation inference.")
    parser.add_argument("--text", required=True, help="Text to moderate.")
    parser.add_argument("--model-path", default=None, help="Optional model artifact path.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Fail if no trained model artifact is available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = moderate_text(
        args.text,
        model_path=args.model_path,
        allow_fallback=not args.no_fallback,
    ).to_dict()
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    print(f"Prediction: {result['predicted_label']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Latency: {result['latency_ms']:.3f} ms")
    print(f"Model: {result['model_name']}")
    print(f"Explanation: {result['explanation']}")


if __name__ == "__main__":
    main()

