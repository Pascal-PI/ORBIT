"""Predict Plastic, Metal, Paper, or Trash for a new image."""

import argparse
import json
from pathlib import Path

import torch

from model import CLASS_NAMES, DEFAULT_CHECKPOINT, image_tensor, load_checkpoint, select_device


def predict(image_path, checkpoint_path=DEFAULT_CHECKPOINT, device="auto"):
    """Return a category, softmax confidence, and all four class probabilities."""
    device = select_device(device)
    network, checkpoint = load_checkpoint(checkpoint_path, device)
    inputs = image_tensor(image_path, architecture=checkpoint["architecture"]).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = network(inputs).softmax(dim=1)[0].cpu()
    predicted = int(probabilities.argmax())
    return {
        "image": str(Path(image_path).resolve()),
        "category": CLASS_NAMES[predicted],
        "confidence": float(probabilities[predicted]),
        "probabilities": {
            name: float(probability)
            for name, probability in zip(CLASS_NAMES, probabilities)
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to a JPEG, PNG, or other Pillow-readable image")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--json", action="store_true", help="Print JSON for a future dashboard or robot")
    parser.add_argument("--output", type=Path, help="Also save the prediction as JSON")
    parser.add_argument("--threads", type=int, default=6, help="CPU threads (try 2 on a Raspberry Pi)")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    try:
        result = predict(args.image, args.checkpoint, args.device)
        serialized = json.dumps(result, indent=2, allow_nan=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized + "\n", encoding="utf-8")
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, f"Prediction failed: {error}\n")
    if args.json:
        print(serialized)
    else:
        print(f"Prediction: {result['category']}")
        print(f"Confidence: {result['confidence']:.1%} (uncalibrated softmax score)")
        for name, probability in result["probabilities"].items():
            print(f"  {name:8s} {probability:.1%}")
    return result


if __name__ == "__main__":
    main()
