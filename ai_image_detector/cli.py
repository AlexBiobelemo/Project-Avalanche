from __future__ import annotations

import argparse
import sys
from typing import List

import numpy as np

from .preprocess import load_image, preprocess_image, PreprocessConfig, ImageLoadError
from .inference import run_inference_on_array, format_prediction


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify images as real or AI-generated")
    parser.add_argument("images", nargs="+", help="Path(s) to image file(s)")
    parser.add_argument("--size", type=int, default=224, help="Target size (square) for model input")
    parser.add_argument("--mean", type=float, nargs=3, default=(0.5, 0.5, 0.5), help="Normalization mean per channel")
    parser.add_argument("--std", type=float, nargs=3, default=(0.5, 0.5, 0.5), help="Normalization std per channel")
    parser.add_argument("--raw", action="store_true", help="Print raw scores JSON-like after the summary line")
    parser.add_argument("--no-ensemble", action="store_true", help="Disable ensemble + TTA (heuristic if no torch)")
    parser.add_argument("--model", type=str, default=None, help="Transformers model name or local path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for AI label")
    parser.add_argument("--weights", type=float, nargs=2, default=None, metavar=("HEURISTIC", "VIT"), help="Ensemble weights (heuristic vit)")
    args = parser.parse_args(argv)
    
    # Convert lists to tuples for mean, std, and weights
    if isinstance(args.mean, list):
        args.mean = tuple(args.mean)
    if isinstance(args.std, list):
        args.std = tuple(args.std)
    if isinstance(args.weights, list):
        args.weights = tuple(args.weights)
    
    return args


def main(argv: List[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    cfg = PreprocessConfig(target_size=(args.size, args.size), normalize_mean=tuple(args.mean), normalize_std=tuple(args.std))

    exit_code = 0
    for path in args.images:
        try:
            img = load_image(path)
            arr_chw = preprocess_image(img, cfg)
            pred = run_inference_on_array(
                arr_chw,
                use_ensemble=not args.no_ensemble,
                model_name_or_path=args.model,
                ensemble_weights=tuple(args.weights) if args.weights else None,
            )
            summary = format_prediction(pred)
            label = "AI-generated" if (pred.raw_scores or {}).get("ai-generated", 1 - pred.confidence) >= args.threshold else "Real"
            print(f"{path}: {label} ({summary})")
            if args.raw and pred.raw_scores is not None:
                print(f"  scores: {pred.raw_scores}")
        except ImageLoadError as e:
            print(f"{path}: ERROR - {e}")
            exit_code = 2
        except Exception as e:
            print(f"{path}: ERROR during inference - {e}")
            exit_code = 3

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())


