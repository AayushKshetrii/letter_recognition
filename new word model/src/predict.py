"""
CLI and Python API for running inference on online handwriting stroke XML files.

Usage from terminal:
    python -m src.predict --xml datasets/lineStrokes/a01/a01-000/a01-000u-01.xml
    python -m src.predict --xml datasets/lineStrokes/a01/a01-000/a01-000u-01.xml --visualize
"""

import argparse
import os
import sys

from . import config
from .data_parser import parse_stroke_xml
from .evaluate import load_model, predict


def main():
    parser = argparse.ArgumentParser(description="Predict text from online handwriting stroke XML.")
    parser.add_argument("--xml", type=str, required=True, help="Path to stroke XML file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pt)")
    parser.add_argument("--visualize", action="store_true", help="Plot and display strokes")
    parser.add_argument("--save-plot", type=str, default=None, help="Path to save visualization plot")

    args = parser.parse_args()

    if not os.path.isfile(args.xml):
        print(f"Error: XML file not found at '{args.xml}'", file=sys.stderr)
        sys.exit(1)

    print(f"Parsing strokes from: {args.xml}")
    strokes = parse_stroke_xml(args.xml)
    total_points = sum(len(s) for s in strokes)
    print(f"Found {len(strokes)} strokes, {total_points} total coordinate points.")

    ckpt_path = args.checkpoint or os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
    if not os.path.isfile(ckpt_path):
        print(f"Checkpoint not found at '{ckpt_path}'. Please train the model first with 'python -m src.train'", file=sys.stderr)
        sys.exit(1)

    model, info = load_model(ckpt_path)
    pred_text = predict(model, strokes)

    print("\n" + "=" * 60)
    print(f"Predicted Text: \"{pred_text}\"")
    print("=" * 60)

    if args.visualize or args.save_plot:
        try:
            from .visualize import plot_strokes
            plot_strokes(
                strokes,
                title=f"Prediction: '{pred_text}'",
                save_path=args.save_plot,
                show=args.visualize,
            )
        except Exception as e:
            print(f"Visualization warning: {e}")


if __name__ == "__main__":
    main()

