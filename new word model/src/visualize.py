"""
Visualization tools for online handwriting data and model predictions.

Supports:
  - Plotting raw stroke sequences with stroke order color-coding and pen-up indicators
  - Visualizing ground-truth vs model prediction side-by-side
  - Saving stroke renderings to image files or rendering in Jupyter/Colab
"""

import os
import matplotlib.pyplot as plt
import numpy as np


def plot_strokes(strokes, title=None, save_path=None, show=True, figsize=(12, 3)):
    """
    Plot online handwriting strokes as 2D curves.

    Each stroke is plotted in a different color to show stroke segmentation,
    with a small dot at the start of each stroke.

    Args:
        strokes: list of strokes (each stroke is a list of {'x', 'y', 'time'} dicts)
        title: optional plot title
        save_path: optional path to save the generated figure
        show: whether to call plt.show()
        figsize: matplotlib figure size tuple (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(strokes), 1)))

    for i, stroke in enumerate(strokes):
        if not stroke:
            continue
        xs = [p["x"] for p in stroke]
        ys = [p["y"] for p in stroke]

        color = colors[i % len(colors)]

        # Plot stroke line
        ax.plot(xs, ys, color=color, linewidth=2.0, alpha=0.85)

        # Mark starting point of stroke
        ax.plot(xs[0], ys[0], "o", color=color, markersize=4)

    # Invert Y-axis because screen coordinates have (0, 0) at top-left
    ax.invert_yaxis()
    ax.set_aspect("equal", "datalim")
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=12, pad=10)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_prediction_sample(strokes, target_text, pred_text=None, cer=None, save_path=None, show=True):
    """
    Plot a sample with its stroke rendering and comparison between target and prediction.

    Args:
        strokes: list of strokes
        target_text: ground truth transcription
        pred_text: model predicted transcription (optional)
        cer: Character Error Rate (optional)
        save_path: optional output image path
        show: whether to display the figure
    """
    title_parts = [f"Ground Truth: '{target_text}'"]
    if pred_text is not None:
        title_parts.append(f"Predicted: '{pred_text}'")
    if cer is not None:
        title_parts.append(f"CER: {cer:.2%}")

    title = "\n".join(title_parts)
    return plot_strokes(strokes, title=title, save_path=save_path, show=show)

