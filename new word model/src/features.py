"""
Stroke feature engineering for online handwriting recognition.

Converts raw (x, y, time) stroke point sequences into a rich feature vector
per timestep suitable for LSTM input.

Features per timestep (11 total):
  0. dx        - horizontal displacement (delta x)
  1. dy        - vertical displacement (delta y)
  2. dt        - time delta
  3. pen_up    - binary: 1 if pen was lifted before this point (stroke boundary)
  4. cos_theta - cosine of writing angle
  5. sin_theta - sine of writing angle
  6. d2x       - second-order delta x (acceleration / curvature)
  7. d2y       - second-order delta y
  8. x_norm    - normalized absolute x position
  9. y_norm    - normalized absolute y position
  10. speed    - instantaneous speed (distance / time)
"""

import numpy as np


def strokes_to_points(strokes):
    """
    Flatten a list of strokes into a single sequence of points with pen-up flags.

    Args:
        strokes: list of strokes, each stroke is a list of point dicts
                 with keys 'x', 'y', 'time'

    Returns:
        np.ndarray of shape (N, 4): columns are [x, y, time, pen_up]
        pen_up is 1.0 at the first point of each stroke (except the very first).
    """
    all_points = []

    for stroke_idx, stroke in enumerate(strokes):
        for point_idx, point in enumerate(stroke):
            # pen_up = 1 at the start of a new stroke (except the first stroke)
            pen_up = 1.0 if (stroke_idx > 0 and point_idx == 0) else 0.0
            all_points.append([
                float(point["x"]),
                float(point["y"]),
                float(point["time"]),
                pen_up,
            ])

    return np.array(all_points, dtype=np.float32)


def extract_features(strokes):
    """
    Extract a feature vector for each timestep from raw stroke data.

    Args:
        strokes: list of strokes (list of point dicts with 'x', 'y', 'time')

    Returns:
        np.ndarray of shape (N, 11): feature matrix
    """
    points = strokes_to_points(strokes)
    N = len(points)

    if N == 0:
        return np.zeros((0, 11), dtype=np.float32)

    x = points[:, 0]
    y = points[:, 1]
    t = points[:, 2]
    pen_up = points[:, 3]

    # ─── First-order deltas ──────────────────────────────────────────────
    dx = np.zeros(N, dtype=np.float32)
    dy = np.zeros(N, dtype=np.float32)
    dt = np.zeros(N, dtype=np.float32)

    dx[1:] = x[1:] - x[:-1]
    dy[1:] = y[1:] - y[:-1]
    dt[1:] = t[1:] - t[:-1]

    # Zero out deltas at pen-up boundaries (stroke transitions are not continuous)
    dx[pen_up == 1.0] = 0.0
    dy[pen_up == 1.0] = 0.0
    dt[pen_up == 1.0] = 0.0

    # Avoid division by zero in time
    dt_safe = np.where(dt > 0, dt, 1.0)

    # ─── Writing angle ───────────────────────────────────────────────────
    dist = np.sqrt(dx**2 + dy**2)
    dist_safe = np.where(dist > 0, dist, 1.0)

    cos_theta = dx / dist_safe
    sin_theta = dy / dist_safe

    # Zero angle at stroke boundaries
    cos_theta[pen_up == 1.0] = 0.0
    sin_theta[pen_up == 1.0] = 0.0

    # ─── Second-order deltas (curvature) ─────────────────────────────────
    d2x = np.zeros(N, dtype=np.float32)
    d2y = np.zeros(N, dtype=np.float32)

    d2x[2:] = dx[2:] - dx[1:-1]
    d2y[2:] = dy[2:] - dy[1:-1]

    # ─── Normalized absolute position ────────────────────────────────────
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_range = x_max - x_min if x_max > x_min else 1.0
    y_range = y_max - y_min if y_max > y_min else 1.0

    x_norm = (x - x_min) / x_range
    y_norm = (y - y_min) / y_range

    # ─── Speed ───────────────────────────────────────────────────────────
    speed = dist / dt_safe
    speed[pen_up == 1.0] = 0.0

    # ─── Assemble feature matrix ─────────────────────────────────────────
    features = np.stack([
        dx, dy, dt, pen_up,
        cos_theta, sin_theta,
        d2x, d2y,
        x_norm, y_norm,
        speed,
    ], axis=1)

    # ─── Per-sample normalization ────────────────────────────────────────
    features = normalize_features(features)

    return features


def normalize_features(features):
    """
    Apply per-sample z-normalization to continuous features.

    The pen_up feature (index 3) is binary and left unnormalized.

    Args:
        features: np.ndarray of shape (N, 11)

    Returns:
        np.ndarray of shape (N, 11): normalized features
    """
    features = features.copy()

    # Indices of continuous features (exclude pen_up at index 3)
    continuous_indices = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]

    for idx in continuous_indices:
        col = features[:, idx]
        mean = col.mean()
        std = col.std()
        if std > 1e-6:
            features[:, idx] = (col - mean) / std
        else:
            features[:, idx] = 0.0

    return features

