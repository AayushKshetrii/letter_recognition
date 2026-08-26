"""
Configuration for the Online Handwriting Recognition system.
Central place for all hyperparameters, paths, and constants.
"""

import os
import string

# ─── Project Paths ───────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets")

# Archive paths
ASCII_ARCHIVE = os.path.join(DATASET_DIR, "ascii-all.tar.gz")
STROKES_ARCHIVE = os.path.join(DATASET_DIR, "lineStrokes-all.tar.gz")

# Extracted data directories
ASCII_DIR = os.path.join(DATASET_DIR, "ascii")
STROKES_DIR = os.path.join(DATASET_DIR, "lineStrokes")

# Processed / cached data
CACHE_DIR = os.path.join(DATASET_DIR, "processed")
CACHE_FILE = os.path.join(CACHE_DIR, "parsed_data.pkl")

# Model checkpoints
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

# ─── Character Set ───────────────────────────────────────────────────────────

# All printable ASCII characters that appear in the IAM dataset
# Includes letters, digits, punctuation, and space
CHARACTERS = string.ascii_letters + string.digits + string.punctuation + " "

# CTC blank token index (conventionally index 0)
BLANK_IDX = 0

# Character-to-index mapping (index 0 reserved for CTC blank)
CHAR_TO_IDX = {ch: i + 1 for i, ch in enumerate(CHARACTERS)}
IDX_TO_CHAR = {i + 1: ch for i, ch in enumerate(CHARACTERS)}
IDX_TO_CHAR[BLANK_IDX] = ""  # blank decodes to empty string

NUM_CLASSES = len(CHARACTERS) + 1  # +1 for CTC blank

# ─── Feature Parameters ─────────────────────────────────────────────────────

# Number of input features per timestep after feature engineering
# [dx, dy, dt, pen_up, cos_theta, sin_theta, d2x, d2y, x_norm, y_norm, speed]
INPUT_FEATURE_DIM = 11

# ─── Model Hyperparameters ───────────────────────────────────────────────────

HIDDEN_SIZE = 256         # LSTM hidden dimension
NUM_LSTM_LAYERS = 3       # Number of stacked LSTM layers
DROPOUT = 0.3             # Dropout between LSTM layers
BIDIRECTIONAL = True      # Use bidirectional LSTM

# ─── Training Hyperparameters ────────────────────────────────────────────────

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 80
GRAD_CLIP_MAX_NORM = 5.0  # Gradient clipping for LSTM stability
NUM_WORKERS = 2 if os.name != "nt" else 0  # 0 for Windows/constrained environments

# Learning rate scheduler
LR_SCHEDULER_FACTOR = 0.5     # Reduce LR by this factor
LR_SCHEDULER_PATIENCE = 5     # Epochs to wait before reducing LR

# Early stopping
EARLY_STOPPING_PATIENCE = 10  # Epochs without improvement before stopping

# ─── Data Split ──────────────────────────────────────────────────────────────

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Random seed for reproducibility
SEED = 42

# ─── Device & System ─────────────────────────────────────────────────────────

import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = torch.cuda.is_available()

