"""
emnist_symbols_dataset.py

Two dataset classes that share ONE preprocessing pipeline with main.py's
live inference path (preprocess_to_tensor), so what the model sees during
training matches what it sees in production.

  - EMNISTFixed:        EMNIST ByMerge (digits + case-aware letters, 47
                         classes), auto-downloaded via torchvision. The
                         well-known EMNIST storage quirk (rotated 90° +
                         mirrored) is corrected here, ONCE, so the model
                         never has to deal with it at inference time.

  - SymbolFolderDataset: Loads hand-picked symbol classes (+, -, ×, ÷, =,
                          etc.) from a Kaggle-style "one folder per class"
                          directory you download separately. No orientation
                          fix applied — these images are already upright.
"""

import glob
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import EMNIST
from torchvision.transforms import functional as TF

MEAN, STD = 0.1307, 0.3081


def emnist_orientation_fix(img: Image.Image) -> Image.Image:
    """
    Standard correction for EMNIST's raw storage orientation. Applied once,
    here, at data-prep time — NOT at inference time. This is what makes
    apply_orientation_fix() in main.py safe to leave as identity.
    """
    img = TF.rotate(img, -90)
    img = TF.hflip(img)
    return img


def preprocess_to_tensor(img: Image.Image) -> torch.Tensor:
    """
    Must exactly mirror the preprocessing in main.py's predict() endpoint:
    grayscale -> 28x28 -> auto-invert to white-stroke-on-black -> normalize.
    """
    img = img.convert("L").resize((28, 28), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    if arr.mean() < 0.5:
        pass  # already dark background, correct
    else:
        arr = 1.0 - arr  # invert if white background
    arr = (arr - MEAN) / STD
    return torch.tensor(arr).unsqueeze(0)  # shape: (1, 28, 28)


class EMNISTFixed(Dataset):
    """Wraps torchvision's EMNIST(split='bymerge') with the orientation fix."""

    def __init__(self, root, train=True, download=True):
        self.base = EMNIST(root=root, split="bymerge", train=train, download=download)
        self.classes = list(self.base.classes)  # ground-truth label strings, e.g. '0'..'9','A'..'Z',...

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        img = emnist_orientation_fix(img)
        tensor = preprocess_to_tensor(img)
        return tensor, label


class SymbolFolderDataset(Dataset):
    """
    Loads symbol images from a Kaggle-style folder-per-class directory.

    folder_to_symbol: dict mapping the ACTUAL folder name on disk to the
    display character you want, e.g. {"times": "×", "+": "+", ...}.
    Run list_symbol_folders() first to see what's actually in your
    extracted dataset before filling this in — folder naming conventions
    vary between dataset versions.

    label_offset: shifts local indices so they continue right after
    EMNIST's classes in the unified label space (set by build_labels.py).
    """

    def __init__(self, root_dir, folder_to_symbol: dict, label_offset: int):
        self.symbols = list(folder_to_symbol.values())  # display chars, in class order
        self.samples = []  # (path, local_idx)
        for local_idx, folder in enumerate(folder_to_symbol.keys()):
            folder_path = os.path.join(root_dir, folder)
            if not os.path.isdir(folder_path):
                print(f"⚠️  Warning: folder '{folder}' not found under {root_dir}, skipping.")
                continue
            paths = glob.glob(os.path.join(folder_path, "*"))
            for p in paths:
                self.samples.append((p, local_idx))
        self.label_offset = label_offset
        print(f"SymbolFolderDataset: loaded {len(self.samples)} images across {len(self.symbols)} symbol classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, local_idx = self.samples[idx]
        img = Image.open(path)
        tensor = preprocess_to_tensor(img)
        return tensor, local_idx + self.label_offset


def list_symbol_folders(root_dir):
    """Utility: print the actual folder names in your extracted dataset,
    so you can build an accurate folder_to_symbol map."""
    if not os.path.isdir(root_dir):
        print(f"Directory not found: {root_dir}")
        return
    names = sorted(os.listdir(root_dir))
    print(f"Found {len(names)} folders under {root_dir}:")
    for n in names:
        full = os.path.join(root_dir, n)
        if os.path.isdir(full):
            count = len(glob.glob(os.path.join(full, "*")))
            print(f"  {n:<25} ({count} files)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        list_symbol_folders(sys.argv[1])
    else:
        print("Usage: python emnist_symbols_dataset.py /path/to/extracted/symbols/dataset")
