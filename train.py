"""
train.py — retrains LetterNet on letters + digits + math symbols.

Run order:
  1. python emnist_symbols_dataset.py /path/to/extracted/symbols  (inspect folder names)
  2. edit SYMBOL_FOLDER_MAP in build_labels.py to match
  3. python build_labels.py            (writes labels.json)
  4. python train.py --symbols_dir /path/to/extracted/symbols

On M1 Macs this uses the MPS backend automatically (falls back to CPU for
any op MPS doesn't support, via PYTORCH_ENABLE_MPS_FALLBACK).
"""

import argparse
import json
import os
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, random_split

from emnist_symbols_dataset import EMNISTFixed, SymbolFolderDataset

EMNIST_ROOT = "data"
SEED = 42


class LetterNet(nn.Module):
    """Same architecture as main.py — only num_classes changes."""
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2),  # 7x7 -> 3x3, same as AdaptiveAvgPool2d(3)
            # NOTE: AdaptiveAvgPool2d(3) was swapped for this because MPS
            # (Apple Silicon GPU backend) doesn't support adaptive pooling
            # when input size isn't evenly divisible by output size (7 / 3).
            # AvgPool2d with matching kernel/stride produces the identical
            # 3x3 output for our fixed 28x28 input, so this is a drop-in
            # swap with no effect on training or accuracy.
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 9, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def compute_class_weights(emnist_train, symbols_train, num_classes, max_weight_ratio=10.0):
    """
    EMNIST ByMerge has significant class imbalance (some classes have far
    more samples than others). Under-represented classes get out-trained
    by common ones during plain cross-entropy training — this is a likely
    cause of confusions like a rare class ('r') being predicted as a
    common one ('Y') that just dominated gradient updates.

    Weights each class inversely to its frequency, clipped to
    max_weight_ratio so a near-empty class can't blow up training with a
    huge weight and destabilize everything else.
    """
    counts = torch.zeros(num_classes, dtype=torch.long)

    # EMNIST: torchvision exposes .targets directly, no need to load images
    emnist_targets = emnist_train.base.targets
    for t in emnist_targets.tolist():
        counts[t] += 1

    # symbols_train is a torch Subset wrapping SymbolFolderDataset — pull
    # labels from the underlying .samples list, again without loading images
    underlying = symbols_train.dataset
    for i in symbols_train.indices:
        _, label = underlying.samples[i]
        counts[label] += 1

    counts = counts.clamp(min=1)  # avoid divide-by-zero for any class with 0 samples
    inv_freq = counts.float().mean() / counts.float()
    weights = inv_freq.clamp(max=max_weight_ratio)
    return weights


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_datasets(symbols_dir, labels_config, val_frac_symbols=0.1):
    emnist_train = EMNISTFixed(EMNIST_ROOT, train=True, download=True)
    emnist_val = EMNISTFixed(EMNIST_ROOT, train=False, download=True)

    symbol_offset = labels_config["emnist_class_count"]
    symbols_full = SymbolFolderDataset(
        symbols_dir, labels_config["symbol_folder_map"], label_offset=symbol_offset
    )
    if len(symbols_full) == 0:
        raise RuntimeError(
            "No symbol images loaded — check --symbols_dir and SYMBOL_FOLDER_MAP "
            "in build_labels.py match your extracted dataset's folder names."
        )

    n_val = max(1, int(len(symbols_full) * val_frac_symbols))
    n_train = len(symbols_full) - n_val
    g = torch.Generator().manual_seed(SEED)
    symbols_train, symbols_val = random_split(symbols_full, [n_train, n_val], generator=g)

    train_ds = ConcatDataset([emnist_train, symbols_train])
    val_ds = ConcatDataset([emnist_val, symbols_val])
    return train_ds, val_ds, emnist_train, symbols_train


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def train(args):
    with open("labels.json") as f:
        labels_config = json.load(f)
    num_classes = labels_config["num_classes"]
    print(f"Training for {num_classes} classes: {labels_config['classes']}")

    device = get_device()
    print(f"Using device: {device}")

    train_ds, val_ds, emnist_train, symbols_train = build_datasets(args.symbols_dir, labels_config)
    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    model = LetterNet(num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print("Computing class weights to counter EMNIST's known class imbalance...")
    class_weights = compute_class_weights(emnist_train, symbols_train, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 200 == 0:
                print(f"  epoch {epoch} step {i}/{len(train_loader)} loss {loss.item():.4f}")

        scheduler.step()
        val_acc = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch}/{args.epochs} — loss {avg_loss:.4f} — val_acc {val_acc*100:.2f}% — {elapsed:.0f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.out)
            print(f"  ✓ New best ({val_acc*100:.2f}%) — saved to {args.out}")

    print(f"\nDone. Best val accuracy: {best_val_acc*100:.2f}%. Model saved to {args.out}")
    print("Next: point MODEL_PATH in main.py to this file, and load LABELS from labels.json.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols_dir", type=str, required=True,
                         help="Path to extracted Kaggle symbols dataset")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--out", type=str, default="letter_digit_symbol_recognizer_final.pth")
    args = parser.parse_args()
    train(args)