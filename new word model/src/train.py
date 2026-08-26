"""
Training loop for the online handwriting recognition model.

Features:
  - CTC loss with Adam optimizer
  - ReduceLROnPlateau scheduler
  - Gradient clipping for LSTM stability
  - Per-epoch validation with CER tracking
  - Early stopping on validation CER
  - Checkpoint saving (best model + periodic)
"""

import os
import time

import torch
import torch.nn as nn
from tqdm import tqdm

from . import config
from .data_parser import load_or_build_dataset
from .dataset import (
    create_dataloaders,
    ctc_greedy_decode,
    decode_label,
)
from .model import HandwritingRecognizer

try:
    import editdistance
    HAS_EDITDISTANCE = True
except ImportError:
    HAS_EDITDISTANCE = False


def compute_cer(predicted_text, target_text):
    """
    Compute Character Error Rate (CER) between predicted and target text.

    CER = edit_distance(pred, target) / len(target)
    """
    if len(target_text) == 0:
        return 0.0 if len(predicted_text) == 0 else 1.0

    if HAS_EDITDISTANCE:
        dist = editdistance.eval(predicted_text, target_text)
    else:
        # Simple fallback using difflib
        dist = _levenshtein(predicted_text, target_text)

    return dist / len(target_text)


def _levenshtein(s1, s2):
    """Levenshtein distance fallback if editdistance is not installed."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def validate(model, val_loader, ctc_loss_fn, device):
    """
    Run validation and compute average loss and CER.

    Returns:
        (avg_loss, avg_cer, sample_predictions)
    """
    model.eval()
    total_loss = 0.0
    total_cer = 0.0
    num_batches = 0
    num_samples = 0
    sample_preds = []

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)
            seq_lens = batch["seq_lens"]
            label_lens = batch["label_lens"]
            texts = batch["texts"]

            # Forward pass
            log_probs = model(features, seq_lens)

            # CTC loss
            loss = ctc_loss_fn(log_probs, labels, seq_lens, label_lens)
            total_loss += loss.item()
            num_batches += 1

            # Decode predictions and compute CER
            batch_size = features.size(0)
            for i in range(batch_size):
                # Get log probs for this sample (up to actual seq length)
                sample_log_probs = log_probs[:seq_lens[i], i, :]
                pred_indices = ctc_greedy_decode(sample_log_probs)
                pred_text = decode_label(pred_indices)
                target_text = texts[i]

                cer = compute_cer(pred_text, target_text)
                total_cer += cer
                num_samples += 1

                # Save a few sample predictions for display
                if len(sample_preds) < 5:
                    sample_preds.append((target_text, pred_text, cer))

    avg_loss = total_loss / max(num_batches, 1)
    avg_cer = total_cer / max(num_samples, 1)

    return avg_loss, avg_cer, sample_preds


def train():
    """Main training function."""
    print("=" * 70)
    print("Online Handwriting Recognition — Training")
    print("=" * 70)

    # Set seed for reproducibility
    torch.manual_seed(config.SEED)

    device = config.DEVICE
    print(f"Device: {device}")

    # ─── Load Data ───────────────────────────────────────────────────────
    print("\n[1/4] Loading data...")
    train_data, val_data, test_data = load_or_build_dataset()

    print("\n[2/4] Creating data loaders...")
    train_loader, val_loader, _ = create_dataloaders(
        train_data, val_data, test_data
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")

    # ─── Model ───────────────────────────────────────────────────────────
    print("\n[3/4] Initializing model...")
    model = HandwritingRecognizer().to(device)
    print(f"  {model}")

    # ─── Training Setup ──────────────────────────────────────────────────
    ctc_loss_fn = nn.CTCLoss(blank=config.BLANK_IDX, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.LR_SCHEDULER_FACTOR,
        patience=config.LR_SCHEDULER_PATIENCE,
    )

    # Checkpoint directory
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # Early stopping
    best_cer = float("inf")
    patience_counter = 0

    # ─── Training Loop ───────────────────────────────────────────────────
    print("\n[4/4] Starting training...")
    print(f"  Epochs: {config.NUM_EPOCHS}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Learning rate: {config.LEARNING_RATE}")
    print()

    for epoch in range(1, config.NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        start_time = time.time()

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch:3d}/{config.NUM_EPOCHS}",
            leave=False,
        )

        for batch in progress:
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)
            seq_lens = batch["seq_lens"]
            label_lens = batch["label_lens"]

            # Forward pass
            log_probs = model(features, seq_lens)

            # CTC loss
            loss = ctc_loss_fn(log_probs, labels, seq_lens, label_lens)

            # Skip if loss is inf or nan
            if torch.isinf(loss) or torch.isnan(loss):
                continue

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.GRAD_CLIP_MAX_NORM
            )

            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

        # Epoch stats
        avg_train_loss = epoch_loss / max(num_batches, 1)
        elapsed = time.time() - start_time

        # Validation
        val_loss, val_cer, sample_preds = validate(
            model, val_loader, ctc_loss_fn, device
        )

        # Learning rate scheduling
        scheduler.step(val_cer)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{config.NUM_EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val CER: {val_cer:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {elapsed:.1f}s"
        )

        # Print sample predictions every 5 epochs
        if epoch % 5 == 0 and sample_preds:
            print("  ── Sample Predictions ──")
            for target, pred, cer in sample_preds[:3]:
                print(f"    Target: '{target}'")
                print(f"    Pred:   '{pred}'")
                print(f"    CER:    {cer:.4f}")
                print()

        # ─── Checkpointing & Early Stopping ──────────────────────────────
        if val_cer < best_cer:
            best_cer = val_cer
            patience_counter = 0

            # Save best model
            best_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_cer": val_cer,
                "val_loss": val_loss,
            }, best_path)
            print(f"  ✓ New best model saved (CER: {val_cer:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"\n  Early stopping after {epoch} epochs (no improvement for {config.EARLY_STOPPING_PATIENCE} epochs)")
                break

        # Save periodic checkpoint every 10 epochs
        if epoch % 10 == 0:
            ckpt_path = os.path.join(
                config.CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pt"
            )
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_cer": val_cer,
                "val_loss": val_loss,
            }, ckpt_path)

    print("\n" + "=" * 70)
    print(f"Training complete. Best validation CER: {best_cer:.4f}")
    print(f"Best model saved to: {os.path.join(config.CHECKPOINT_DIR, 'best_model.pt')}")
    print("=" * 70)


if __name__ == "__main__":
    train()

