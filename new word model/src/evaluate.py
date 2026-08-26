"""
Evaluation and inference for the online handwriting recognition model.

Features:
  - Load trained model from checkpoint
  - Compute CER (Character Error Rate) and WER (Word Error Rate) on test set
  - Greedy CTC decoding
  - Print detailed predictions vs ground truth
  - Standalone inference function for new stroke data
"""

import os

import torch
from tqdm import tqdm

from . import config
from .data_parser import load_or_build_dataset
from .dataset import (
    create_dataloader,
    create_dataloaders,
    ctc_greedy_decode,
    decode_label,
    HandwritingDataset,
)
from .model import HandwritingRecognizer
from .features import extract_features
from .train import compute_cer

try:
    import editdistance
    HAS_EDITDISTANCE = True
except ImportError:
    HAS_EDITDISTANCE = False


def compute_wer(predicted_text, target_text):
    """
    Compute Word Error Rate (WER) between predicted and target text.

    WER = edit_distance(pred_words, target_words) / len(target_words)
    """
    pred_words = predicted_text.split()
    target_words = target_text.split()

    if len(target_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0

    if HAS_EDITDISTANCE:
        dist = editdistance.eval(pred_words, target_words)
    else:
        dist = _levenshtein_words(pred_words, target_words)

    return dist / len(target_words)


def _levenshtein_words(s1, s2):
    """Levenshtein distance on word lists."""
    if len(s1) < len(s2):
        return _levenshtein_words(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, w1 in enumerate(s1):
        curr_row = [i + 1]
        for j, w2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (w1 != w2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def load_model(checkpoint_path=None):
    """
    Load a trained model from a checkpoint file.

    Args:
        checkpoint_path: path to .pt checkpoint file.
                         If None, loads the best model.

    Returns:
        (model, checkpoint_info)
    """
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading model from: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)

    model = HandwritingRecognizer().to(config.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    info = {
        "epoch": checkpoint.get("epoch", "?"),
        "val_cer": checkpoint.get("val_cer", "?"),
        "val_loss": checkpoint.get("val_loss", "?"),
    }
    print(f"  Epoch: {info['epoch']}, Val CER: {info['val_cer']}, Val Loss: {info['val_loss']}")

    return model, info


def evaluate_test_set(model=None, checkpoint_path=None):
    """
    Evaluate the model on the test set.

    Computes CER and WER, prints detailed results.
    """
    device = config.DEVICE

    # Load model if not provided
    if model is None:
        model, _ = load_model(checkpoint_path)

    # Load data
    _, _, test_data = load_or_build_dataset()
    test_loader = create_dataloader(test_data, batch_size=config.BATCH_SIZE, shuffle=False)

    print(f"\nEvaluating on {len(test_data)} test samples...")
    print("=" * 70)

    model.eval()
    total_cer = 0.0
    total_wer = 0.0
    num_samples = 0
    all_results = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            features = batch["features"].to(device)
            seq_lens = batch["seq_lens"]
            texts = batch["texts"]

            # Forward pass
            log_probs = model(features, seq_lens)

            # Decode each sample
            batch_size = features.size(0)
            for i in range(batch_size):
                sample_log_probs = log_probs[:seq_lens[i], i, :]
                pred_indices = ctc_greedy_decode(sample_log_probs)
                pred_text = decode_label(pred_indices)
                target_text = texts[i]

                cer = compute_cer(pred_text, target_text)
                wer = compute_wer(pred_text, target_text)

                total_cer += cer
                total_wer += wer
                num_samples += 1

                all_results.append({
                    "target": target_text,
                    "predicted": pred_text,
                    "cer": cer,
                    "wer": wer,
                })

    avg_cer = total_cer / max(num_samples, 1)
    avg_wer = total_wer / max(num_samples, 1)

    print(f"\n{'='*70}")
    print(f"Test Results ({num_samples} samples)")
    print(f"{'='*70}")
    print(f"  Character Error Rate (CER): {avg_cer:.4f} ({avg_cer*100:.2f}%)")
    print(f"  Word Error Rate (WER):      {avg_wer:.4f} ({avg_wer*100:.2f}%)")

    # Print detailed samples
    print(f"\n{'─'*70}")
    print("Sample Predictions (first 10):")
    print(f"{'─'*70}")

    for i, result in enumerate(all_results[:10]):
        print(f"\n  [{i+1}] Target:    '{result['target']}'")
        print(f"       Predicted: '{result['predicted']}'")
        print(f"       CER: {result['cer']:.4f}, WER: {result['wer']:.4f}")

    # Error distribution
    cers = [r["cer"] for r in all_results]
    perfect = sum(1 for c in cers if c == 0.0)
    good = sum(1 for c in cers if 0.0 < c <= 0.1)
    moderate = sum(1 for c in cers if 0.1 < c <= 0.3)
    poor = sum(1 for c in cers if c > 0.3)

    print(f"\n{'─'*70}")
    print("Error Distribution:")
    print(f"  Perfect (CER = 0):       {perfect:4d} ({perfect/num_samples*100:.1f}%)")
    print(f"  Good (CER ≤ 0.1):        {good:4d} ({good/num_samples*100:.1f}%)")
    print(f"  Moderate (CER ≤ 0.3):    {moderate:4d} ({moderate/num_samples*100:.1f}%)")
    print(f"  Poor (CER > 0.3):        {poor:4d} ({poor/num_samples*100:.1f}%)")
    print(f"{'='*70}")

    return avg_cer, avg_wer, all_results


def predict(model, strokes):
    """
    Run inference on a single stroke sequence.

    Args:
        model: trained HandwritingRecognizer
        strokes: list of strokes (list of point dicts with 'x', 'y', 'time')

    Returns:
        str: predicted text
    """
    model.eval()
    device = config.DEVICE

    # Extract features
    features = extract_features(strokes)
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    seq_lens = torch.tensor([len(features)], dtype=torch.long)

    with torch.no_grad():
        log_probs = model(features_tensor, seq_lens)
        pred_indices = ctc_greedy_decode(log_probs[:, 0, :])
        pred_text = decode_label(pred_indices)

    return pred_text


if __name__ == "__main__":
    evaluate_test_set()

