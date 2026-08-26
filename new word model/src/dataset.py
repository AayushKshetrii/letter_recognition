"""
PyTorch Dataset and DataLoader for online handwriting recognition.

Handles variable-length stroke sequences with proper padding and collation
for batched training with CTC loss.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from . import config
from .features import extract_features


class HandwritingDataset(Dataset):
    """
    Dataset that yields (feature_sequence, label_indices) pairs.

    Each sample is a handwritten line: stroke features as input,
    character index sequence as target.
    """

    def __init__(self, data_samples):
        """
        Args:
            data_samples: list of dicts with 'strokes' and 'label' keys
                          (output from data_parser.load_or_build_dataset)
        """
        self.samples = data_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Extract features from stroke data
        features = extract_features(sample["strokes"])  # (seq_len, feature_dim)

        # Encode label to integer indices
        label_indices = encode_label(sample["label"])

        return {
            "features": torch.tensor(features, dtype=torch.float32),
            "label": torch.tensor(label_indices, dtype=torch.long),
            "seq_len": len(features),
            "label_len": len(label_indices),
            "text": sample["label"],
        }


def encode_label(text):
    """
    Convert a text string to a list of character indices.

    Args:
        text: string to encode

    Returns:
        list[int]: character indices (1-indexed, 0 reserved for CTC blank)
    """
    return [config.CHAR_TO_IDX[ch] for ch in text if ch in config.CHAR_TO_IDX]


def decode_label(indices):
    """
    Convert a sequence of character indices back to text.

    Args:
        indices: list or tensor of integer indices

    Returns:
        str: decoded text
    """
    if isinstance(indices, torch.Tensor):
        indices = indices.tolist()
    return "".join(config.IDX_TO_CHAR.get(idx, "") for idx in indices)


def ctc_greedy_decode(log_probs):
    """
    Greedy CTC decoding: take argmax at each timestep, collapse repeats,
    remove blanks.

    Args:
        log_probs: tensor of shape (seq_len, num_classes) — log probabilities

    Returns:
        list[int]: decoded character indices
    """
    # Argmax at each timestep
    predictions = torch.argmax(log_probs, dim=-1)  # (seq_len,)

    # Collapse consecutive duplicates and remove blanks
    decoded = []
    prev = -1
    for idx in predictions.tolist():
        if idx != prev:
            if idx != config.BLANK_IDX:
                decoded.append(idx)
        prev = idx

    return decoded


def collate_fn(batch):
    """
    Custom collate function for variable-length sequences.

    Pads feature sequences to the maximum length in the batch.
    Labels are concatenated (CTC convention).

    Args:
        batch: list of dicts from HandwritingDataset.__getitem__

    Returns:
        dict with:
            - 'features': (batch, max_seq_len, feature_dim) padded tensor
            - 'labels': (total_label_len,) concatenated label tensor
            - 'seq_lens': (batch,) actual sequence lengths
            - 'label_lens': (batch,) actual label lengths
            - 'texts': list of original text strings
    """
    # Sort by sequence length (descending) for pack_padded_sequence compatibility
    batch = sorted(batch, key=lambda x: x["seq_len"], reverse=True)

    max_seq_len = batch[0]["seq_len"]
    feature_dim = batch[0]["features"].shape[1]
    batch_size = len(batch)

    # Pad features
    padded_features = torch.zeros(batch_size, max_seq_len, feature_dim)
    seq_lens = torch.zeros(batch_size, dtype=torch.long)
    label_lens = torch.zeros(batch_size, dtype=torch.long)
    all_labels = []
    texts = []

    for i, sample in enumerate(batch):
        seq_len = sample["seq_len"]
        padded_features[i, :seq_len, :] = sample["features"]
        seq_lens[i] = seq_len
        label_lens[i] = sample["label_len"]
        all_labels.append(sample["label"])
        texts.append(sample["text"])

    # Concatenate all labels (CTC loss expects this format)
    labels = torch.cat(all_labels, dim=0)

    return {
        "features": padded_features,
        "labels": labels,
        "seq_lens": seq_lens,
        "label_lens": label_lens,
        "texts": texts,
    }


def create_dataloader(data_samples, batch_size=None, shuffle=False, num_workers=None):
    """
    Create a single PyTorch DataLoader for a list of data samples.

    Args:
        data_samples: list of sample dicts
        batch_size: optional batch size (defaults to config.BATCH_SIZE)
        shuffle: whether to shuffle the dataset
        num_workers: optional worker count (defaults to config.NUM_WORKERS)

    Returns:
        DataLoader or None if data_samples is empty
    """
    if not data_samples:
        return None

    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if num_workers is None:
        num_workers = config.NUM_WORKERS

    dataset = HandwritingDataset(data_samples)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
        drop_last=False,
    )


def create_dataloaders(train_data=None, val_data=None, test_data=None, batch_size=None, num_workers=None):
    """
    Create PyTorch DataLoaders for train, validation, and test sets.

    Args:
        train_data, val_data, test_data: lists of sample dicts (or None)
        batch_size: optional batch size (defaults to config.BATCH_SIZE)
        num_workers: optional worker count (defaults to config.NUM_WORKERS)

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_loader = create_dataloader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = create_dataloader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = create_dataloader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

