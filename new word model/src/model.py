"""
BiLSTM + CTC model for online handwriting recognition.

Architecture:
  Input Projection → Bidirectional LSTM (3 layers) → Output Projection

Input:  (batch, seq_len, feature_dim)  — stroke feature sequences
Output: (seq_len, batch, num_classes)  — log probabilities for CTC loss
"""

import torch
import torch.nn as nn

from . import config


class HandwritingRecognizer(nn.Module):
    """
    Bidirectional LSTM with CTC output layer for stroke-to-text recognition.
    """

    def __init__(
        self,
        input_dim=None,
        hidden_size=None,
        num_layers=None,
        num_classes=None,
        dropout=None,
        bidirectional=None,
    ):
        super().__init__()

        # Use config defaults if not specified
        input_dim = input_dim or config.INPUT_FEATURE_DIM
        hidden_size = hidden_size or config.HIDDEN_SIZE
        num_layers = num_layers or config.NUM_LSTM_LAYERS
        num_classes = num_classes or config.NUM_CLASSES
        dropout = dropout if dropout is not None else config.DROPOUT
        bidirectional = bidirectional if bidirectional is not None else config.BIDIRECTIONAL

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Input projection: map features to hidden dimension
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Bidirectional LSTM encoder
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Output projection: map LSTM output to character probabilities
        lstm_output_dim = hidden_size * self.num_directions
        self.output_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, num_classes),
        )

    def forward(self, features, seq_lens):
        """
        Forward pass.

        Args:
            features: (batch, max_seq_len, feature_dim) — padded input features
            seq_lens: (batch,) — actual sequence lengths

        Returns:
            log_probs: (max_seq_len, batch, num_classes) — log softmax output
                       Time-first format as required by CTC loss.
        """
        batch_size = features.size(0)

        # Input projection
        x = self.input_proj(features)  # (batch, max_seq_len, hidden_size)

        # Pack padded sequences for efficient LSTM processing
        packed = nn.utils.rnn.pack_padded_sequence(
            x, seq_lens.cpu(), batch_first=True, enforce_sorted=True
        )

        # LSTM forward pass
        packed_output, _ = self.lstm(packed)

        # Unpack back to padded format
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True
        )  # (batch, max_seq_len, hidden_size * num_directions)

        # Output projection
        logits = self.output_proj(output)  # (batch, max_seq_len, num_classes)

        # Log softmax for CTC loss (expects log probabilities)
        log_probs = torch.log_softmax(logits, dim=-1)

        # Transpose to (max_seq_len, batch, num_classes) for CTC loss
        log_probs = log_probs.permute(1, 0, 2)

        return log_probs

    def count_parameters(self):
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        return (
            f"HandwritingRecognizer(\n"
            f"  hidden_size={self.hidden_size},\n"
            f"  num_layers={self.num_layers},\n"
            f"  bidirectional={self.bidirectional},\n"
            f"  parameters={self.count_parameters():,}\n"
            f")"
        )

