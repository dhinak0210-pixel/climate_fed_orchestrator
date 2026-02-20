"""
mnist_cnn.py — GreenNet-Mini
═════════════════════════════
Surgical precision architecture optimised for CPU inference efficiency.

Target Specification:
  • Accuracy:    ≥99.2% on standard MNIST test set
  • Parameters:  80,650 (fits in L2 cache on modern CPUs)
  • FLOPs:       ~2.1 MFLOPs per forward pass
  • Memory:      ~342 KB (float32) / ~171 KB (float16)

Architecture:
  Conv(1→16, 3×3) → BN → ReLU → MaxPool(2)
  → Conv(16→32, 3×3) → BN → ReLU → AdaptiveAvgPool(7×7)
  → Flatten → FC(32·7·7 → 48) → Dropout(0.25)
  → FC(48 → 10)

The Adaptive Average Pool ensures correct output size regardless of whether
input is 28×28 (MNIST) or up to 32×32, giving flexibility for future tasks.

Batch Normalisation after each convolution significantly stabilises training
across heterogeneous non-IID node datasets, which would otherwise exhibit
high gradient variance due to distribution shift.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EcoCNN(nn.Module):
    """
    GreenNet-Mini: Efficient CNN for federated learning on MNIST.

    Architecture designed around three principles:
      1. Minimal parameters (< 100K) to reduce communication overhead.
      2. Batch normalisation for convergence stability across non-IID nodes.
      3. Adaptive pooling for input-size agnosticism.

    Args:
        input_channels:  Number of image channels (1 for MNIST grayscale).
        num_classes:     Output classes (10 for MNIST digits).
        conv1_filters:   Filters in first convolutional block.
        conv2_filters:   Filters in second convolutional block.
        fc_hidden:       Hidden size of the fully-connected head.
        dropout:         Dropout probability before output layer.
    """

    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 10,
        conv1_filters: int = 16,
        conv2_filters: int = 32,
        fc_hidden: int = 48,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()

        # ── Feature Extractor ─────────────────────────────────────────────
        self.features = nn.Sequential(
            # Block 1: (B, 1, 28, 28) → (B, 16, 13, 13)
            nn.Conv2d(
                input_channels, conv1_filters, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(conv1_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2: (B, 16, 13, 13) → (B, 32, 11, 11)
            nn.Conv2d(
                conv1_filters, conv2_filters, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(conv2_filters),
            nn.ReLU(inplace=True),
            # Adaptive pool to fixed size for linear layers
            nn.AdaptiveAvgPool2d(output_size=(7, 7)),  # → (B, 32, 7, 7)
        )

        flat_dim = conv2_filters * 7 * 7  # 32 × 7 × 7 = 1568

        # ── Classifier Head ───────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, fc_hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fc_hidden, num_classes, bias=True),
        )

        # ── Weight Initialisation ─────────────────────────────────────────
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Logit tensor of shape (B, num_classes).
        """
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _init_weights(self) -> None:
        """Kaiming Normal for conv layers; Xavier uniform for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    @property
    def num_parameters(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
