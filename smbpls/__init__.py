"""
SMBPLS: Sparse Multi-Block Partial Least Squares implemented in PyTorch.

This package provides:
- SMBPLSNet: PyTorch implementation of sparse multi-block PLS
- generate_data: synthetic data generator for experiments
"""

__version__ = "0.1.0"

# Core model
from .models.smbpls_model import SMBPLSNet

# Data utilities
from .data.simulate_sata import generate

# Training
from .train.train_smbpls import train_smbpls

__all__ = [
    "SMBPLSNet",
    "generate",
]
