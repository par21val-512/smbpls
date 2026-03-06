"""
SMBPLS: Sparse Multi-Block Partial Least Squares implemented in PyTorch.

This package provides:
- SMBPLSNet: PyTorch implementation of sparse multi-block PLS
- generate_data: synthetic data generator for experiments
"""

__version__ = "0.1.0"

# Core model
from .models.smbpls_net import SMBPLSNet

# Data utilities
from .data.simulate import generate_data

__all__ = [
    "SMBPLSNet",
    "generate_data",
]