"""LAMF (Learned Accelerated Mixture Fitter) module.

This module provides a neural network-based GMM fitter that replaces
traditional EM algorithm with learnable iterative refinement.
"""

from .model import (
    LAMFFitter,
    InitNet,
    InitNetV2,
    RefineBlock,
    SinusoidalPositionalEncoding,
    LearnedPositionalEncoding,
)
from .infer import fit_gmm1d_to_pdf_lamf

__all__ = [
    "LAMFFitter",
    "InitNet",
    "InitNetV2",
    "RefineBlock",
    "SinusoidalPositionalEncoding",
    "LearnedPositionalEncoding",
    "fit_gmm1d_to_pdf_lamf",
]

