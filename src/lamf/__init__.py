"""LAMF (Learned Accelerated Mixture Fitter) module.

This module provides a neural network-based GMM fitter that replaces
traditional EM algorithm with learnable iterative refinement.
"""

from .model import LAMFFitter, InitNet, RefineBlock
from .infer import fit_gmm1d_to_pdf_lamf

__all__ = [
    "LAMFFitter",
    "InitNet", 
    "RefineBlock",
    "fit_gmm1d_to_pdf_lamf",
]

