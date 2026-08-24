"""Arc Virtual Cell Challenge 2026 -- zero-shot CRISPRi response prediction.

The challenge: given only the *unperturbed* cells of a cell line the model has
never seen perturbed, plus a list of genes to knock down, predict the
post-perturbation single-cell expression profiles.

Pipeline
--------
    sources (h5ad)  --build_signature_library-->  SignatureLibrary
    SignatureLibrary + CellContext(target line controls)
        --ContextTransferModel-->  delta matrix (P x G)
        --CellSampler-->           single cells (AnnData)
        --write_submission-->      submission.h5ad -> `cell-eval prep` -> .vcc
"""

from .context import CellContext
from .library import SignatureLibrary, build_signature_library
from .model import ContextTransferModel, ModelConfig
from .sampler import CellSampler, SamplerConfig
from .submit import write_submission

__all__ = [
    "CellContext",
    "SignatureLibrary",
    "build_signature_library",
    "ContextTransferModel",
    "ModelConfig",
    "CellSampler",
    "SamplerConfig",
    "write_submission",
]
