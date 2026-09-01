"""Arc Virtual Cell Challenge 2026 -- zero-shot CRISPRi response prediction.

The task: given only the *unperturbed* cells of three cell lines the model has
never seen perturbed, plus a list of genes to knock down, predict the
post-perturbation single-cell profiles for all three.

    controls bundle            Bundle.load(...)
      |                          three contexts, 18,533 genes, 300 targets
      v
    a predictor per context    ControlOnlyPredictor  (controls only)
      |                        ContextTransferModel  (+ a signature library)
      v  natural-log fold changes
    counts sampler             build_submission(...)
      |                          400 raw-count cells per target per context
      v
    submission.h5ad  ->  vcc prep  ->  .vcc  ->  vcc submit
"""

from .challenge import Bundle
from .coexpression import ControlOnlyConfig, ControlOnlyPredictor
from .context import CellContext
from .library import SignatureLibrary, add_source_streaming, build_signature_library
from .model import ContextTransferModel, ModelConfig
from .submission import SubmissionConfig, build_submission
from .writer import SparseH5adWriter

__all__ = [
    "Bundle",
    "CellContext",
    "ContextTransferModel",
    "ControlOnlyConfig",
    "ControlOnlyPredictor",
    "ModelConfig",
    "SignatureLibrary",
    "SparseH5adWriter",
    "SubmissionConfig",
    "add_source_streaming",
    "build_signature_library",
    "build_submission",
]
