"""Low-rank response programs.

Knockdowns do not produce 300 unrelated deltas; they produce a handful of
shared stress/growth programmes (ribosome biogenesis, proteasome load, cell
cycle arrest, the integrated stress response) mixed in different proportions,
plus a small gene-specific residual.  Factorising the library's delta matrix
recovers that basis, and it buys two things:

* **denoising** -- a signature measured on 40 cells is mostly noise off the
  program manifold, and projecting onto K components strips it;
* **extrapolation** -- an unseen target gene has no signature of its own, but
  its *loadings* can be predicted from gene features, and the basis turns those
  K numbers back into a full transcriptome-wide delta.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ProgramBasis:
    """Truncated SVD basis over consensus perturbation deltas."""

    targets: list[str]
    components: np.ndarray  # (K, G) orthonormal rows
    loadings: np.ndarray  # (P, K) loadings of the training targets

    @classmethod
    def fit(cls, targets: list[str], deltas: np.ndarray, n_components: int = 30) -> ProgramBasis:
        if deltas.shape[0] == 0:
            raise ValueError("cannot fit a program basis on zero signatures")
        k = int(min(n_components, deltas.shape[0], deltas.shape[1]))
        # Economy SVD on (P x G); P is a few hundred so this is cheap even at G=18k.
        _u, _s, vt = np.linalg.svd(deltas, full_matrices=False)
        comps = vt[:k]
        return cls(targets=list(targets), components=comps, loadings=deltas @ comps.T)

    @property
    def n_components(self) -> int:
        return self.components.shape[0]

    def project(self, delta: np.ndarray) -> np.ndarray:
        return delta @ self.components.T

    def reconstruct(self, loadings: np.ndarray) -> np.ndarray:
        return loadings @ self.components

    def denoise(self, delta: np.ndarray) -> np.ndarray:
        return self.reconstruct(self.project(delta))

    def predict_loadings(self, weights: np.ndarray) -> np.ndarray:
        """Weighted average of training loadings -- the kNN step for unseen targets."""
        total = weights.sum()
        if total <= 0:
            return np.zeros(self.n_components)
        return (weights @ self.loadings) / total


def neighbour_weights(
    similarity: np.ndarray,
    k: int = 25,
    power: float = 3.0,
    min_similarity: float = 0.0,
) -> np.ndarray:
    """Sparse, sharpened kNN weights from a similarity vector.

    The power sharpens the kernel: gene-gene cosine similarities live in a
    narrow band, so a plain weighted average over the top-k drifts towards the
    global mean signature -- which is exactly the failure mode that flattens the
    discrimination score (every perturbation predicted to look the same ranks
    at chance).
    """
    sim = np.clip(np.nan_to_num(similarity, nan=0.0), min_similarity, None)
    if sim.size == 0 or not np.any(sim > 0):
        return np.zeros_like(sim)
    k = int(min(k, sim.size))
    cutoff = np.partition(sim, -k)[-k]
    w = np.where(sim >= cutoff, sim, 0.0)
    return w**power
