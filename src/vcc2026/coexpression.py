"""What you can predict from control cells alone.

The 2026 task hands you 18,400 unperturbed cells per context and nothing else
about that context.  Before any transfer library exists, two things are already
predictable from those cells:

**The on-target knockdown.**  CRISPRi drops the gene it targets.  That is the
single most certain fact about the experiment, it needs no training data, and
in real data the target gene is usually the most significant DE hit -- so it
carries real weight in four of the six metrics.  It carries none in the fifth:
a discrimination score computed on effect vectors is dominated by the trans
signature, and if every prediction differs only in one coordinate the
perturbations are nearly indistinguishable.

The knockdown residual is not a guess: on the 150 measured knockdowns in the
2025 public screen the target gene falls to a median 0.148 of its baseline, with
every single one of the 150 going down (`scripts/validate_on_2025.py`).

**A first-order trans signature -- which turned out not to work.**  The
plausible idea is that genes co-varying with the target across the control
population are the ones a knockdown drags with it: same complex, same pathway,
same programme.  Measured against those same 150 knockdowns, it carries no
signal at all -- the correlation between the predicted and measured trans
response averages -0.003, and only 44.7% of perturbations come out positive,
which is below chance.  A leave-one-out test of the stronger version of the same
idea (predict a held-out signature from the co-expression-weighted average of the
others) does no better: r=0.165 against r=0.188 for simply predicting the mean
of all other signatures, and a discrimination score of 0.509 against 0.500 for
chance.

So `trans_beta` defaults to 0 and should stay there.  This is not caution about
an unmeasured prior; it is a measurement.  And it matters more than it would
have in 2025: the 2026 aggregate has no floor at zero, so a trans signature made
of noise scores *below* the context-mean baseline rather than merely failing to
beat it.  Predicting the context mean always scores exactly 0, which makes it the
honest fallback wherever there is no signal.

The same harness shows where the signal actually is.  An oracle similarity built
from the true signatures reaches r=0.471 and discrimination 0.748, so the
model's *shape* is fine -- a similarity-weighted average of measured signatures
can work well. The bottleneck is the gene representation, and a functional
interaction network is a real if modest improvement on co-expression: STRING
neighbourhood similarity reaches r=0.235 and discrimination 0.584 on the same
leave-one-out test (`vcc2026/network.py`, `scripts/validate_knn_2025.py`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from sklearn.utils.extmath import randomized_svd

logger = logging.getLogger(__name__)


@dataclass
class ControlOnlyConfig:
    knockdown_residual: float = 0.15  # measured on the 2025 screen; see docs/05
    n_components: int = 50
    trans_beta: float = 0.0  # 0 disables the trans term entirely
    min_baseline: float = 0.02  # normlog mean below this: gene is off here
    max_abs_lfc: float = 4.0
    seed: int = 0


class ControlOnlyPredictor:
    """Log-fold-change predictions from one context's control cells."""

    def __init__(self, config: ControlOnlyConfig | None = None) -> None:
        self.config = config or ControlOnlyConfig()
        self.genes: np.ndarray | None = None
        self.mu: np.ndarray | None = None  # normlog gene means
        self._embedding: np.ndarray | None = None  # (G, K) gene coordinates
        self._norms: np.ndarray | None = None

    def fit(self, counts: sp.csr_matrix, genes: np.ndarray) -> ControlOnlyPredictor:
        cfg = self.config
        self.genes = np.asarray(genes, dtype=object)
        self._index = {g: i for i, g in enumerate(self.genes)}

        x = _normlog_dense(counts)
        self.mu = x.mean(axis=0)
        x -= self.mu
        sd = x.std(axis=0)
        sd[sd == 0] = 1.0
        x /= sd

        k = int(min(cfg.n_components, min(x.shape) - 1))
        _u, s, vt = randomized_svd(x, n_components=k, random_state=cfg.seed)
        # Gene coordinates in program space; scaling rows by the singular values
        # makes the cosine below a correlation restricted to the top-k subspace.
        emb = (vt * s[:, None]).T
        self._embedding = emb
        norms = np.linalg.norm(emb, axis=1)
        norms[norms == 0] = 1.0
        self._norms = norms
        logger.info(
            "control-only fit: %d cells, %d genes, %d components, %.1f%% variance kept",
            x.shape[0],
            x.shape[1],
            k,
            100 * (s**2).sum() / max((x**2).sum(), 1e-9),
        )
        return self

    def program_correlation(self, target: str) -> np.ndarray:
        """Denoised correlation of `target` with every gene, in program space."""
        if self._embedding is None:
            raise RuntimeError("call fit() first")
        i = self._index.get(target)
        if i is None:
            return np.zeros(self.genes.size)
        v = self._embedding[i]
        return (self._embedding @ v) / (self._norms * self._norms[i])

    def predict_lfc(self, targets: list[str]) -> np.ndarray:
        """(P, G) natural-log fold changes, ready for the counts sampler."""
        if self.mu is None:
            raise RuntimeError("call fit() first")
        cfg = self.config
        expressed = self.mu > cfg.min_baseline
        out = np.zeros((len(targets), self.genes.size), dtype=np.float32)

        for p, target in enumerate(targets):
            i = self._index.get(target)
            if i is None or not expressed[i]:
                # Cannot knock down what this context does not express.  The
                # context mean is the honest prediction, and it scores 0 rather
                # than negative.
                continue
            on_target = float(np.log(max(cfg.knockdown_residual, 1e-6)))
            row = out[p]
            if cfg.trans_beta > 0:
                corr = self.program_correlation(target)
                corr[i] = 0.0
                row += (cfg.trans_beta * on_target * corr).astype(np.float32)
                row[~expressed] = 0.0
            row[i] = on_target

        np.clip(out, -cfg.max_abs_lfc, cfg.max_abs_lfc, out=out)
        return out


def _normlog_dense(counts: sp.csr_matrix, target_sum: float | None = None) -> np.ndarray:
    totals = np.asarray(counts.sum(axis=1)).ravel()
    totals[totals == 0] = 1.0
    if target_sum is None:
        target_sum = float(np.median(totals))
    scaled = sp.diags(target_sum / totals) @ counts
    dense = np.asarray(scaled.todense(), dtype=np.float32)
    np.log1p(dense, out=dense)
    return dense
