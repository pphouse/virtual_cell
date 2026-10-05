"""Count normalisation -- and the one place where getting the *scale* wrong is fatal.

cell-eval scores in normalised-log space, and it produces that space by calling
``scanpy.pp.normalize_total`` with **no** ``target_sum``.  That is not the
familiar CPM-or-1e4 convention: with no target sum, scanpy normalises every
cell to the **median library size of that AnnData**, and only then applies
``log1p``.

So the absolute scale of the reference data is set by the sequencing depth of
the held-out experiment.  A submission written in the conventional
``target_sum=1e4`` normlog sits on a *different* scale from the real data, and
MAE -- the only metric in the panel measured in absolute expression units --
collapses.  Measured on this repository's simulator, that scale mismatch alone
turned an MAE of 0.073 into 0.646, a 9x penalty, with the biology unchanged.

The defence is to normalise to the median library size of the *target line's
own control cells*, which come from the same experiment and therefore the same
depth as the perturbed cells being scored.  That is the default here
(``target_sum=None``).  ``--target-sum`` remains available to sweep on the
validation leaderboard, since that is a cheap way to confirm the estimate.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def library_sizes(x: np.ndarray | sp.spmatrix) -> np.ndarray:
    return np.asarray(x.sum(axis=1)).ravel()


def median_library_size(x: np.ndarray | sp.spmatrix) -> float:
    totals = library_sizes(x)
    totals = totals[totals > 0]
    return float(np.median(totals)) if totals.size else 1.0


def is_discrete(x: np.ndarray | sp.spmatrix, n_probe: int = 10000) -> bool:
    """Cheap heuristic: are these raw integer counts rather than normlog values?"""
    vals = x.data[:n_probe] if sp.issparse(x) else np.asarray(x).ravel()[:n_probe]
    if vals.size == 0:
        return True
    return bool(np.allclose(vals, np.round(vals)))


def normlog(
    x: np.ndarray | sp.spmatrix, target_sum: float | None = None
) -> tuple[np.ndarray, float]:
    """Normalise then log1p.  Returns ``(dense normlog matrix, target sum used)``.

    ``target_sum=None`` reproduces ``scanpy.pp.normalize_total``'s default --
    the median library size -- which is what cell-eval applies to the reference
    data.
    """
    if sp.issparse(x):
        x = x.toarray()
    x = np.asarray(x, dtype=np.float64)
    if target_sum is None:
        target_sum = median_library_size(x)
    totals = x.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return np.log1p(target_sum * x / totals), float(target_sum)


def as_normlog(x: np.ndarray | sp.spmatrix, target_sum: float | None = None) -> np.ndarray:
    """Normlog `x` if it looks like counts, otherwise densify and pass through."""
    if is_discrete(x):
        return normlog(x, target_sum=target_sum)[0]
    if sp.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float64)


def renormalise_rows(x: np.ndarray, target_sum: float) -> np.ndarray:
    """Project normlog rows back onto the normlog simplex for `target_sum`.

    Adding a delta to a normlog vector breaks the constraint that
    ``sum(expm1(x)) == target_sum``, which real profiles satisfy by
    construction.  Off by default in the sampler: enforcing it perturbs the
    pseudobulk mean, and preserving the mean exactly is worth more across the
    metric panel than closing a small library-size gap.
    """
    lin = np.expm1(np.clip(x, 0.0, None))
    totals = lin.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return np.log1p(target_sum * lin / totals)
