"""A fast local stand-in for cell-eval, for iterating on a held-out cell line.

`cell-eval run` is the ground truth, but it re-runs a full differential
expression test per perturbation, which is far too slow for a calibration
sweep.  Every expression-space metric in cell-eval is computed on
*per-perturbation pseudobulk means*, so those can be reproduced here exactly
and cheaply.  The DE-family metrics cannot -- what is provided is an explicitly
labelled rank-based proxy, useful for ordering candidate configurations and not
for reporting a number.

The scoring aggregation follows the 2026 convention: each metric is expressed
relative to the "predict the context mean" baseline,

    lower-is-better:   1 - user / base
    higher-is-better:  (user - base) / (1 - base)

so 0 means "no better than pasting the average cell onto every perturbation".

**There is no floor at zero.**  That was true of the 2025 scorer and is not
true of the 2026 one: only the expression-accuracy metric stops at 0, and the
rest bottom out at their own depths (the DE log-FC metric is floored at -6, the
direction-fidelity metric around -1.9).  A confidently wrong prediction is
therefore worse than no prediction, and the context mean -- which scores exactly
0 -- is always available as the honest abstention.  Only `mse`-family metrics
are clipped here, matching that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LOWER_IS_BETTER = {"mae", "mse", "mae_delta", "mse_delta"}
# The only metric family the real scorer floors at zero (expression accuracy).
FLOORED_AT_ZERO = {"mse", "mse_delta"}


@dataclass
class BulkPair:
    """Aligned pseudobulk means for one cell line."""

    perts: list[str]
    genes: np.ndarray
    pred: np.ndarray  # (P, G)
    real: np.ndarray  # (P, G)
    pred_ctrl: np.ndarray  # (G,)
    real_ctrl: np.ndarray  # (G,)

    @property
    def pred_delta(self) -> np.ndarray:
        return self.pred - self.pred_ctrl

    @property
    def real_delta(self) -> np.ndarray:
        return self.real - self.real_ctrl


def mae(p: BulkPair) -> float:
    return float(np.abs(p.pred - p.real).mean())


def mse(p: BulkPair) -> float:
    return float(((p.pred - p.real) ** 2).mean())


def mae_delta(p: BulkPair) -> float:
    return float(np.abs(p.pred_delta - p.real_delta).mean())


def pearson_delta(p: BulkPair) -> float:
    a, b = p.pred_delta, p.real_delta
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(den > 0, num / den, 0.0)
    return float(np.nan_to_num(r).mean())


def discrimination_score_l1(p: BulkPair, exclude_target_gene: bool = True) -> float:
    """cell-eval's PDS: is each predicted effect closest to its *own* real effect?

    Note the target-gene exclusion -- getting the on-target knockdown right
    contributes nothing here, so this metric is entirely a test of the trans
    signature, and (because it ranks by L1 distance over ~18k genes) it is
    dominated by whether the *magnitude* of each effect is right.
    """
    real, pred = p.real_delta, p.pred_delta
    n = len(p.perts)
    gene_pos = {g: i for i, g in enumerate(np.asarray(p.genes, dtype=object))}
    scores = np.zeros(n)
    for i, pert in enumerate(p.perts):
        keep = np.ones(real.shape[1], dtype=bool)
        if exclude_target_gene:
            j = gene_pos.get(pert)
            if j is not None:
                keep[j] = False
        d = np.abs(real[:, keep] - pred[i, keep][None, :]).sum(axis=1)
        rank = int(np.flatnonzero(np.argsort(d) == i)[0])
        scores[i] = 1.0 - rank / n
    return float(scores.mean())


def overlap_at_n_proxy(p: BulkPair, n_top: int = 100) -> float:
    """Rank-based stand-in for cell-eval's `overlap_at_N` DE-set overlap.

    Real `overlap_at_N` ranks genes by DE significance and sets N from the
    number of genes that are actually significant.  Here both sides are ranked
    by absolute pseudobulk delta with a fixed N.  It tracks the real metric
    well enough to order configurations and should never be quoted as a score.
    """
    real, pred = p.real_delta, p.pred_delta
    k = int(min(n_top, real.shape[1]))
    out = np.zeros(len(p.perts))
    for i in range(len(p.perts)):
        r = set(np.argpartition(-np.abs(real[i]), k - 1)[:k].tolist())
        q = set(np.argpartition(-np.abs(pred[i]), k - 1)[:k].tolist())
        out[i] = len(r & q) / k
    return float(out.mean())


def direction_match(p: BulkPair, n_top: int = 100) -> float:
    """Fraction of the real top-N DE genes whose predicted sign is correct."""
    real, pred = p.real_delta, p.pred_delta
    k = int(min(n_top, real.shape[1]))
    out = np.zeros(len(p.perts))
    for i in range(len(p.perts)):
        idx = np.argpartition(-np.abs(real[i]), k - 1)[:k]
        out[i] = float((np.sign(real[i][idx]) == np.sign(pred[i][idx])).mean())
    return float(out.mean())


PANEL = {
    "mae": mae,
    "mae_delta": mae_delta,
    "pearson_delta": pearson_delta,
    "discrimination_score_l1": discrimination_score_l1,
    "overlap_at_N_proxy": overlap_at_n_proxy,
    "de_direction_match_proxy": direction_match,
}


def evaluate(p: BulkPair, panel: dict | None = None) -> dict[str, float]:
    panel = panel or PANEL
    return {name: float(fn(p)) for name, fn in panel.items()}


def baseline_pair(p: BulkPair) -> BulkPair:
    """The cell-eval baseline: predict the control mean for every perturbation."""
    return BulkPair(
        perts=p.perts,
        genes=p.genes,
        pred=np.repeat(p.real_ctrl[None, :], len(p.perts), axis=0),
        real=p.real,
        pred_ctrl=p.real_ctrl,
        real_ctrl=p.real_ctrl,
    )


def score_against_baseline(user: dict[str, float], base: dict[str, float]) -> dict[str, float]:
    """Reproduce cell-eval's baseline normalisation and overall average."""
    out: dict[str, float] = {}
    for name, u in user.items():
        b = base.get(name)
        if b is None:
            continue
        if name in LOWER_IS_BETTER:
            s = 1.0 - (u / b) if b != 0 else 0.0
        else:
            s = (u - b) / (1.0 - b) if b != 1.0 else 0.0
        s = float(np.nan_to_num(s))
        out[name] = max(s, 0.0) if name in FLOORED_AT_ZERO else s
    out["avg_score"] = float(np.mean(list(out.values()))) if out else 0.0
    return out


def bulk_pair_from_predictions(
    targets: list[str],
    genes: np.ndarray,
    pred_delta: np.ndarray,
    pred_ctrl: np.ndarray,
    real_means: dict[str, np.ndarray],
    real_ctrl: np.ndarray,
) -> BulkPair:
    """Assemble a `BulkPair` from model output and measured held-out means."""
    keep = [t for t in targets if t in real_means]
    if not keep:
        raise ValueError("no overlap between predicted targets and measured means")
    rows = [targets.index(t) for t in keep]
    return BulkPair(
        perts=keep,
        genes=np.asarray(genes, dtype=object),
        pred=pred_ctrl[None, :] + pred_delta[rows],
        real=np.vstack([real_means[t] for t in keep]),
        pred_ctrl=pred_ctrl,
        real_ctrl=real_ctrl,
    )
