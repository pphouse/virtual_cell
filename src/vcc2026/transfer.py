"""Moving a perturbation signature from a source cell line into an unseen one.

This is the whole zero-shot problem in one function.  A knockdown of gene *g*
does not produce the same expression *delta* in every context: a gene that is
silent in the target line cannot go down, and a program that is already
saturated cannot go up.  What does travel between contexts is the
*multiplicative* effect -- the fold change.

So the transfer operator converts a source-line normlog delta into a fold
change, applies that fold change to the target line's own baseline, and reads
the result back out as a normlog delta:

    fc_j      = (expm1(mu_src_j + d_j) + eps) / (expm1(mu_src_j) + eps)
    d_tgt_j   = log1p(expm1(mu_tgt_j) * fc_j) - log1p(expm1(mu_tgt_j))

The pseudo-count ``eps`` is what keeps this honest.  Without it, a gene with
near-zero source expression yields an unbounded fold change that then explodes
in a target line where the gene *is* expressed; ``eps`` caps the influence of
exactly those unreliable ratios.  Genes silent in the target collapse to
``d_tgt_j -> 0`` automatically, which is the behaviour we want and the reason
this is preferred over an additive transfer with a hand-tuned expression gate.
"""

from __future__ import annotations

import numpy as np

EPS_PSEUDOCOUNT = 0.5


def fold_change(mu_src: np.ndarray, delta: np.ndarray, eps: float = EPS_PSEUDOCOUNT) -> np.ndarray:
    """Linear-space fold change implied by a normlog delta on a normlog baseline."""
    e0 = np.expm1(np.clip(mu_src, 0.0, None))
    e1 = np.expm1(np.clip(mu_src + delta, 0.0, None))
    return (e1 + eps) / (e0 + eps)


def apply_fold_change(
    mu_tgt: np.ndarray, fc: np.ndarray, eps: float = EPS_PSEUDOCOUNT
) -> np.ndarray:
    """Normlog delta produced by applying `fc` to the target-line baseline."""
    e0 = np.expm1(np.clip(mu_tgt, 0.0, None))
    e1 = np.clip((e0 + eps) * fc - eps, 0.0, None)
    return np.log1p(e1) - np.log1p(e0)


def rebase(
    delta: np.ndarray,
    mu_src: np.ndarray,
    mu_tgt: np.ndarray,
    eps: float = EPS_PSEUDOCOUNT,
    fc_clip: float = 8.0,
) -> np.ndarray:
    """Transfer a normlog delta from the source context to the target context."""
    fc = fold_change(mu_src, delta, eps=eps)
    fc = np.clip(fc, 1.0 / fc_clip, fc_clip)
    return apply_fold_change(mu_tgt, fc, eps=eps)


def line_similarity(mu_a: np.ndarray, mu_b: np.ndarray, top_k: int = 4000) -> float:
    """Similarity between two cell lines, from their unperturbed profiles alone.

    Restricted to the genes with the largest spread between the two profiles --
    the housekeeping bulk is identical everywhere and would push every pair of
    human cell lines to r ~ 0.95, washing out the ranking we actually need.
    """
    diff = np.abs(mu_a - mu_b)
    k = min(top_k, diff.size)
    idx = np.argpartition(-diff, k - 1)[:k]
    a, b = mu_a[idx], mu_b[idx]
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def line_weights(
    mu_tgt: np.ndarray,
    baselines: dict[str, np.ndarray],
    temperature: float = 0.15,
) -> dict[str, float]:
    """Softmax weights over source lines by similarity to the target context."""
    sims = {ln: line_similarity(mu_tgt, mu) for ln, mu in baselines.items()}
    if not sims:
        return {}
    vals = np.array(list(sims.values()))
    logits = (vals - vals.max()) / max(temperature, 1e-6)
    w = np.exp(logits)
    w /= w.sum()
    return dict(zip(sims.keys(), w.tolist(), strict=False))
