"""Emitting raw counts, which is what the 2026 scorer actually reads.

The submission is a raw integer count matrix -- scoring runs in counts space
and rejects a fractional one outright.  So the model may think in normlog (the
right space for a multiplicative perturbation effect), but the last step has to
put a plausible *count* vector on every one of the 360,000 cells.

The construction here starts from a real control cell and keeps it:

    p_i  = normalise( (counts_i + a * L_i * m) * exp(r_i * delta) )
    out  ~ Poisson(L_i * p_i)

Each piece is doing a specific job.

* ``counts_i`` is a real control cell from the same context, so the cell-to-cell
  biological spread the DE test sees is the real spread, not something invented.
* ``a * L_i * m`` is a pseudo-count pulling toward the context mean profile,
  written as a fraction of the cell's own library size so that ``a = 1`` means
  "the prior carries as much mass as the cell".  Without it a gene that reads
  zero in this particular cell can never be *up*-regulated -- multiplying zero
  by a fold change leaves zero -- and the up half of every predicted signature
  would silently vanish.  ``a`` is the one free parameter, and it is calibrated
  against the real controls rather than guessed.
* ``L_i`` is the real cell's own library size, so the emitted depth distribution
  matches the reference data.
* the Poisson draw supplies the shot noise a real measurement has.

``a`` also controls dispersion, and that is the reason it has to be fitted
rather than set to zero.  The real control cell already carries one round of
measurement noise; drawing Poisson counts on top of it adds a second, so an
unsmoothed emission is over-dispersed (measured here: 1.40x the real per-gene
variance) and detects too few genes.  Shrinking toward the context mean removes
exactly that excess.  Over-dispersion is not a cosmetic problem -- the DE test
that four of the six metrics run reads the within-perturbation spread directly.

Calibrating ``a``: run this with ``delta = 0`` on held-out control cells and
compare the emitted cells to real ones.  Too small an ``a`` and the output is a
noisy copy of single cells (over-dispersed, too few detected genes); too large
and every cell collapses toward the mean (under-dispersed, DE calls everything
significant).  `calibrate_pseudocount` picks the value that matches the real
per-gene variance.
"""

from __future__ import annotations

import logging

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)


def context_mean_proportions(counts: sp.csr_matrix) -> np.ndarray:
    """Mean expression proportion per gene over a pool of control cells."""
    totals = np.asarray(counts.sum(axis=1)).ravel()
    totals[totals == 0] = 1.0
    scaled = sp.diags(1.0 / totals) @ counts
    m = np.asarray(scaled.mean(axis=0)).ravel()
    s = m.sum()
    return m / s if s > 0 else m


def emit_counts(
    base: sp.csr_matrix,
    mean_proportions: np.ndarray,
    log_fold_change: np.ndarray,
    pseudocount: float,
    rng: np.random.Generator,
    library_sizes: np.ndarray | None = None,
    max_counts_per_cell: int = 1_000_000,
) -> sp.csr_matrix:
    """Draw predicted count vectors for a block of cells.

    `log_fold_change` is (n_cells, n_genes) or (n_genes,) -- the natural-log
    fold change to apply per cell, already scaled by that cell's response.
    """
    n_cells, n_genes = base.shape
    if library_sizes is None:
        library_sizes = np.asarray(base.sum(axis=1)).ravel()
    library_sizes = np.clip(library_sizes, 1.0, max_counts_per_cell)

    fc = np.exp(np.clip(log_fold_change, -30.0, 30.0))
    prior_unit = pseudocount * mean_proportions

    rows_data: list[np.ndarray] = []
    rows_idx: list[np.ndarray] = []
    indptr = np.zeros(n_cells + 1, dtype=np.int64)

    # Row-at-a-time keeps peak memory at one dense gene vector rather than a
    # dense (cells x genes) block, which at this panel size does not fit.
    dense = np.empty(n_genes, dtype=np.float64)
    for i in range(n_cells):
        np.multiply(prior_unit, library_sizes[i], out=dense)
        lo, hi = base.indptr[i], base.indptr[i + 1]
        np.add.at(dense, base.indices[lo:hi], base.data[lo:hi])
        row_fc = fc[i] if fc.ndim == 2 else fc
        dense *= row_fc
        total = dense.sum()
        if total <= 0:
            dense[:] = mean_proportions
            total = dense.sum()
        lam = dense * (library_sizes[i] / total)
        drawn = rng.poisson(lam)
        nz = np.flatnonzero(drawn)
        rows_idx.append(nz.astype(np.int32))
        rows_data.append(drawn[nz].astype(np.float32))
        indptr[i + 1] = indptr[i] + nz.size

    return sp.csr_matrix(
        (
            np.concatenate(rows_data) if rows_data else np.zeros(0, np.float32),
            np.concatenate(rows_idx) if rows_idx else np.zeros(0, np.int32),
            indptr,
        ),
        shape=(n_cells, n_genes),
    )


def calibrate_pseudocount(
    controls: sp.csr_matrix,
    candidates: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0),
    n_probe: int = 2000,
    seed: int = 0,
) -> tuple[float, list[dict]]:
    """Pick the pseudo-count whose null emission best matches the real controls.

    Runs the emission model with no perturbation on a held-out half of the
    control pool and scores each candidate on how closely the emitted cells
    reproduce the real per-gene variance and detection rate.  This is a
    self-consistency check on real data -- no held-out response is involved.
    """
    rng = np.random.default_rng(seed)
    n = controls.shape[0]
    perm = rng.permutation(n)
    probe = min(n_probe, n // 2)
    fit_idx, eval_idx = perm[:probe], perm[probe : 2 * probe]

    fit = controls[fit_idx]
    real = controls[eval_idx]
    mean_p = context_mean_proportions(controls)

    real_nnz = np.diff(real.indptr).mean()
    real_var = _gene_log_variance(real)
    real_mean = _gene_log_mean(real)

    results = []
    for a in candidates:
        emitted = emit_counts(
            fit,
            mean_proportions=mean_p,
            log_fold_change=np.zeros(controls.shape[1]),
            pseudocount=a,
            rng=np.random.default_rng(seed + 1),
        )
        var = _gene_log_variance(emitted)
        mean = _gene_log_mean(emitted)
        keep = (real_mean > 0.05) | (mean > 0.05)
        results.append(
            {
                "pseudocount": a,
                "nnz_ratio": float(np.diff(emitted.indptr).mean() / max(real_nnz, 1)),
                "var_ratio": float(var[keep].mean() / max(real_var[keep].mean(), 1e-9)),
                "mean_abs_dev": float(np.abs(mean[keep] - real_mean[keep]).mean()),
            }
        )
        logger.info("pseudocount %-5s -> %s", a, results[-1])

    # Match the dispersion first: it is what the DE test reads.  Break ties on
    # the detection rate, which the density cap and the DE filter both care about.
    best = min(
        results,
        key=lambda r: (
            abs(np.log(max(r["var_ratio"], 1e-9))),
            abs(np.log(max(r["nnz_ratio"], 1e-9))),
        ),
    )
    return float(best["pseudocount"]), results


def _normlog(counts: sp.csr_matrix, target_sum: float | None = None) -> sp.csr_matrix:
    totals = np.asarray(counts.sum(axis=1)).ravel()
    totals[totals == 0] = 1.0
    if target_sum is None:
        target_sum = float(np.median(totals))
    out = sp.diags(target_sum / totals) @ counts
    out = out.tocsr()
    out.data = np.log1p(out.data)
    return out


def _gene_log_mean(counts: sp.csr_matrix) -> np.ndarray:
    return np.asarray(_normlog(counts).mean(axis=0)).ravel()


def _gene_log_variance(counts: sp.csr_matrix) -> np.ndarray:
    x = _normlog(counts)
    mean = np.asarray(x.mean(axis=0)).ravel()
    sq = np.asarray(x.multiply(x).mean(axis=0)).ravel()
    return np.clip(sq - mean**2, 0.0, None)
