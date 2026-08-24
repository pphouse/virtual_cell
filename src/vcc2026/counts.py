"""Emitting raw counts, which is what the 2026 scorer actually reads.

The submission is a raw integer count matrix -- scoring runs in counts space
and rejects a fractional one outright.  So the model may think in normlog (the
right space for a multiplicative perturbation effect), but the last step has to
put a plausible *count* vector on every one of the 360,000 cells.

**The emission must be the identity when the model predicts nothing.**  That
requirement is not obvious and it cost a submission to learn.  The first version
resampled every gene of every cell -- start from a real control cell, smooth it
toward the context mean, apply the fold change, redraw Poisson counts -- and
calibrated the smoothing so the emitted cells matched the real ones in per-gene
variance and detection rate.  Both matched to within 1%.  It was still wrong:
the Wilcoxon test the scorer runs reads the whole distribution, not two moments
of it.  Measured on the real context A controls, with the fold change set to
zero and the emitted cells tested against held-out real cells
(`scripts/null_emission_test.py`):

    real cells            0 of 11,658 genes called significant
    full resample, a=1.0  93 significant, 89% of them called UP
    full resample, a=0.0  5,705 significant, 99.8% of them called DOWN

93 confidently-wrong DE genes, in a consistent direction, handed to the scorer
for all 300 perturbations at once.  Real knockdown responses are mostly *down*,
so that artefact alone drove DE direction fidelity to 0.41 against a baseline of
about 0.51 -- below chance -- and cost -0.34 on that metric.

So the emission only touches genes the model actually predicts a change for:

    fc = exp(lfc)
    active genes:  lam = counts_i * fc + a * L_i * m * max(fc - 1, 0)
    draw ~ Poisson(lam)
    every other gene passes through from the real control cell, untouched

With no prediction there is nothing to touch and the output *is* the input, so
the null is exact by construction rather than by calibration.  Every DE call the
scorer makes now traces back to a deliberate prediction.

The pieces of the active-gene formula each still do a job:

* ``counts_i`` is the real cell's own count for that gene, so the spread across
  cells stays real;
* ``a * L_i * m * (fc - 1)+`` is a pseudo-count toward the context mean,
  written as a fraction of the cell's library size.  Without it a gene reading
  zero in this particular cell could never be *up*-regulated -- multiplying zero
  by a fold change leaves zero -- and the up half of every signature would
  vanish.  It applies to *up*-regulation only, and that asymmetry is not a
  detail: added on both sides it inflates the pre-multiplication mass, so an
  intended knockdown to 0.148 of baseline is delivered at 0.30 instead.  The
  first submission shipped with exactly that bug, halving every knockdown it
  claimed to make;
* the Poisson draw supplies the shot noise a real measurement has.

The library size is deliberately *not* renormalised afterwards.  Knocking down
one gene removes its mass, and on this panel the target genes carry 0.002-0.02%
of the transcriptome, so redistributing that across 18,533 genes would add a
systematic shift far larger than the thing it corrects.
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
    resample_all: bool = False,
) -> sp.csr_matrix:
    """Draw predicted count vectors for a block of cells.

    `log_fold_change` is (n_cells, n_genes) or (n_genes,) -- the natural-log
    fold change to apply per cell, already scaled by that cell's response.

    Only genes with a non-zero fold change are redrawn; the rest pass through
    from `base` exactly.  `resample_all=True` restores the original behaviour
    and exists so `scripts/null_emission_test.py` can measure what it costs.
    """
    n_cells, n_genes = base.shape
    if library_sizes is None:
        library_sizes = np.asarray(base.sum(axis=1)).ravel()
    library_sizes = np.clip(library_sizes, 1.0, max_counts_per_cell)

    lfc = np.asarray(log_fold_change)
    if resample_all:
        return _emit_dense(base, mean_proportions, lfc, pseudocount, rng, library_sizes)

    active = np.flatnonzero(lfc != 0) if lfc.ndim == 1 else np.flatnonzero((lfc != 0).any(axis=0))
    if active.size == 0:
        out = base.copy()
        out.sort_indices()
        return out

    block = np.asarray(base[:, active].todense(), dtype=np.float64)
    prior = pseudocount * mean_proportions[active][None, :] * library_sizes[:, None]
    fc = np.exp(np.clip(lfc[:, active] if lfc.ndim == 2 else lfc[active][None, :], -30.0, 30.0))
    # The prior only opens the way *up*. Applied symmetrically it would dilute
    # every knockdown by the mass it adds before the multiplication.
    lam = block * fc + prior * np.clip(fc - 1.0, 0.0, None)
    drawn = rng.poisson(np.clip(lam, 0.0, None))

    touched = np.zeros(n_genes, dtype=bool)
    touched[active] = True
    rows_i: list[np.ndarray] = []
    rows_d: list[np.ndarray] = []
    indptr = np.zeros(n_cells + 1, dtype=np.int64)
    for i in range(n_cells):
        lo, hi = base.indptr[i], base.indptr[i + 1]
        idx, dat = base.indices[lo:hi], base.data[lo:hi]
        keep = ~touched[idx]
        nz = drawn[i] > 0
        row_i = np.concatenate([idx[keep], active[nz]]).astype(np.int32)
        row_d = np.concatenate([dat[keep], drawn[i][nz]]).astype(np.float32)
        rows_i.append(row_i)
        rows_d.append(row_d)
        indptr[i + 1] = indptr[i] + row_i.size

    out = sp.csr_matrix(
        (np.concatenate(rows_d), np.concatenate(rows_i), indptr), shape=(n_cells, n_genes)
    )
    out.sort_indices()
    return out


def _emit_dense(
    base: sp.csr_matrix,
    mean_proportions: np.ndarray,
    lfc: np.ndarray,
    pseudocount: float,
    rng: np.random.Generator,
    library_sizes: np.ndarray,
) -> sp.csr_matrix:
    """The original whole-cell resample, kept only so its cost stays measurable."""
    n_cells, n_genes = base.shape
    fc = np.exp(np.clip(lfc, -30.0, 30.0))
    prior_unit = pseudocount * mean_proportions
    rows_data, rows_idx = [], []
    indptr = np.zeros(n_cells + 1, dtype=np.int64)
    dense = np.empty(n_genes, dtype=np.float64)
    for i in range(n_cells):
        np.multiply(prior_unit, library_sizes[i], out=dense)
        lo, hi = base.indptr[i], base.indptr[i + 1]
        np.add.at(dense, base.indices[lo:hi], base.data[lo:hi])
        dense *= fc[i] if fc.ndim == 2 else fc
        total = dense.sum()
        if total <= 0:
            dense[:] = mean_proportions
            total = dense.sum()
        drawn = rng.poisson(dense * (library_sizes[i] / total))
        nz = np.flatnonzero(drawn)
        rows_idx.append(nz.astype(np.int32))
        rows_data.append(drawn[nz].astype(np.float32))
        indptr[i + 1] = indptr[i] + nz.size
    return sp.csr_matrix(
        (np.concatenate(rows_data), np.concatenate(rows_idx), indptr), shape=(n_cells, n_genes)
    )


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
