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

    fc  = exp(lfc)
    w   = w0 * clip(1 - 1/fc, 0, 1)                 # 0 when fc <= 1
    lam = fc * ((1 - w) * counts_i + w * L_i * m)
    active genes:  stochastic_round(lam)
    every other gene passes through from the real control cell, untouched

With no prediction there is nothing to touch and the output *is* the input, so
the null is exact by construction rather than by calibration.  Every DE call the
scorer makes now traces back to a deliberate prediction.

The pieces of the active-gene formula each still do a job:

* ``counts_i`` is the real cell's own count for that gene, so the spread across
  cells stays real;
* ``w`` *mixes* the cell toward the context's population rate instead of
  adding to it.  A gene reading zero in this particular cell can never be
  *up*-regulated by multiplication alone, so some mass has to come from
  somewhere -- but it must come out of the cell's own share, not on top of it.
  Written as an addition (the previous two versions of this file) it delivers
  the wrong mean in whichever direction it is applied: added on both sides it
  diluted knockdowns, so an intended residual of 0.148 arrived at 0.30; applied
  to up-regulation only it over-delivered instead, and the emitted populations
  came out 81-86% up on their significant genes where the measured H1
  signatures are 52%.  Written as a mix, ``E[lam] = fc * L_i * m`` for every
  ``w``, so the requested fold change is delivered in both directions and ``w``
  only decides how much of it reaches cells that read zero.  ``w`` vanishes at
  ``fc = 1``, which is what keeps the null exact;
* the existing count is *transformed*, not re-measured.  A Poisson redraw of
  ``counts_i * fc`` looks natural and is not: the control cell is already one
  noisy measurement, and drawing again adds a second round.  That second round
  is not symmetric -- resampling a sparse count vector turns 1s into 0s about
  37% of the time -- so it shifts genes down in exactly the rank sense the
  scorer's Wilcoxon test reads.  Measured on the real context A controls,
  redrawing all 18,533 genes with a negligible fold change produces 5,815
  spurious significant genes, 99.8% of them called DOWN; even 1,500 redrawn
  genes produce 203.  Stochastic rounding instead preserves the mean exactly and
  adds only the variance integrality demands.  Real knockdown responses are
  mostly down, so this artefact would have *flattered* the direction metrics
  while generalising to nothing;


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
    population = mean_proportions[active][None, :] * library_sizes[:, None]
    fc = np.exp(np.clip(lfc[:, active] if lfc.ndim == 2 else lfc[active][None, :], -30.0, 30.0))
    # Mix toward the population rate, never add to it, and only on the way up.
    # The weight vanishes at fc = 1 so an unchanged gene stays untouched.
    w = pseudocount * np.clip(1.0 - 1.0 / np.maximum(fc, 1e-9), 0.0, 1.0)
    scaled = np.clip(fc * ((1.0 - w) * block + w * population), 0.0, None)
    floor = np.floor(scaled)
    drawn = (floor + (rng.random(scaled.shape) < (scaled - floor))).astype(np.int64)

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
