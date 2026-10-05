#!/usr/bin/env python
"""Fit the two constants `vcc2026.budget` stands on, from a real reference.

The first is the fold change at which a gene of a given control expression gets
called: the emission has to clear it for a call to happen at all, and clearing
it by too much is what `de_wilcoxon_lfc_nmae` charges for.  The second is the
propensity -- how often a gene is called for *some* knockdown -- which is what
ranks the call set.  Both are read off a reference DE table and the control
cells that produced it, so both can be refitted on any screen cut to the
competition's shape rather than trusted from this repository.
"""

from __future__ import annotations

import argparse

import numpy as np
import polars as pl
from scipy.stats import spearmanr


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--de", required=True, help="reference DE table (parquet)")
    p.add_argument("--control-stats", required=True, help="npz from control statistics")
    args = p.parse_args()

    de = pl.read_parquet(args.de)
    perts = np.array(sorted(de["target"].unique().to_list()))
    genes = np.array(sorted(de["feature"].unique().to_list()))
    pi = {t: i for i, t in enumerate(perts)}
    gi = {g: i for i, g in enumerate(genes)}
    lfc = np.zeros((perts.size, genes.size))
    padj = np.ones((perts.size, genes.size))
    rows = np.array([pi[t] for t in de["target"].to_numpy()])
    cols = np.array([gi[g] for g in de["feature"].to_numpy()])
    lfc[rows, cols] = de["log2_fold_change"].to_numpy()
    padj[rows, cols] = de["p_adj"].to_numpy()
    sig = padj < 0.05

    stats = np.load(args.control_stats, allow_pickle=True)
    index = {g: i for i, g in enumerate(stats["genes"])}
    sel = np.array([index[g] for g in genes])
    mean, var = stats["mean"][sel], stats["var"][sel]
    fano = var / np.maximum(mean, 1e-9)

    # THRESHOLD: within an expression band, the |log2 fold change| at which half
    # the (perturbation, gene) pairs come back significant.
    xs, ys = [], []
    edges = np.geomspace(max(mean.min(), 5.0), np.percentile(mean, 99.5), 9)
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        band = (mean >= lo) & (mean < hi)
        if band.sum() < 50:
            continue
        a, s = np.abs(lfc[:, band]).ravel(), sig[:, band].ravel()
        order = np.argsort(a)
        window = max(2001, order.size // 40 | 1)
        run = np.convolve(s[order].astype(float), np.ones(window) / window, mode="same")
        crossed = np.flatnonzero(run >= 0.5)
        if crossed.size:
            xs.append(np.log10(np.sqrt(lo * hi)))
            ys.append(np.log10(a[order][crossed[0]]))
    slope, intercept = np.polyfit(xs, ys, 1)
    print(f"THRESHOLD_SCALE     = {10 ** intercept:.3f}")
    print(f"THRESHOLD_EXPONENT  = {slope:.3f}   (over {len(xs)} expression bands)")

    # PROPENSITY: least squares on the control statistics alone.
    lm = np.log10(mean + 1e-3)
    lf = np.log10(fano + 1e-9)
    design = np.c_[np.ones(genes.size), lm, lf, lm * lf]
    truth = sig.mean(0)
    coef = np.linalg.lstsq(design, truth, rcond=None)[0]
    print("PROPENSITY_COEF     = " + repr(tuple(round(float(c), 6) for c in coef)))
    rho = spearmanr(design @ coef, truth).statistic
    print(f"  Spearman against the measured propensity: {rho:.3f}")


if __name__ == "__main__":
    main()
