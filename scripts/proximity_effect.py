#!/usr/bin/env python
"""Measure how far a CRISPRi knockdown reaches along the chromosome.

dCas9-KRAB silences by depositing H3K9me3, which spreads from the guide and is
bounded by local chromatin rather than by anything cell-line specific.  If that
shows up at this depth, the genes near a target should be enriched for being
called differentially expressed and should go DOWN -- a target-specific,
directional prediction that needs no transfer of a measured response, which is
the one thing this model has never been able to do well.

Reads a reference DE table (`scripts/local_reference_de.py`) and a gene position
table, and reports the enrichment and the direction bias by distance.  Linear
distance stands in for the domain: if a boundary set were used instead it would
sharpen the cutoff, but the measured decay is smooth and over by 500 kb.
"""

from __future__ import annotations

import argparse

import numpy as np
import polars as pl

BINS = [(0, 1e5), (1e5, 5e5), (5e5, 1e6), (1e6, 5e6), (5e6, np.inf)]
LABELS = ["<100 kb", "100-500 kb", "0.5-1 Mb", "1-5 Mb", ">5 Mb same chr"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--de", required=True, help="reference DE table (parquet)")
    p.add_argument("--positions", required=True, help="{symbol: [chrom, midpoint]} JSON")
    p.add_argument("--p-adj", type=float, default=0.05)
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
    sig = padj < args.p_adj

    from vcc2026.proximity import load_gene_positions

    pos = load_gene_positions(args.positions)
    chrom = np.array([pos[g][0] if g in pos else "" for g in genes])
    centre = np.array([pos[g][1] if g in pos else np.nan for g in genes])
    placed = chrom != ""
    print(f"{int(placed.sum())} of {genes.size} tested genes placed; "
          f"{sum(t in pos for t in perts)} of {perts.size} targets placed")

    acc = {label: {"n": 0, "sig": 0, "down": 0, "lfc": []} for label in LABELS}
    far = {"n": 0, "sig": 0, "down": 0, "lfc": []}
    close_counts = []
    for i, target in enumerate(perts):
        if target not in pos:
            continue
        tchrom, tcentre = pos[target]
        own = genes == target
        same = placed & (chrom == tchrom) & ~own
        other = placed & (chrom != tchrom) & ~own
        dist = np.abs(centre - tcentre)
        close_counts.append(int((same & (dist < 5e5)).sum()))
        for (lo, hi), label in zip(BINS, LABELS, strict=True):
            band = same & (dist >= lo) & (dist < hi)
            if not band.any():
                continue
            acc[label]["n"] += int(band.sum())
            acc[label]["sig"] += int(sig[i][band].sum())
            acc[label]["down"] += int((lfc[i][band] < 0).sum())
            acc[label]["lfc"].append(lfc[i][band])
        far["n"] += int(other.sum())
        far["sig"] += int(sig[i][other].sum())
        far["down"] += int((lfc[i][other] < 0).sum())
        far["lfc"].append(lfc[i][other])

    print(f"\n{'distance':22s} {'pairs':>8s} {'P(sig)':>8s} {'P(down)':>8s} {'mean lfc':>9s}")
    for label in LABELS:
        row = acc[label]
        if row["n"] == 0:
            continue
        values = np.concatenate(row["lfc"])
        print(f"{label:22s} {row['n']:8d} {row['sig'] / row['n']:8.4f} "
              f"{row['down'] / row['n']:8.4f} {values.mean():9.4f}")
    values = np.concatenate(far["lfc"])
    print(f"{'different chromosome':22s} {far['n']:8d} {far['sig'] / far['n']:8.4f} "
          f"{far['down'] / far['n']:8.4f} {values.mean():9.4f}")
    print(f"\ngenes within 500 kb per perturbation: median {np.median(close_counts):.0f}, "
          f"mean {np.mean(close_counts):.1f}")


if __name__ == "__main__":
    main()
