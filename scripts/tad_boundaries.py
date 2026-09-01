#!/usr/bin/env python
"""Do published TAD boundaries predict where a CRISPRi knockdown stops reaching?

The proximity effect (`scripts/proximity_effect.py`) is real, and the obvious
mechanism -- KRAB spreading stopped by an insulator -- predicts that a neighbour
on the far side of a domain boundary should be spared.  This checks that against
the Rao 2014 Arrowhead contact domains, in hg19 with hg19 gene coordinates so no
lift-over is involved.

The first way of asking, "does one domain contain both genes", conflates "an
insulator separates them" with "neither is inside an annotated domain" -- the
calls do not tile the genome.  Counting the boundaries strictly between the two
positions separates those, and restricting to targets that sit inside some
domain removes the unannotated regions entirely.

The answer is no: crossing a boundary leaves the effect intact.  The effect
lives at 10 to 100 kb and these domains average a quarter of a megabase, so the
boundary is the wrong ruler for it.

    scripts/tad_boundaries.py <scratch-dir>

expects <scratch-dir>/genome/gene_pos_hg19.json, <scratch-dir>/ref60_de.npz and
the two domain lists under <scratch-dir>/tad/ (GEO GSE63525).
"""
import gzip
import pathlib
import sys
from collections import defaultdict

import numpy as np

SP = sys.argv[1]

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from vcc2026.proximity import load_gene_positions  # noqa: E402

pos = load_gene_positions(f"{SP}/genome/gene_pos_hg19.json")
d = np.load(f"{SP}/ref60_de.npz", allow_pickle=True)
LFC, PADJ = d["LFC"], d["PADJ"]
SIG = PADJ < 0.05
genes = [str(x) for x in d["genes"]]
perts = [str(x) for x in d["perts"]]
gc = np.array([pos[g][0] if g in pos else "" for g in genes])
gp = np.array([pos[g][1] if g in pos else np.nan for g in genes], dtype=float)
have = gc != ""

def load(path):
    doms = defaultdict(list)
    with gzip.open(path, "rt") as f:
        next(f)
        for line in f:
            p = line.split("\t")
            doms["chr" + p[0]].append((int(p[1]), int(p[2])))
    return doms

def analyse(name, doms):
    edges = {c: np.array(sorted({x for iv in v for x in iv})) for c, v in doms.items()}
    covered = {c: np.array(sorted(v)) for c, v in doms.items()}
    span = sum(e - s for v in doms.values() for s, e in v) / 3.1e9
    print(f"\n=== {name}: {sum(len(v) for v in doms.values())} domains, "
          f"{span:.0%} of the genome covered")

    def inside(c, x):
        iv = covered.get(c)
        if iv is None:
            return False
        return bool(((iv[:, 0] <= x) & (x <= iv[:, 1])).any())

    BINS = [(0, 1e5), (1e5, 5e5), (5e5, 1e6)]
    LAB = ["<100 kb", "100-500 kb", "0.5-1 Mb"]
    acc = {
        (label, k): {"n": 0, "sig": 0, "dn": 0, "lfc": []}
        for label in LAB
        for k in ("0", "1+")
    }
    for i, t in enumerate(perts):
        if t not in pos:
            continue
        tc, tp = pos[t]
        if tc not in edges or not inside(tc, tp):
            continue
        e = edges[tc]
        own = np.array([g == t for g in genes])
        same = have & (gc == tc) & ~own
        dist = np.abs(gp - tp)
        for (lo, hi), lab in zip(BINS, LAB, strict=True):
            for j in np.flatnonzero(same & (dist >= lo) & (dist < hi)):
                a_, b_ = (tp, gp[j]) if tp <= gp[j] else (gp[j], tp)
                n_cross = int(np.searchsorted(e, b_, "left") - np.searchsorted(e, a_, "right"))
                key = (lab, "0" if n_cross == 0 else "1+")
                a = acc[key]
                a["n"] += 1
                a["sig"] += int(SIG[i, j])
                a["dn"] += int(LFC[i, j] < 0)
                a["lfc"].append(LFC[i, j])
    print(
        f"{'distance':14s} {'boundaries':>11s} {'pairs':>7s} "
        f"{'P(sig)':>8s} {'P(down)':>8s} {'mean lfc':>9s}"
    )
    for lab in LAB:
        for k in ("0", "1+"):
            a = acc[(lab, k)]
            if a["n"] == 0:
                continue
            v = np.array(a["lfc"])
            print(f"{lab:14s} {k:>11s} {a['n']:7d} {a['sig']/a['n']:8.4f} "
                  f"{a['dn']/a['n']:8.4f} {v.mean():9.4f}")

analyse("GM12878", load(f"{SP}/tad/GSE63525_GM12878_primary+replicate_Arrowhead_domainlist.txt.gz"))
analyse("IMR90", load(f"{SP}/tad/GSE63525_IMR90_Arrowhead_domainlist.txt.gz"))
