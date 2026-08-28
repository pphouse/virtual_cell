#!/usr/bin/env python
"""Cache DepMap's gene-effect matrix as features, restricted to the genes we ask about.

The published CSV is 429 MB of cell lines by genes; a prediction only ever needs
the library's targets and the panel's, so the cached npz is a few megabytes and
loads in a second instead of a minute.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from vcc2026.coessentiality import read_gene_effect
from vcc2026.library import SignatureLibrary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gene-effect", required=True, help="DepMap CRISPRGeneEffect.csv")
    p.add_argument("--library", required=True, help="restrict to this library's targets")
    p.add_argument("--also", default=None, help="comma-separated extra gene symbols")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    lib = SignatureLibrary.load(args.library)
    wanted = sorted(set(lib.targets()) | set(str(g) for g in lib.genes))
    if args.also:
        wanted = sorted(set(wanted) | {x.strip() for x in args.also.split(",")})
    feats = read_gene_effect(args.gene_effect, subset=wanted)
    np.savez_compressed(
        args.out, genes=np.asarray(feats.genes, dtype=str), matrix=feats.matrix
    )
    print(f"wrote {args.out}: {feats.genes.size} genes x {feats.matrix.shape[1]} lines")


if __name__ == "__main__":
    main()
