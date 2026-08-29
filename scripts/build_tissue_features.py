#!/usr/bin/env python
"""Cache GTEx's median-TPM-by-tissue matrix as features, restricted to our genes.

Mirrors `build_coessentiality.py`: the published file covers 59,033 genes and a
prediction only ever asks about the library's targets and the panel's, so the
cached npz is small and loads instantly.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from vcc2026.library import SignatureLibrary
from vcc2026.tissue import read_median_tpm


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--median-tpm", required=True, help="GTEx gene_median_tpm.gct.gz")
    p.add_argument("--library", required=True, help="restrict to this library's targets")
    p.add_argument("--also", default=None, help="comma-separated extra gene symbols")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    lib = SignatureLibrary.load(args.library)
    wanted = sorted(set(lib.targets()) | {str(g) for g in lib.genes})
    if args.also:
        wanted = sorted(set(wanted) | {x.strip() for x in args.also.split(",") if x.strip()})

    feats = read_median_tpm(args.median_tpm, subset=wanted)
    np.savez_compressed(
        args.out, genes=np.asarray(feats.genes, dtype=str), matrix=feats.matrix
    )
    logging.info(
        "wrote %s: %d of %d requested genes covered", args.out, feats.genes.size, len(wanted)
    )


if __name__ == "__main__":
    main()
