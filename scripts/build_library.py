#!/usr/bin/env python
"""Collapse source perturbation screens into a cached SignatureLibrary.

    python scripts/build_library.py \
        --genes data/gene_names.csv \
        --source H1=data/vcc2025/adata_Training.h5ad \
        --source K562=data/replogle/K562_gwps.h5ad \
        --out outputs/library.npz

Every source is streamed once and reduced to one pseudobulk vector per
perturbation, so the cached library is a few hundred MB at most regardless of
how many cells went in.  Re-running the model never needs to touch the raw
h5ads again.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import anndata as ad
import numpy as np
from vcc2026.submit import read_gene_list

from vcc2026.library import SignatureLibrary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--genes", required=True, help="challenge gene list CSV (no header)")
    p.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="a source screen, repeatable",
    )
    p.add_argument("--out", required=True)
    p.add_argument("--pert-col", default="target_gene")
    p.add_argument("--control-name", default="non-targeting")
    p.add_argument("--min-cells", type=int, default=5)
    p.add_argument(
        "--backed", action="store_true", help="read h5ads in backed mode (low memory, slower)"
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    genes = read_gene_list(args.genes)
    lib = SignatureLibrary(genes=np.asarray(genes, dtype=object))

    for spec in args.source:
        if "=" not in spec:
            raise SystemExit(f"--source expects NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        logging.info("reading %s from %s", name, path)
        adata = ad.read_h5ad(path, backed="r" if args.backed else None)
        lib.add_source(
            adata,
            line=name,
            pert_col=args.pert_col,
            control_name=args.control_name,
            min_cells=args.min_cells,
        )
        del adata

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    lib.save(args.out)
    logging.info(
        "wrote %s: %d lines, %d distinct targets", args.out, len(lib.lines), len(lib.targets())
    )


if __name__ == "__main__":
    main()
