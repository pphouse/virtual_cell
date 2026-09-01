#!/usr/bin/env python
"""Pseudobulk the public 2025 Virtual Cell Challenge screen into a library.

The 2025 release is the only large public CRISPRi Perturb-seq measured on the
same platform as the 2026 challenge, so it is the natural first transfer
source.  Two caveats worth knowing before relying on it:

* it is a *single* cell line (H1 hESC), so it cannot on its own teach a model
  how responses differ between contexts -- that needs a second source line;
* only 13 of the 2026 challenge's 300 targets appear in it, so direct transfer
  covers 4% of the panel and the rest rides on the program extrapolation.

It is still worth building, because it is the only place to measure -- rather
than guess -- what a real CRISPRi knockdown looks like: how far the target gene
actually falls, and how much of the trans response a co-expression prior
recovers.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from vcc2026.library import SignatureLibrary, add_source_streaming


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--genes",
        required=True,
        help="the *2026* gene list -- the library lives in submission gene space",
    )
    p.add_argument("--source", action="append", required=True, metavar="NAME=PATH")
    p.add_argument("--out", required=True)
    p.add_argument("--min-cells", type=int, default=5)
    p.add_argument("--block", type=int, default=4096)
    args = p.parse_args()

    import pandas as pd

    genes = pd.read_csv(args.genes).iloc[:, 0].astype(str).to_numpy()
    lib = SignatureLibrary(genes=np.asarray(genes, dtype=object))
    for spec in args.source:
        name, path = spec.split("=", 1)
        add_source_streaming(lib, path, line=name, min_cells=args.min_cells, block=args.block)
    lib.save(args.out)
    logging.info("wrote %s: lines %s, %d targets", args.out, lib.lines, len(lib.targets()))


if __name__ == "__main__":
    main()
