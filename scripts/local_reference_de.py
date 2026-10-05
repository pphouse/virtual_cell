#!/usr/bin/env python
"""Compute a local reference's DE table with the competition's own code.

Four of the six scored metrics read nothing but this table, and three of those
four are decided as much by *how many* genes it calls as by which: fidelity
divides by `max(n_pred, n_real)` and the Jaccard by the union, so the size of a
submission's call set is itself a scored quantity.  Until this table exists that
size is being guessed.

Run under the scorer's own environment (`.venv-eval`), with `cell_eval2`'s
`vcc2026` parameters passed explicitly so the table is the one the leaderboard
would compute.
"""

from __future__ import annotations

import argparse
import logging

import anndata as ad
from cell_eval2.de_compute import compute_de

logger = logging.getLogger("refde")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--adata", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pert-col", default="target")
    p.add_argument("--control", default="non-targeting")
    p.add_argument("--backend", default="pdex")
    args = p.parse_args()

    adata = ad.read_h5ad(args.adata)
    logger.info("loaded %s", adata.shape)
    df = compute_de(
        adata,
        backend=args.backend,
        groupby=args.pert_col,
        reference=args.control,
        mean_calc="arithmetic",
        epsilon=1e-9,
        input_type="counts",
        target_sum=1e6,
        clip_value=None,
        filter_gene_min_cpm_cell=5.0,
        fdr_scope="per_pert",
        threads=-1,
        device="cpu",
    )
    logger.info("DE table: %s rows, columns %s", df.height, df.columns)
    df.write_parquet(args.out)
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
