#!/usr/bin/env python
"""Predict one cell line's perturbation responses and write a submission.

    python scripts/predict.py \
        --library outputs/library.npz \
        --genes data/gene_names.csv \
        --context HEK293T=data/val/adata_HEK293T.h5ad \
        --pert-counts data/val/pert_counts_Validation.csv \
        --out outputs/submission.h5ad --prep

`--context` may be repeated: the 2026 phases each score three held-out lines,
and the leaderboard takes them in a single file, so all contexts are predicted
and concatenated.  If one h5ad holds every line, pass it once as
``--context all=path.h5ad --celltype-col celltype`` and the lines are split out
of that column instead.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import anndata as ad
import numpy as np
from vcc2026.submit import read_gene_list, read_pert_counts, write_submission

from vcc2026 import CellContext, CellSampler, ContextTransferModel, ModelConfig, SamplerConfig
from vcc2026.features import load_embeddings
from vcc2026.library import SignatureLibrary

logger = logging.getLogger("predict")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--library", required=True)
    p.add_argument("--genes", required=True)
    p.add_argument("--context", action="append", required=True, metavar="NAME=PATH")
    p.add_argument("--celltype-col", default=None, help="split one context file by this obs column")
    p.add_argument(
        "--pert-counts", required=True, help="pert_counts CSV: how many cells per perturbation"
    )
    p.add_argument(
        "--counts-celltype-col", default=None, help="obs/CSV column naming the cell line per row"
    )
    p.add_argument("--embeddings", default=None, help="npz with `genes` and `embeddings`")
    p.add_argument("--out", required=True)
    p.add_argument("--prep", action="store_true", help="also run `cell-eval prep`")
    p.add_argument("--alpha", type=float, default=ModelConfig.alpha)
    p.add_argument("--n-components", type=int, default=ModelConfig.n_components)
    p.add_argument("--n-neighbours", type=int, default=ModelConfig.n_neighbours)
    p.add_argument("--magnitude-gamma", type=float, default=ModelConfig.magnitude_gamma)
    p.add_argument(
        "--non-responder-fraction", type=float, default=SamplerConfig.non_responder_fraction
    )
    p.add_argument("--response-shape", type=float, default=SamplerConfig.response_shape)
    p.add_argument("--max-control-cells", type=int, default=20000)
    p.add_argument(
        "--target-sum",
        type=float,
        default=None,
        help="normlog scale; default = the context's own median library size, "
        "which is what cell-eval applies to the reference data",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_contexts(args, genes: np.ndarray) -> dict[str, CellContext]:
    rng = np.random.default_rng(args.seed)
    contexts: dict[str, CellContext] = {}
    for spec in args.context:
        if "=" not in spec:
            raise SystemExit(f"--context expects NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        adata = ad.read_h5ad(path)
        if args.celltype_col and args.celltype_col in adata.obs:
            for line in adata.obs[args.celltype_col].astype(str).unique():
                sub = adata[adata.obs[args.celltype_col].astype(str) == line]
                contexts[str(line)] = CellContext.from_anndata(
                    sub, name=str(line), genes=genes, target_sum=args.target_sum
                ).subsample(args.max_control_cells, rng)
        else:
            contexts[name] = CellContext.from_anndata(
                adata, name=name, genes=genes, target_sum=args.target_sum
            ).subsample(args.max_control_cells, rng)
    return contexts


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    genes = read_gene_list(args.genes)
    library = SignatureLibrary.load(args.library)
    contexts = load_contexts(args, genes)
    logger.info(
        "contexts: %s",
        {k: (v.n_control, round(v.target_sum)) for k, v in contexts.items()},
    )

    counts = read_pert_counts(args.pert_counts)
    embeddings = load_embeddings(args.embeddings) if args.embeddings else None

    config = ModelConfig(
        alpha=args.alpha,
        n_components=args.n_components,
        n_neighbours=args.n_neighbours,
        magnitude_gamma=args.magnitude_gamma,
        seed=args.seed,
    )
    model = ContextTransferModel(config).fit(library, embeddings=embeddings)
    sampler = CellSampler(
        SamplerConfig(
            non_responder_fraction=args.non_responder_fraction,
            response_shape=args.response_shape,
            seed=args.seed,
        )
    )

    blocks = []
    for name, ctx in contexts.items():
        targets = sorted(t for t in counts if t != "non-targeting")
        pred = model.predict(targets, ctx)
        logger.info(
            "%s: %d targets (%d with a measured signature, %d fell back), median |delta| = %.3f",
            name,
            len(targets),
            int(pred.observed.sum()),
            int(pred.fallback.sum()),
            float(np.median(pred.magnitude)),
        )
        blocks.append(sampler.sample(ctx, pred, counts))

    adata = blocks[0] if len(blocks) == 1 else ad.concat(blocks, index_unique=None)
    adata.obs_names = np.arange(adata.n_obs).astype(str)
    out = write_submission(
        adata,
        Path(args.out),
        genes=genes,
        gene_list_path=args.genes,
        run_prep=args.prep,
    )
    logger.info("submission ready: %s", out)


if __name__ == "__main__":
    main()
