#!/usr/bin/env python
"""Is the counts emission a statistical no-op when nothing is perturbed?

The first submission predicted a change in exactly one gene per perturbation and
still lost 0.34 on DE direction fidelity -- a raw 0.41 where the context-mean
baseline scores about 0.51, i.e. the predicted directions came out *worse than
chance*.  The non-target predictions are noise by construction (measured: 47-49%
of genes up, median log fold change -0.002), so noise alone cannot produce a
systematically wrong direction.  Something in the emission itself must be.

This tests that directly, and needs no held-out data to do it.  Split a
context's real control cells in two.  Emit cells from one half with the fold
change set to *zero* -- a prediction of "nothing happens" -- and run the same
Wilcoxon test the scorer runs against the other half.  A faithful emission
should call almost nothing significant, and whatever it does call should be
symmetric in direction.  Real cells drawn from the same half give the true null
to compare against.

Any excess is an artefact the model never asked for, applied identically to all
300 perturbations -- which is exactly the shape of a systematic direction error.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
from scipy.stats import false_discovery_control, mannwhitneyu

from vcc2026.counts import context_mean_proportions, emit_counts

logger = logging.getLogger("nulltest")


def normlog(x: sp.csr_matrix) -> np.ndarray:
    totals = np.asarray(x.sum(axis=1)).ravel()
    totals[totals == 0] = 1.0
    target = float(np.median(totals))
    out = sp.diags(target / totals) @ x
    out = np.asarray(out.todense(), dtype=np.float32)
    return np.log1p(out)


def de_against(reference: np.ndarray, sample: np.ndarray, alpha: float = 0.05) -> dict:
    """Per-gene Wilcoxon of `sample` vs `reference`, FDR-controlled."""
    keep = (reference > 0).mean(axis=0) > 0.01
    stat, p = mannwhitneyu(sample[:, keep], reference[:, keep], axis=0, alternative="two-sided")
    q = false_discovery_control(np.nan_to_num(p, nan=1.0))
    sig = q < alpha
    lfc = sample[:, keep].mean(axis=0) - reference[:, keep].mean(axis=0)
    return {
        "n_tested": int(keep.sum()),
        "n_significant": int(sig.sum()),
        "fraction_significant": float(sig.mean()),
        "fraction_of_significant_that_are_up": float((lfc[sig] > 0).mean())
        if sig.any()
        else float("nan"),
        "median_lfc_of_significant": float(np.median(lfc[sig])) if sig.any() else float("nan"),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--controls", required=True, help="one context_X.h5ad")
    p.add_argument("--out", required=True)
    p.add_argument("--n-emit", type=int, default=400)
    p.add_argument("--pseudocounts", default="0.0,0.5,1.0,2.0")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    adata = ad.read_h5ad(args.controls)
    counts = adata.X.tocsr()
    del adata
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(counts.shape[0])
    half = counts.shape[0] // 2
    reference = counts[perm[:half]]
    source = counts[perm[half:]]
    logger.info("reference %d cells, source pool %d cells", reference.shape[0], source.shape[0])

    ref_log = normlog(reference)
    mean_p = context_mean_proportions(source)
    library = np.asarray(source.sum(axis=1)).ravel()

    results = {}

    # The true null: real cells, drawn from the same pool, tested the same way.
    pick = rng.choice(source.shape[0], size=args.n_emit, replace=False)
    results["real_cells"] = de_against(ref_log, normlog(source[pick]))
    logger.info("real cells      -> %s", results["real_cells"])

    for a in [float(x) for x in args.pseudocounts.split(",")]:
        for mode, resample_all in (("full_resample", True), ("active_only", False)):
            pick = rng.choice(source.shape[0], size=args.n_emit, replace=False)
            emitted = emit_counts(
                source[pick],
                mean_proportions=mean_p,
                log_fold_change=np.zeros(counts.shape[1]),
                pseudocount=a,
                rng=np.random.default_rng(args.seed + 7),
                library_sizes=library[pick],
                resample_all=resample_all,
            )
            key = f"{mode}_pseudocount_{a}"
            results[key] = de_against(ref_log, normlog(emitted))
            logger.info("%-14s a=%-4s -> %s", mode, a, results[key])

    # And with an actual prediction: does a single knocked-down gene stay the
    # only thing the DE test picks up?
    pick = rng.choice(source.shape[0], size=args.n_emit, replace=False)
    lfc = np.zeros(counts.shape[1])
    strong = int(np.argsort(-np.asarray(source.sum(axis=0)).ravel())[50])
    lfc[strong] = np.log(0.15)
    emitted = emit_counts(
        source[pick],
        mean_proportions=mean_p,
        log_fold_change=lfc,
        pseudocount=1.0,
        rng=np.random.default_rng(args.seed + 9),
        library_sizes=library[pick],
    )
    results["one_gene_knockdown"] = de_against(ref_log, normlog(emitted))
    logger.info("one-gene knockdown -> %s", results["one_gene_knockdown"])

    Path(args.out).write_text(json.dumps(results, indent=2))
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
