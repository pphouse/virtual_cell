#!/usr/bin/env python
"""Score an emitted prediction against a local reference, member by member.

The differential-expression table for the prediction is computed by
`cell_eval2`'s own `compute_de`, against the *reference's* control cells -- which
is what the competition does on both sides -- and the four DE members are then
evaluated exactly as `docs/vcc2026_metrics` defines them.  The raw values are
printed beside the officially measured baseline and replicate so a number can be
read as the score it would imply.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as sp
from cell_eval2.de_compute import compute_de
from cell_eval2.metrics.de import de_lfc_nmae, de_sig_jaccard
from cell_eval2.metrics.direction import de_direction_fidelity_yield_raw, de_direction_reach

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_local_context import read_obs_column, read_var_names, stream_rows  # noqa: E402
from strategy_search import scaled  # noqa: E402

logger = logging.getLogger("localscore")


def tables_to_arrays(real: pl.DataFrame, pred: pl.DataFrame):
    perts = sorted(set(real["target"].unique().to_list()) & set(pred["target"].unique().to_list()))
    genes = sorted(real["feature"].unique().to_list())
    pi = {p: i for i, p in enumerate(perts)}
    gi = {g: i for i, g in enumerate(genes)}
    shape = (len(perts), len(genes))

    def fill(df):
        lfc = np.zeros(shape)
        padj = np.ones(shape)
        t = df["target"].to_numpy()
        f = df["feature"].to_numpy()
        keep = np.array([x in pi for x in t]) & np.array([x in gi for x in f])
        rows = np.array([pi[x] for x in t[keep]])
        cols = np.array([gi[x] for x in f[keep]])
        lfc[rows, cols] = df["log2_fold_change"].to_numpy()[keep]
        padj[rows, cols] = df["p_adj"].to_numpy()[keep]
        return lfc, np.nan_to_num(padj, nan=1.0)

    return perts, genes, fill(real), fill(pred)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", required=True)
    p.add_argument("--prediction", required=True)
    p.add_argument("--de-real", required=True)
    p.add_argument("--de-pred", default=None, help="reuse a table instead of recomputing")
    p.add_argument("--out-de-pred", default=None)
    p.add_argument(
        "--max-n-real",
        type=int,
        default=0,
        help="score only perturbations the reference moves at most this many genes for; "
        "0 keeps all. The 2026 panel's mean n_real is about 275 (docs/06), and a local "
        "panel of stronger knockdowns answers a different question.",
    )
    args = p.parse_args()

    if args.de_pred:
        pred_de = pl.read_parquet(args.de_pred)
    else:
        genes = read_var_names(args.reference, None)
        ref_labels = read_obs_column(args.reference, "target")
        ctrl = np.flatnonzero(ref_labels == "non-targeting")
        blocks = [b for _, b in stream_rows(args.reference, ctrl, block=8192)]
        control = sp.vstack(blocks, format="csr").astype(np.float32)
        del blocks

        pred_labels = read_obs_column(args.prediction, "target")
        keep = np.flatnonzero(pred_labels != "non-targeting")
        blocks = [b for _, b in stream_rows(args.prediction, keep, block=8192)]
        pert = sp.vstack(blocks, format="csr").astype(np.float32)
        del blocks

        obs = pd.DataFrame(
            {
                "target": pd.Categorical(
                    np.concatenate(
                        [np.repeat("non-targeting", control.shape[0]), pred_labels[keep]]
                    )
                )
            },
            index=np.arange(control.shape[0] + pert.shape[0]).astype(str),
        )
        adata = ad.AnnData(
            X=sp.vstack([control, pert], format="csr"),
            obs=obs,
            var=pd.DataFrame(index=pd.Index(genes)),
        )
        del control, pert
        logger.info("prediction DE input: %s", adata.shape)
        pred_de = compute_de(
            adata,
            backend="pdex",
            groupby="target",
            reference="non-targeting",
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
        if args.out_de_pred:
            pred_de.write_parquet(args.out_de_pred)

    real_de = pl.read_parquet(args.de_real)
    perts, genes, (rl, rp), (pll, pp) = tables_to_arrays(real_de, pred_de)
    logger.info(
        "%d perturbations, %d genes; n_real median %d, n_pred median %d",
        len(perts),
        len(genes),
        int(np.median((rp < 0.05).sum(1))),
        int(np.median((pp < 0.05).sum(1))),
    )

    if args.max_n_real:
        n_real = (rp < 0.05).sum(1)
        keep = {perts[i] for i in np.flatnonzero(n_real <= args.max_n_real)}
        logger.info(
            "restricting to %d of %d perturbations with n_real <= %d (mean %d)",
            len(keep),
            len(perts),
            args.max_n_real,
            int(n_real[n_real <= args.max_n_real].mean()),
        )
        real_de = real_de.filter(pl.col("target").is_in(list(keep)))
        pred_de = pred_de.filter(pl.col("target").is_in(list(keep)))

    # The scorer's own readers, which take DE tables directly -- no matrices, and
    # no room for this project's reimplementation to drift from them.
    kw = dict(de_pred=pred_de, de_real=real_de, control="non-targeting")
    out = {
        "fid": de_direction_fidelity_yield_raw(**kw),
        "reach": de_direction_reach(**kw),
        "jac": de_sig_jaccard(**kw),
        "nmae": de_lfc_nmae(**kw),
    }
    out = {
        k: (float(np.mean(list(v.values()))) if isinstance(v, dict) else float(v))
        for k, v in out.items()
    }
    total = sum(scaled(k, out[k]) for k in out)
    print(
        "  ".join(f"{k}={out[k]:.4f}" for k in ("fid", "reach", "jac", "nmae"))
        + f"   sum_scaled/6={total / 6:.4f}"
    )


if __name__ == "__main__":
    main()
