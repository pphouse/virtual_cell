#!/usr/bin/env python
"""Measure, on real data, the two things the control-only predictor guesses.

The 2026 contexts come with no measured response, so nothing about a prediction
for them can be checked locally.  The 2025 public release *is* measured: 150
CRISPRi knockdowns in H1 hESC.  Running the control-only predictor there --
fitting it on the 2025 control cells alone, exactly as it is fitted on a 2026
context -- turns two guesses into measurements:

1. **The on-target knockdown residual.**  `knockdown_residual` is set from the
   CRISPRi literature.  Here it can be read off the data: how far does the
   target gene actually fall?

2. **Whether the co-expression trans prior is worth anything.**  This is the
   open question that decides `trans_beta`, and it is not a free parameter to
   guess at: the 2026 aggregate has no floor at zero, so a trans signature that
   is noise scores *below* the context-mean baseline rather than merely failing
   to beat it.  The number that matters is the correlation between the
   predicted trans signature and the measured one, per perturbation.

Both are measured in one cell line, so they transfer to A/B/C only as far as
CRISPRi biology is shared between contexts -- which is exactly the assumption
the whole challenge is testing.  Treated as a sanity floor, not a guarantee.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from vcc2026.coexpression import ControlOnlyConfig, ControlOnlyPredictor
from vcc2026.library import (
    SignatureLibrary,
    _read_categorical,
    _read_string_index,
    add_source_streaming,
)

logger = logging.getLogger("validate2025")


def load_control_cells(
    path: str | Path,
    genes: np.ndarray,
    pert_col: str = "target_gene",
    control_name: str = "non-targeting",
    max_cells: int = 20000,
    seed: int = 0,
) -> sp.csr_matrix:
    """Pull just the control cells out of a screen too large to open."""
    with h5py.File(path, "r") as f:
        labels = _read_categorical(f["obs"], pert_col)
        src_genes = _read_string_index(f["var"])
        idx = np.flatnonzero(labels == control_name)
        if idx.size == 0:
            raise ValueError(f"no {control_name!r} cells in {path}")
        rng = np.random.default_rng(seed)
        if idx.size > max_cells:
            idx = np.sort(rng.choice(idx, size=max_cells, replace=False))
        logger.info("reading %d control cells", idx.size)

        ptr = f["X/indptr"][:]
        data, indices = f["X/data"], f["X/indices"]
        rows_d, rows_i, indptr = [], [], np.zeros(idx.size + 1, dtype=np.int64)
        for k, i in enumerate(idx):
            lo, hi = int(ptr[i]), int(ptr[i + 1])
            rows_d.append(data[lo:hi])
            rows_i.append(indices[lo:hi])
            indptr[k + 1] = indptr[k] + (hi - lo)
        src = sp.csr_matrix(
            (np.concatenate(rows_d), np.concatenate(rows_i).astype(np.int64), indptr),
            shape=(idx.size, src_genes.size),
        )

    # Re-index into the submission gene space by rewriting the column indices in
    # place.  A sparse permutation matmul would do the same thing and allocate
    # several copies of a matrix this size on the way.
    pos = {g: i for i, g in enumerate(np.asarray(genes, dtype=str))}
    col = np.array([pos.get(g, -1) for g in src_genes], dtype=np.int64)
    new_indices = col[src.indices]
    keep = new_indices >= 0
    if not keep.all():
        # Drop entries for genes absent from the submission gene space, fixing
        # up the row boundaries as we go.
        per_row = (
            np.add.reduceat(keep.astype(np.int64), src.indptr[:-1])
            if src.nnz
            else np.zeros(src.shape[0], np.int64)
        )
        per_row[np.diff(src.indptr) == 0] = 0
        indptr = np.concatenate([[0], np.cumsum(per_row)])
    else:
        indptr = src.indptr
    out = sp.csr_matrix(
        (src.data[keep].astype(np.float32), new_indices[keep], indptr),
        shape=(src.shape[0], len(genes)),
    )
    out.sort_indices()
    logger.info(
        "control matrix %s, %d stored entries (%d dropped: gene not in the 2026 space)",
        out.shape,
        out.nnz,
        int((~keep).sum()),
    )
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--screen", required=True, help="2025 adata_Training.h5ad")
    p.add_argument("--genes", required=True, help="2026 gene_names.csv (the shared gene space)")
    p.add_argument("--library", default=None, help="cached library npz (built if absent)")
    p.add_argument("--out", required=True, help="where to write the JSON report")
    p.add_argument("--max-control-cells", type=int, default=18400)
    p.add_argument("--betas", default="0.0,0.05,0.1,0.2,0.4,0.8")
    p.add_argument("--n-components", type=int, default=50)
    args = p.parse_args()

    genes = pd.read_csv(args.genes).iloc[:, 0].astype(str).to_numpy()

    if args.library and Path(args.library).exists():
        lib = SignatureLibrary.load(args.library)
    else:
        lib = SignatureLibrary(genes=np.asarray(genes, dtype=object))
        add_source_streaming(lib, args.screen, line="H1")
        if args.library:
            lib.save(args.library)

    line = lib.lines[0]
    targets = [t for t in lib.targets(line) if lib.gene_index(t) is not None]
    logger.info("%d measured targets inside the gene space", len(targets))
    measured = np.vstack([lib.deltas[line][t] for t in targets])
    pos = np.array([lib.gene_index(t) for t in targets])

    # --- 1. the on-target knockdown, measured ---------------------------
    base = lib.baseline[line]
    e0 = np.expm1(np.clip(base[pos], 0, None))
    e1 = np.expm1(np.clip(base[pos] + measured[np.arange(len(targets)), pos], 0, None))
    residual = (e1 + 0.5) / (e0 + 0.5)
    expressed = base[pos] > 0.05
    report = {
        "n_targets": len(targets),
        "n_expressed_targets": int(expressed.sum()),
        "on_target_residual": {
            "median": float(np.median(residual[expressed])),
            "q25": float(np.quantile(residual[expressed], 0.25)),
            "q75": float(np.quantile(residual[expressed], 0.75)),
            "fraction_down": float((residual[expressed] < 1).mean()),
        },
    }
    logger.info("on-target residual: %s", report["on_target_residual"])

    # --- 2. is the co-expression trans prior real? ----------------------
    controls = load_control_cells(args.screen, genes, max_cells=args.max_control_cells)
    model = ControlOnlyPredictor(
        ControlOnlyConfig(n_components=args.n_components, trans_beta=1.0)
    ).fit(controls, np.asarray(genes, dtype=object))

    trans_rows = []
    for i, t in enumerate(targets):
        corr = model.program_correlation(t)
        corr[pos[i]] = 0.0
        truth = measured[i].copy()
        truth[pos[i]] = 0.0
        keep = base > 0.05
        if corr[keep].std() == 0 or truth[keep].std() == 0:
            continue
        trans_rows.append(
            {
                "target": t,
                "r": float(np.corrcoef(corr[keep], truth[keep])[0, 1]),
                "effect_size": float(np.linalg.norm(truth[keep])),
            }
        )
    r = np.array([row["r"] for row in trans_rows])
    strong = np.array([row["effect_size"] for row in trans_rows])
    order = np.argsort(-strong)[: max(len(r) // 4, 1)]
    report["coexpression_trans"] = {
        "n_scored": int(r.size),
        "mean_r": float(r.mean()),
        "median_r": float(np.median(r)),
        "fraction_positive": float((r > 0).mean()),
        "mean_r_strongest_quartile": float(r[order].mean()),
    }
    logger.info("co-expression trans prior: %s", report["coexpression_trans"])

    Path(args.out).write_text(json.dumps(report, indent=2))
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
