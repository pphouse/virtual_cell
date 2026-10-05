#!/usr/bin/env python
"""Cut a public screen down to the exact shape of a 2026 evaluation context.

The leaderboard is the only place the real scorer has ever run on this work, and
it costs a submission and an hour to ask it anything.  `cell_eval2` is public, so
the scorer itself can run here -- but it needs a reference with ground truth, and
the challenge ships only control cells for A/B/C.

So the reference is built from a screen that does have ground truth, cut to the
shape the competition scores: 300-ish target constructs at 400 cells each, a
pooled non-targeting control, and a median depth of 20,000 UMI per cell.  Depth
matters more than it looks: every DE metric is a Wilcoxon test, and how many
genes come back significant is set by depth and group size as much as by
biology.  A reference measured at the release's own 54,000 UMI would answer a
different question from the one the leaderboard asks.
"""

from __future__ import annotations

import argparse
import logging

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger("context")


def read_obs_column(path: str, col: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        node = f["obs"][col]
        if isinstance(node, h5py.Group):  # categorical
            codes = node["codes"][:]
            cats = node["categories"][:]
            cats = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in cats])
            return cats[codes]
        raw = node[:]
        return np.array([c.decode() if isinstance(c, bytes) else str(c) for c in raw])


def read_var_names(path: str, key: str | None) -> np.ndarray:
    with h5py.File(path, "r") as f:
        var = f["var"]
        name = key or var.attrs.get("_index", "_index")
        if isinstance(name, bytes):
            name = name.decode()
        raw = var[name][:]
    return np.array([c.decode() if isinstance(c, bytes) else str(c) for c in raw])


def stream_rows(path: str, rows: np.ndarray, block: int = 8192):
    """Yield (offset, csr_block) for `rows` (sorted) without loading the matrix."""
    with h5py.File(path, "r") as f:
        X = f["X"]
        dense = isinstance(X, h5py.Dataset)
        if dense:
            n_genes = X.shape[1]
        else:
            n_genes = int(X.attrs["shape"][1])
            indptr = X["indptr"][:]
            data_d, idx_d = X["data"], X["indices"]
        for start in range(0, rows.size, block):
            chunk = rows[start : start + block]
            if dense:
                yield start, sp.csr_matrix(X[chunk, :])
                continue
            lo, hi = indptr[chunk], indptr[chunk + 1]
            counts = hi - lo
            total = int(counts.sum())
            data = np.empty(total, dtype=data_d.dtype)
            indices = np.empty(total, dtype=idx_d.dtype)
            at = 0
            for a, b in zip(lo, hi, strict=True):
                n = int(b - a)
                if n:
                    data[at : at + n] = data_d[a:b]
                    indices[at : at + n] = idx_d[a:b]
                at += n
            ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
            yield start, sp.csr_matrix((data, indices, ptr), shape=(chunk.size, n_genes))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pert-col", default="target_gene")
    p.add_argument("--control-name", default="non-targeting")
    p.add_argument("--var-key", default=None)
    p.add_argument("--cells-per-pert", type=int, default=400)
    p.add_argument("--control-cells", type=int, default=18400)
    p.add_argument("--median-umi", type=float, default=20000.0)
    p.add_argument("--max-perts", type=int, default=0, help="0 = all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--half",
        type=int,
        default=0,
        help="1 or 2 to take a disjoint half of each group's eligible cells; 0 takes all",
    )
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    labels = read_obs_column(args.source, args.pert_col)
    genes = read_var_names(args.source, args.var_key)
    logger.info("source: %d cells, %d genes", labels.size, genes.size)

    perts = sorted({label for label in labels if label != args.control_name})
    counts = pd.Series(labels).value_counts()
    perts = [t for t in perts if counts[t] >= args.cells_per_pert]
    if args.max_perts:
        perts = perts[: args.max_perts]
    logger.info(
        "%d targets with >= %d cells; control pool %d",
        len(perts),
        args.cells_per_pert,
        int(counts.get(args.control_name, 0)),
    )

    picked, assigned = [], []
    for name, want in [(args.control_name, args.control_cells)] + [
        (t, args.cells_per_pert) for t in perts
    ]:
        pool = np.flatnonzero(labels == name)
        pool = rng.permutation(pool)
        if args.half:
            mid = pool.size // 2
            pool = pool[:mid] if args.half == 1 else pool[mid:]
        take = pool[: min(want, pool.size)]
        picked.append(take)
        assigned.append(np.repeat(name, take.size))
    rows = np.concatenate(picked)
    obs_labels = np.concatenate(assigned)
    order = np.argsort(rows, kind="stable")
    rows, obs_labels = rows[order], obs_labels[order]
    logger.info("selected %d cells", rows.size)

    from vcc2026.writer import SparseH5adWriter

    obs = pd.DataFrame(
        {"target": pd.Categorical(obs_labels)},
        index=np.arange(rows.size).astype(str),
    )
    keep = None
    nnz = 0
    with SparseH5adWriter(args.out, obs=obs, var_names=genes) as writer:
        for _, block in stream_rows(args.source, rows):
            block = block.astype(np.float32)
            if args.median_umi:
                if keep is None:
                    median = float(np.median(np.asarray(block.sum(axis=1)).ravel()))
                    keep = min(1.0, args.median_umi / median) if median > 0 else 1.0
                    logger.info("median UMI %.0f -> thinning by %.3f", median, keep)
                if keep < 1.0:
                    block.data = rng.binomial(
                        block.data.astype(np.int32), keep
                    ).astype(np.float32)
                    block.eliminate_zeros()
            nnz += block.nnz
            writer.append(block)

    logger.info("wrote %s  (%d, %d)  nnz=%d", args.out, rows.size, genes.size, nnz)


if __name__ == "__main__":
    main()
