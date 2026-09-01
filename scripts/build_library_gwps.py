#!/usr/bin/env python
"""Add Replogle's genome-wide K562 screen to a signature library, from pseudobulk.

This project spent its first twelve submissions on `ReplogleWeissman2022_K562_
essential` -- 2,058 targets, none of which is one of the challenge's 300.  That
absence was read as "the panel does not overlap any public screen, so every
prediction has to be extrapolated from gene similarity", and everything since
was built on it.  The genome-wide release covers 9,867 targets and **272 of the
300**, which turns most of the task from extrapolation into look-up.  Several
leaderboard entries say as much in their own descriptions.

The single-cell release is 8.8 GB and nothing here needs the cells: Replogle
also publishes the per-perturbation pseudobulk (`K562_gwps_raw_bulk_01.h5ad`,
375 MB), which is exactly what a `SignatureLibrary` stores.  Use the *raw* bulk
rather than the normalized one -- the normalized file is z-scored against the
controls and carries infinities where a gene's control standard deviation is
zero, while the raw counts can be put through the same normlog the rest of the
library uses, so the new line is directly comparable to the existing ones.

Rows are keyed `<n>_<SYMBOL>_<protospacer>_<ENSG>`, with 587 `non-targeting`
rows standing in for the control; a target screened with two protospacers gets
one row each and they are averaged in normlog space.
"""

from __future__ import annotations

import argparse
import logging

import h5py
import numpy as np

from vcc2026.library import SignatureLibrary

logger = logging.getLogger(__name__)

CONTROL = "non-targeting"


def _strings(group: h5py.Group, key: str) -> np.ndarray:
    """Read a string column, following the `__categories` indirection if present."""
    node = group[key]
    if isinstance(node, h5py.Group):  # anndata >= 0.8 categorical
        cats = np.asarray([_decode(x) for x in node["categories"][:]])
        return cats[node["codes"][:]]
    values = node[:]
    cats = group.get("__categories")
    if cats is not None and key in cats:  # anndata 0.7 categorical
        table = np.asarray([_decode(x) for x in cats[key][:]])
        return table[values]
    return np.asarray([_decode(x) for x in values])


def _decode(x) -> str:
    return x.decode() if isinstance(x, bytes) else str(x)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--bulk", required=True, help="K562_gwps_raw_bulk_01.h5ad")
    p.add_argument("--library", required=True, help="library to extend (read)")
    p.add_argument("--out", required=True)
    p.add_argument("--line", default="K562gwps", help="name for the new source line")
    p.add_argument(
        "--restrict-to",
        default=None,
        help="one gene symbol per line: keep only these targets from the screen. "
        "The full 9,743 make the transfer model's (targets x genes) accumulator "
        "exceed this machine's memory, and the direct arm only ever reads the "
        "targets actually being predicted, so restricting to the panel keeps the "
        "whole gain at a fraction of the cost",
    )
    p.add_argument(
        "--min-cells",
        type=int,
        default=25,
        help="drop a perturbation pseudobulked from fewer cells than this",
    )
    args = p.parse_args()

    lib = SignatureLibrary.load(args.library)
    gene_index = {str(g): i for i, g in enumerate(lib.genes)}

    with h5py.File(args.bulk, "r") as f:
        x = np.asarray(f["X"][:], dtype=np.float64)
        rows = _strings(f["obs"], "gene_transcript")
        n_cells = np.nan_to_num(f["obs"]["num_cells_unfiltered"][:], nan=0.0)
        src_genes = _strings(f["var"], "gene_name")

    # Same normlog the rest of the library uses, so the new line's deltas are on
    # the same scale as the ones already there.
    totals = x.sum(axis=1)
    totals[totals == 0] = 1.0
    target_sum = float(np.median(totals))
    normed = np.log1p(x * (target_sum / totals)[:, None])

    keep = np.array([g in gene_index for g in src_genes])
    dest = np.array([gene_index[g] for g in src_genes[keep]])
    logger.info(
        "%s: %d perturbations x %d genes (%d shared with the library), target sum %.0f",
        args.line,
        x.shape[0],
        x.shape[1],
        int(keep.sum()),
        target_sum,
    )

    symbols = np.array([r.split("_")[1] for r in rows])
    is_control = symbols == CONTROL
    if not is_control.any():
        raise ValueError(f"{args.bulk}: no {CONTROL!r} rows")
    base_small = normed[is_control][:, keep].mean(axis=0)
    base = np.zeros(lib.genes.size)
    base[dest] = base_small

    lib.baseline[args.line] = base
    lib.target_sum[args.line] = target_sum
    lib.deltas.setdefault(args.line, {})
    lib.n_cells.setdefault(args.line, {})
    lib.n_cells[args.line][CONTROL] = int(n_cells[is_control].sum())

    wanted = None
    if args.restrict_to:
        with open(args.restrict_to) as fh:
            wanted = {ln.strip() for ln in fh if ln.strip() and ln.strip() != "target_gene"}
        logger.info("restricting %s to %d requested targets", args.line, len(wanted))

    kept = dropped = 0
    for symbol in np.unique(symbols[~is_control]):
        if wanted is not None and str(symbol) not in wanted:
            continue
        sel = symbols == symbol
        cells = float(n_cells[sel].sum())
        if cells < args.min_cells:
            dropped += 1
            continue
        # Two protospacers against one gene are two measurements of the same
        # thing; average them rather than letting the gene vote twice.
        delta = np.zeros(lib.genes.size)
        delta[dest] = normed[sel][:, keep].mean(axis=0) - base_small
        lib.deltas[args.line][str(symbol)] = delta.astype(np.float32)
        lib.n_cells[args.line][str(symbol)] = int(cells)
        kept += 1

    logger.info(
        "%s: kept %d targets, dropped %d below %d cells, %d control rows",
        args.line,
        kept,
        dropped,
        args.min_cells,
        int(is_control.sum()),
    )
    lib.save(args.out)
    logger.info("wrote %s with lines %s", args.out, ",".join(lib.lines))


if __name__ == "__main__":
    main()
