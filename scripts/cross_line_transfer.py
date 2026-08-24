#!/usr/bin/env python
"""Does a signature measured in one cell line transfer to another?

This is the assumption the whole approach rests on and, until a second cell
line was available, the one thing that could not be checked. The transfer
operator claims that what survives a context switch is the *fold change*, not
the additive delta. Here that claim is tested directly: build the library from
one cell line only, predict another line's measured targets, and score.

Everything is restricted to the genes both lines measure, so a source with
wider coverage cannot look better for that reason alone, and the within-line
case is run on the same restricted gene set as the reference to compare
against. The gap between them is the cost of crossing contexts.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from offline_score_2025 import de_metrics, pds_cosine  # noqa: E402

from vcc2026.context import CellContext  # noqa: E402
from vcc2026.library import SignatureLibrary  # noqa: E402
from vcc2026.model import ContextTransferModel, ModelConfig  # noqa: E402

logger = logging.getLogger("crossline")


def subset(lib: SignatureLibrary, lines: list[str]) -> SignatureLibrary:
    out = SignatureLibrary(genes=lib.genes)
    for line in lines:
        out.baseline[line] = lib.baseline[line]
        out.deltas[line] = dict(lib.deltas[line])
        out.n_cells[line] = dict(lib.n_cells.get(line, {}))
        out.target_sum[line] = lib.target_sum.get(line, 1e4)
    return out


def run(
    lib: SignatureLibrary, source: list[str], target: str, cfg: ModelConfig, features, genes_mask
):
    targets = [t for t in lib.targets(target) if lib.gene_index(t) is not None]
    truth = np.vstack([lib.deltas[target][t] for t in targets])
    pos = np.array([lib.gene_index(t) for t in targets])
    rows = np.arange(len(targets))

    mu = lib.baseline[target]
    ctx = CellContext(
        name=target,
        genes=lib.genes,
        control=np.repeat(mu[None, :], 2, axis=0),
        target_sum=lib.target_sum.get(target, 1e4),
    )
    model = ContextTransferModel(cfg).fit(subset(lib, source), features=features)
    pred = model.predict(targets, ctx).delta.copy()
    truth = truth.copy()
    pred[rows, pos] = 0.0
    truth[rows, pos] = 0.0

    keep = genes_mask.copy()
    keep[pos] = False
    pred, truth = pred[:, keep], truth[:, keep]
    out = {"n_targets": len(targets), "n_genes": int(keep.sum()), "pds": pds_cosine(pred, truth)}
    out.update(de_metrics(pred, truth))
    return out


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--library", required=True)
    p.add_argument("--string-adj", required=True)
    p.add_argument(
        "--case",
        action="append",
        required=True,
        metavar="SRC1+SRC2->TGT",
        help="repeatable, e.g. H1+H1val->K562",
    )
    args = p.parse_args()

    lib = SignatureLibrary.load(args.library)
    import scipy.sparse as sp

    from vcc2026.network import string_features

    adj = sp.load_npz(args.string_adj)
    features = string_features(adj, lib.genes, subset=sorted(lib.targets()))
    cfg = ModelConfig(
        alpha=0.85, magnitude_gamma=1.0, n_components=100, neighbour_power=2.0, n_neighbours=100
    )

    print(
        f"{'case':30s} {'n':>5s} {'genes':>6s} {'pds':>7s} {'agree':>7s} "
        f"{'yield':>7s} {'jac':>7s} {'nmae':>7s}"
    )
    for case in args.case:
        src, tgt = case.split("->")
        source = src.split("+")
        # Only genes every line involved actually measured.
        mask = np.ones(lib.genes.size, dtype=bool)
        for line in [*source, tgt]:
            mask &= lib.baseline[line] > 0.05
        r = run(lib, source, tgt, cfg, features, mask)
        print(
            f"{case:30s} {r['n_targets']:5d} {r['n_genes']:6d} {r['pds']:7.4f} "
            f"{r['direction_agreement']:7.4f} {r['de_yield']:7.4f} {r['jac']:7.4f} {r['nmae']:7.4f}"
        )


if __name__ == "__main__":
    main()
