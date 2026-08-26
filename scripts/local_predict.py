#!/usr/bin/env python
"""Emit a prediction for a local reference, in the shape a submission takes.

The leaderboard scores emitted cells, not fold changes, and the gap between the
two is where three of this project's four measured emission bugs lived.  So a
candidate is checked the same way the competition checks it: cells are drawn
from the context's own controls, the model's fold changes are applied to them,
and the result is scored by `cell_eval2`'s own differential-expression code.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_local_context import read_obs_column, read_var_names, stream_rows  # noqa: E402

from vcc2026.budget import BudgetConfig, BudgetedPredictor  # noqa: E402
from vcc2026.counts import context_mean_proportions, emit_counts  # noqa: E402
from vcc2026.library import SignatureLibrary  # noqa: E402
from vcc2026.writer import SparseH5adWriter  # noqa: E402

logger = logging.getLogger("localpred")


def generic_signal(lib: SignatureLibrary, lines: list[str], genes: np.ndarray) -> np.ndarray:
    """Mean knockdown response over a screen's targets, on the submission's gene axis.

    Only the sign and the relative size are used -- the emitted magnitude comes
    from the significance threshold -- so the library's normalized-log delta is
    carried across as it stands rather than rebased.
    """
    per_line = [
        np.vstack([lib.deltas[line][t] for t in lib.targets(line)]).mean(0) for line in lines
    ]
    mean = np.mean(per_line, axis=0)
    index = {str(g): i for i, g in enumerate(lib.genes)}
    out = np.zeros(genes.size)
    for i, g in enumerate(genes):
        j = index.get(str(g))
        if j is not None:
            out[i] = mean[j]
    return out


class _Specific:
    """The transfer model's delta for one context, as a ranking signal.

    The library lives on the 2026 gene axis and a local reference on its own, so
    the delta is mapped back to the reference's genes on the way out.
    """

    def __init__(self, model, ctx, columns) -> None:
        self._model, self._ctx, self._columns = model, ctx, columns

    def predict_log2fc(self, targets: list[str]):
        delta = self._model.predict(targets, self._ctx).delta
        out = np.zeros((len(targets), self._columns.size))
        have = self._columns >= 0
        out[:, have] = delta[:, self._columns[have]]
        return out


def specific_signal(lib, lines, string_adj, shrink, targets, control, genes):
    """Fit the transfer model on `lines` and expose its delta on `genes`.

    The control block is only used for its mean profile, which is what the model
    reads as the context; it is the same block the significance threshold and the
    propensity come from, so nothing outside the reference enters the prediction.
    """
    import scipy.sparse as sp_

    from vcc2026.context import CellContext
    from vcc2026.model import ContextTransferModel, ModelConfig
    from vcc2026.network import string_features
    from vcc2026.normalize import normlog

    sub = SignatureLibrary(genes=lib.genes)
    for line in lines:
        sub.baseline[line] = lib.baseline[line]
        sub.deltas[line] = dict(lib.deltas[line])
        sub.n_cells[line] = dict(lib.n_cells.get(line, {}))
        sub.target_sum[line] = lib.target_sum.get(line, 1e4)
    feats = string_features(
        sp_.load_npz(string_adj), lib.genes, subset=sorted(set(targets) | set(sub.targets()))
    )
    model = ContextTransferModel(
        ModelConfig(
            alpha=1.0,
            n_components=30,
            n_neighbours=100,
            neighbour_power=2.0,
            shared_shrink=shrink,
        )
    ).fit(sub, features=feats)
    x, target_sum = normlog(control)
    mu_local = np.asarray(x.mean(axis=0)).ravel()
    index = {str(g): i for i, g in enumerate(genes)}
    mu = np.array([mu_local[index[str(g)]] if str(g) in index else 0.0 for g in lib.genes])
    ctx = CellContext(
        name="local",
        genes=lib.genes,
        control=np.repeat(mu[None, :], 2, axis=0),
        target_sum=target_sum,
    )
    lg = {str(g): i for i, g in enumerate(lib.genes)}
    columns = np.array([lg.get(str(g), -1) for g in genes])
    return _Specific(model, ctx, columns)


def _specific_signal(lib, args, targets, control, genes):
    lines = [x.strip() for x in args.specific_lines.split(",")]
    return specific_signal(
        lib, lines, args.string_adj, args.specific_shrink, targets, control, genes
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--library", required=True)
    p.add_argument("--lines", default="K562,RPE1")
    p.add_argument(
        "--specific-lines",
        default=None,
        help="fit the transfer model on these lines and rank by its delta instead "
        "of the generic response; needs --string-adj",
    )
    p.add_argument("--string-adj", default=None)
    p.add_argument("--gene-positions", default=None)
    p.add_argument("--proximity-weight", type=float, default=0.0)
    p.add_argument("--specific-shrink", type=float, default=0.0)
    p.add_argument("--n-calls", type=int, default=BudgetConfig.n_calls)
    p.add_argument("--margin", type=float, default=BudgetConfig.margin)
    p.add_argument("--top-margin", type=float, default=BudgetConfig.top_margin)
    p.add_argument("--pseudocount", type=float, default=1.0)
    p.add_argument("--cells-per-pert", type=int, default=400)
    p.add_argument("--control-cells", type=int, default=0, help="0 = as many as the reference")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    labels = read_obs_column(args.reference, "target")
    genes = read_var_names(args.reference, None)
    targets = sorted({x for x in labels if x != "non-targeting"})
    ctrl_rows = np.flatnonzero(labels == "non-targeting")
    logger.info("%d targets, %d control cells, %d genes", len(targets), ctrl_rows.size, genes.size)

    blocks = [blk for _, blk in stream_rows(args.reference, ctrl_rows, block=8192)]
    control = sp.vstack(blocks, format="csr").astype(np.float32)
    del blocks

    lib = SignatureLibrary.load(args.library)
    generic = generic_signal(lib, [x.strip() for x in args.lines.split(",")], genes)
    specific = None
    if args.specific_lines:
        specific = _specific_signal(lib, args, targets, control, genes)
    proximity = None
    if args.gene_positions and args.proximity_weight:
        from vcc2026.proximity import ProximitySignal, load_gene_positions

        proximity = ProximitySignal(genes, load_gene_positions(args.gene_positions))
    model = BudgetedPredictor(
        BudgetConfig(
            n_calls=args.n_calls,
            margin=args.margin,
            top_margin=args.top_margin,
            generic_weight=0.0 if args.specific_lines else 1.0,
            specific_weight=1.0 if args.specific_lines else 0.0,
            proximity_weight=args.proximity_weight,
            seed=args.seed,
        ),
        generic,
        specific=specific,
        proximity=proximity,
    ).fit(control, genes)

    proportions = context_mean_proportions(control)
    n_ctrl = args.control_cells or ctrl_rows.size
    groups = [("non-targeting", n_ctrl)] + [(t, args.cells_per_pert) for t in targets]
    obs = pd.DataFrame(
        {"target": pd.Categorical(np.concatenate([np.repeat(n, k) for n, k in groups]))},
        index=np.arange(sum(k for _, k in groups)).astype(str),
    )

    lfc = model.predict_lfc(targets)
    logger.info(
        "call set: %d genes per target (median |log2fc| %.3f)",
        int((lfc[0] != 0).sum()),
        float(np.median(np.abs(lfc[0][lfc[0] != 0])) / np.log(2.0)),
    )

    with SparseH5adWriter(args.out, obs=obs, var_names=genes) as writer:
        for name, k in groups:
            pick = rng.integers(0, control.shape[0], size=k)
            base = control[pick]
            if name == "non-targeting":
                writer.append(base)
                continue
            row = lfc[targets.index(name)]
            writer.append(
                emit_counts(base, proportions, row, args.pseudocount, rng).astype(np.float32)
            )
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
