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


def specific_signal(
    lib,
    lines,
    string_adj,
    shrink,
    targets,
    control,
    genes,
    model_config=None,
    features_npz=None,
    feature_blend=None,
    feature_topk=None,
    holdout_targets=None,
):
    """Fit the transfer model on `lines` and expose its delta on `genes`.

    The control block is only used for its mean profile, which is what the model
    reads as the context; it is the same block the significance threshold and the
    propensity come from, so nothing outside the reference enters the prediction.

    `model_config` defaults to `ModelConfig()` -- the same defaults the submission
    builder uses when it is not told otherwise.  It used to be a hardcoded variant
    (100 neighbours at power 2 against the shipped 25 at power 3), which quietly
    made every offline measurement a measurement of a different model.

    `holdout_targets` drops those genes from every source line, so no target being
    predicted is measured anywhere in the library.  Holding out the scored *line*
    is not enough: the panel's targets come from Replogle, so scoring K562 leaves
    RPE1 holding a signature for all 300 of them, `w_obs > 0`, and the model
    answers from the target's own rebased measurement instead of from the
    similarity.  The real panel's 300 targets appear in no source screen at all,
    so there every prediction comes from the neighbour arm.  Without this the
    panel scores a regime the competition does not have, and dilutes any change
    to the similarity by the confidence weight -- which is how it called the
    co-essentiality blend at +0.0005 against a measured +0.0062 (`docs/08` 23).
    """
    from dataclasses import replace

    import scipy.sparse as sp_

    from vcc2026.context import CellContext
    from vcc2026.model import ContextTransferModel, ModelConfig
    from vcc2026.network import string_features
    from vcc2026.normalize import normlog

    drop = set(holdout_targets or ())
    sub = SignatureLibrary(genes=lib.genes)
    for line in lines:
        sub.baseline[line] = lib.baseline[line]
        sub.deltas[line] = {t: d for t, d in lib.deltas[line].items() if t not in drop}
        sub.n_cells[line] = {
            t: n for t, n in lib.n_cells.get(line, {}).items() if t not in drop
        }
        sub.target_sum[line] = lib.target_sum.get(line, 1e4)
    if drop:
        kept = sum(len(sub.deltas[ln]) for ln in lines)
        held = sum(len(lib.deltas[ln]) for ln in lines) - kept
        logger.info(
            "held out %d target signatures from the source lines (%d left as neighbours)",
            held,
            kept,
        )
    subset = sorted(set(targets) | set(sub.targets()))
    # --features/--feature-blend may be given more than once, so a run can stack
    # several kinds of similarity: each extra block is blended onto whatever is
    # built so far, at its own weight.  A single --features with no weight still
    # means "replace STRING outright", which is how the co-essentiality-only
    # arms in `docs/08` 20 were measured.
    paths = [features_npz] if isinstance(features_npz, str) else list(features_npz or ())
    weights = [feature_blend] if not isinstance(feature_blend, list) else list(feature_blend)
    if paths and len(weights) not in (0, len(paths)):
        raise ValueError(
            f"{len(paths)} feature files but {len(weights)} blend weights -- "
            "give one weight per file, or none at all to replace STRING"
        )
    if paths:
        from vcc2026.coessentiality import blend, load_features, neighbourhood

        extra = []
        for path in paths:
            block = load_features(path, subset=subset)
            extra.append(neighbourhood(block, feature_topk) if feature_topk else block)
        if not weights or weights[0] is None:
            if len(extra) > 1:
                raise ValueError("blend weights are required when stacking several --features")
            feats = extra[0]
        else:
            feats = string_features(sp_.load_npz(string_adj), lib.genes, subset=subset)
            for block, weight in zip(extra, weights, strict=True):
                feats = blend(feats, block, weight)
    else:
        feats = string_features(sp_.load_npz(string_adj), lib.genes, subset=subset)
    cfg = replace(model_config or ModelConfig(), shared_shrink=shrink)
    logger.info(
        "transfer model: %d components, %d neighbours, power %.1f, shrink %.2f",
        cfg.n_components,
        cfg.n_neighbours,
        cfg.neighbour_power,
        cfg.shared_shrink,
    )
    model = ContextTransferModel(cfg).fit(sub, features=feats)
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
    from vcc2026.model import ModelConfig

    lines = [x.strip() for x in args.specific_lines.split(",")]
    d = ModelConfig()
    cfg = ModelConfig(
        alpha=args.alpha,
        n_components=args.n_components or d.n_components,
        n_neighbours=args.n_neighbours or d.n_neighbours,
        neighbour_power=(
            d.neighbour_power if args.neighbour_power is None else args.neighbour_power
        ),
        confidence_prior=(
            d.confidence_prior if args.confidence_prior is None else args.confidence_prior
        ),
    )
    return specific_signal(
        lib,
        lines,
        args.string_adj,
        args.specific_shrink,
        targets,
        control,
        genes,
        cfg,
        features_npz=args.features,
        feature_blend=args.feature_blend,
        feature_topk=args.feature_topk,
        holdout_targets=targets if args.holdout_target_signatures else None,
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
    p.add_argument(
        "--generic-weight",
        type=float,
        default=None,
        help="mix the generic response in beside the target-specific one; each "
        "component is put on unit scale first, so this is a ratio. Defaults to 1 "
        "without --specific-lines and 0 with it.",
    )
    p.add_argument("--gene-positions", default=None)
    p.add_argument("--proximity-weight", type=float, default=0.0)
    p.add_argument(
        "--no-magnitude-floor",
        action="store_true",
        help="let a predicted size fall below the significance magnitude, trading "
        "the DE-set members for log-fold-change accuracy",
    )
    p.add_argument(
        "--predicted-magnitude",
        type=float,
        default=0.0,
        help="how much of a call's size comes from the model's own predicted fold "
        "change rather than from the gene's expression (0 = expression alone)",
    )
    p.add_argument("--call-size-gamma", type=float, default=0.0)
    p.add_argument("--max-call-scale", type=float, default=3.0)
    p.add_argument("--propensity-power", type=float, default=1.0)
    p.add_argument("--specific-shrink", type=float, default=0.0)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument(
        "--confidence-prior",
        type=float,
        default=None,
        help="how much measured weight a target needs before its own rebased "
        "signature outweighs the neighbour average (b = w/(w+prior)); it was set "
        "when no production target had a signature at all",
    )
    p.add_argument("--n-components", type=int, default=None)
    p.add_argument("--n-neighbours", type=int, default=None)
    p.add_argument("--neighbour-power", type=float, default=None)
    p.add_argument(
        "--features",
        action="append",
        default=None,
        help="a gene-similarity features npz (scripts/build_coessentiality.py, "
        "scripts/build_tissue_features.py); repeat to stack several kinds. "
        "Alone and without a weight it replaces the STRING neighbourhood profile",
    )
    p.add_argument(
        "--feature-blend",
        type=float,
        action="append",
        default=None,
        help="keep STRING and add each --features block at this weight; give one "
        "weight per --features. Unset means a lone --features replaces STRING outright",
    )
    p.add_argument(
        "--holdout-target-signatures",
        action="store_true",
        help="drop the scored targets' own signatures from the source lines, so every "
        "prediction comes from the similarity as it does on the real panel, whose 300 "
        "targets appear in no source screen; without this the panel answers from the "
        "target's own rebased measurement and dilutes any similarity change",
    )
    p.add_argument(
        "--feature-topk",
        type=int,
        default=None,
        help="compare co-essentiality partner sets instead of raw fitness profiles, "
        "keeping this many partners per gene",
    )
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
            generic_weight=(
                args.generic_weight
                if args.generic_weight is not None
                else (0.0 if args.specific_lines else 1.0)
            ),
            specific_weight=1.0 if args.specific_lines else 0.0,
            proximity_weight=args.proximity_weight,
            call_size_gamma=args.call_size_gamma,
            predicted_magnitude=args.predicted_magnitude,
            predicted_magnitude_floor=not args.no_magnitude_floor,
            max_call_scale=args.max_call_scale,
            propensity_power=args.propensity_power,
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
