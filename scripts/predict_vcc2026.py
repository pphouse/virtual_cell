#!/usr/bin/env python
"""Build a 2026 Virtual Cell Challenge submission.

    python scripts/predict_vcc2026.py --bundle ~/vcc --out out/submission.h5ad

With `--n-calls` it uses the budgeted predictor: a ranked call set of that size
per perturbation, everything else passed through unchanged.  With `--library`
and no budget it uses the full Context-Conditioned Delta Transfer model,
transferring measured signatures from source screens into each held-out
context.  Without either it falls back to the control-only predictor, which
knows nothing but the target line's unperturbed cells -- the on-target knockdown
and, if `--trans-beta` is set, a co-expression trans signature.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from vcc2026.challenge import Bundle
from vcc2026.coexpression import ControlOnlyConfig, ControlOnlyPredictor
from vcc2026.model import ModelConfig
from vcc2026.submission import SubmissionConfig, build_submission

logger = logging.getLogger("predict")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--bundle", required=True, help="directory holding the controls bundle")
    p.add_argument("--out", required=True)
    p.add_argument("--vcc", default=None, help="also package the result as a .vcc for upload")
    p.add_argument("--library", default=None, help="SignatureLibrary .npz for the transfer model")
    p.add_argument("--embeddings", default=None)
    p.add_argument(
        "--string-adj",
        default=None,
        help="STRING adjacency .npz: the gene similarity used for unseen targets",
    )
    p.add_argument(
        "--trans-similarity-floor",
        type=float,
        default=0.0,
        help="below this best-neighbour similarity, predict no trans effect at all",
    )
    p.add_argument("--knockdown-residual", type=float, default=ControlOnlyConfig.knockdown_residual)
    p.add_argument("--trans-beta", type=float, default=ControlOnlyConfig.trans_beta)
    p.add_argument(
        "--n-components",
        type=int,
        default=None,
        help="control-only: PCA components; transfer: response programmes",
    )
    p.add_argument("--alpha", type=float, default=1.0, help="transfer model: global effect scale")
    p.add_argument("--n-neighbours", type=int, default=ModelConfig.n_neighbours)
    p.add_argument("--magnitude-gamma", type=float, default=ModelConfig.magnitude_gamma)
    p.add_argument("--neighbour-power", type=float, default=ModelConfig.neighbour_power)
    p.add_argument("--shared-shrink", type=float, default=ModelConfig.shared_shrink)
    p.add_argument(
        "--knockdown-lines",
        default=None,
        help="comma-separated source lines to estimate knockdown efficiency from; "
        "it is a property of the reagent, not the biology, so it does not "
        "transfer across protocols the way a trans signature does",
    )
    p.add_argument("--pseudocount", type=float, default=SubmissionConfig.pseudocount)
    p.add_argument(
        "--min-abs-lfc",
        type=float,
        default=SubmissionConfig.min_abs_lfc,
        help="predicted |log fold change| below this is treated as no claim",
    )
    p.add_argument(
        "--non-responder-fraction", type=float, default=SubmissionConfig.non_responder_fraction
    )
    p.add_argument(
        "--response-shape",
        type=float,
        default=SubmissionConfig.response_shape,
        help="Gamma shape for per-cell response spread; inf = deterministic full effect",
    )
    p.add_argument("--contexts", default=None, help="comma-separated subset (debugging only)")
    p.add_argument(
        "--limit-perts",
        type=int,
        default=None,
        help="debugging only; produces an invalid submission",
    )
    p.add_argument(
        "--n-calls",
        type=int,
        default=0,
        help="budgeted mode: declare this many genes per perturbation as responding",
    )
    p.add_argument("--margin", type=float, default=1.35, help="budgeted mode: magnitude margin")
    p.add_argument(
        "--generic-weight",
        type=float,
        default=1.0,
        help="budgeted mode: weight on the generic knockdown response",
    )
    p.add_argument(
        "--magnitude-gamma",
        type=float,
        default=0.0,
        help="budgeted mode: let the target's own ranking score set the size of its "
        "calls, not just their signs; 0 keeps the size a function of expression alone",
    )
    p.add_argument("--max-magnitude-scale", type=float, default=3.0)
    p.add_argument(
        "--specific-weight",
        type=float,
        default=0.0,
        help="budgeted mode: weight on the transfer model's target-specific delta "
        "(needs --library and --string-adj)",
    )
    p.add_argument("--top-margin", type=float, default=2.0, help="budgeted mode: head margin")
    p.add_argument(
        "--gene-positions",
        default=None,
        help="budgeted mode: {symbol: [chromosome, midpoint]} JSON, enabling the "
        "CRISPRi proximity term",
    )
    p.add_argument(
        "--proximity-weight",
        type=float,
        default=0.0,
        help="budgeted mode: how hard to pull a target's genomic neighbours toward "
        "silence, relative to the mean |signal|",
    )
    p.add_argument(
        "--generic-lines",
        default=None,
        help="budgeted mode: library lines the generic response is averaged over",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    bundle = Bundle.load(args.bundle)
    if args.contexts:
        keep = {c.strip() for c in args.contexts.split(",")}
        bundle.context_files = {k: v for k, v in bundle.context_files.items() if k in keep}
    if args.limit_perts:
        bundle.perturbations = bundle.perturbations[: args.limit_perts]
        logger.warning("--limit-perts set: this file is for timing only, not a valid submission")

    missing = (~bundle.check_targets_predictable()).sum()
    if missing:
        logger.warning("%d official targets are outside the submission gene space", missing)

    if args.n_calls:
        from vcc2026.budget import BudgetConfig, BudgetedPredictor
        from vcc2026.library import SignatureLibrary

        library = SignatureLibrary.load(args.library)
        lines = (
            [x.strip() for x in args.generic_lines.split(",")]
            if args.generic_lines
            else list(library.lines)
        )
        generic = _generic_signal(library, lines, bundle.genes)
        logger.info("generic response averaged over %s", lines)
        cfg = BudgetConfig(
            n_calls=args.n_calls,
            margin=args.margin,
            top_margin=args.top_margin,
            generic_weight=args.generic_weight,
            specific_weight=args.specific_weight,
            proximity_weight=args.proximity_weight,
            magnitude_gamma=args.magnitude_gamma,
            max_magnitude_scale=args.max_magnitude_scale,
            seed=args.seed,
        )
        transfer = _fit_transfer(args, bundle, library) if args.specific_weight else None
        proximity = None
        if args.gene_positions and args.proximity_weight:
            from vcc2026.proximity import ProximitySignal, load_gene_positions

            proximity = ProximitySignal(bundle.genes, load_gene_positions(args.gene_positions))

        def predictor_for(context: str, counts: sp.csr_matrix, genes: np.ndarray):
            specific = (
                _SpecificSignal(transfer, context, counts, genes) if transfer is not None else None
            )
            return BudgetedPredictor(cfg, generic, specific=specific, proximity=proximity).fit(
                counts, genes
            )

    elif args.library:
        from vcc2026.features import load_embeddings
        from vcc2026.library import SignatureLibrary
        from vcc2026.model import ContextTransferModel

        library = SignatureLibrary.load(args.library)
        embeddings = load_embeddings(args.embeddings) if args.embeddings else None

        features = None
        if args.string_adj:
            import scipy.sparse as sp_

            from vcc2026.network import string_features

            adj = sp_.load_npz(args.string_adj)
            # Only the rows actually queried are needed: the targets being
            # predicted and the targets the library can transfer from. The full
            # profile matrix would be 1.4 GB dense for no benefit.
            needed = sorted(set(map(str, bundle.perturbations)) | set(library.targets()))
            features = string_features(adj, bundle.genes, subset=needed)

        model = ContextTransferModel(
            ModelConfig(
                alpha=args.alpha,
                n_components=args.n_components or ModelConfig.n_components,
                n_neighbours=args.n_neighbours,
                magnitude_gamma=args.magnitude_gamma,
                neighbour_power=args.neighbour_power,
                shared_shrink=args.shared_shrink,
                knockdown_lines=(
                    tuple(x.strip() for x in args.knockdown_lines.split(","))
                    if args.knockdown_lines
                    else None
                ),
                trans_similarity_floor=args.trans_similarity_floor,
                seed=args.seed,
            )
        ).fit(library, embeddings=embeddings, features=features)

        def predictor_for(context: str, counts: sp.csr_matrix, genes: np.ndarray):
            return _TransferAdapter(model, context, counts, genes)
    else:
        logger.info(
            "no --library: using the control-only predictor "
            "(knockdown residual %.2f, trans beta %.3f)",
            args.knockdown_residual,
            args.trans_beta,
        )

        def predictor_for(context: str, counts: sp.csr_matrix, genes: np.ndarray):
            cfg = ControlOnlyConfig(
                knockdown_residual=args.knockdown_residual,
                trans_beta=args.trans_beta,
                n_components=args.n_components or ControlOnlyConfig.n_components,
                seed=args.seed,
            )
            return ControlOnlyPredictor(cfg).fit(counts, genes)

    out = build_submission(
        bundle,
        predictor_for,
        args.out,
        SubmissionConfig(
            pseudocount=args.pseudocount,
            min_abs_lfc=args.min_abs_lfc,
            non_responder_fraction=args.non_responder_fraction,
            response_shape=args.response_shape,
            seed=args.seed,
        ),
    )
    meta = {
        "bundle": str(bundle.root),
        "contexts": bundle.contexts,
        "n_perturbations": int(bundle.perturbations.size),
        "cells_per_pert": bundle.cells_per_pert,
        "model": "budgeted" if args.n_calls else ("transfer" if args.library else "control-only"),
        "n_calls": args.n_calls,
        "margin": args.margin,
        "generic_weight": args.generic_weight,
        "specific_weight": args.specific_weight,
        "proximity_weight": args.proximity_weight,
        "magnitude_gamma": args.magnitude_gamma,
        "knockdown_residual": args.knockdown_residual,
        "trans_beta": args.trans_beta,
        "alpha": args.alpha,
        "pseudocount": args.pseudocount,
        "min_abs_lfc": args.min_abs_lfc,
        "trans_similarity_floor": args.trans_similarity_floor,
        "seed": args.seed,
    }
    Path(str(out) + ".json").write_text(json.dumps(meta, indent=2))
    logger.info("submission written: %s", out)

    if args.vcc:
        from vcc2026.vccfile import write_vcc

        write_vcc(out, args.vcc)


def _fit_transfer(args, bundle, library):
    """The CCDT model, fitted for use as the budgeted predictor's specific signal."""
    import scipy.sparse as sp_

    from vcc2026.model import ContextTransferModel
    from vcc2026.network import string_features

    adj = sp_.load_npz(args.string_adj)
    needed = sorted(set(map(str, bundle.perturbations)) | set(library.targets()))
    features = string_features(adj, bundle.genes, subset=needed)
    return ContextTransferModel(
        ModelConfig(
            alpha=args.alpha,
            n_components=args.n_components or ModelConfig.n_components,
            n_neighbours=args.n_neighbours,
            magnitude_gamma=args.magnitude_gamma,
            neighbour_power=args.neighbour_power,
            shared_shrink=args.shared_shrink,
            trans_similarity_floor=args.trans_similarity_floor,
            seed=args.seed,
        )
    ).fit(library, features=features)


class _SpecificSignal:
    """The transfer model's delta, as a ranking signal rather than a fold change.

    Only the sign and the relative size are read, so the normalized-log delta is
    handed over as it stands; the emitted magnitude comes from the significance
    threshold instead.
    """

    def __init__(self, model, context: str, counts: sp.csr_matrix, genes: np.ndarray) -> None:
        from vcc2026.context import CellContext
        from vcc2026.normalize import normlog

        x, target_sum = normlog(counts)
        self._ctx = CellContext(name=context, genes=genes, control=x, target_sum=target_sum)
        self._model = model

    def predict_log2fc(self, targets: list[str]) -> np.ndarray:
        return self._model.predict(targets, self._ctx).delta


def _generic_signal(library, lines: list[str], genes: np.ndarray) -> np.ndarray:
    """Mean knockdown response over the named screens, on the submission gene axis."""
    per_line = [
        np.vstack([library.deltas[line][t] for t in library.targets(line)]).mean(0)
        for line in lines
    ]
    mean = np.mean(per_line, axis=0)
    index = {str(g): i for i, g in enumerate(library.genes)}
    out = np.zeros(genes.size)
    for i, gene in enumerate(genes):
        j = index.get(str(gene))
        if j is not None:
            out[i] = mean[j]
    return out


class _TransferAdapter:
    """Expose the normlog CCDT model through the counts pipeline's LFC interface."""

    def __init__(self, model, context: str, counts: sp.csr_matrix, genes: np.ndarray) -> None:
        from vcc2026.context import CellContext
        from vcc2026.normalize import normlog

        x, target_sum = normlog(counts)
        self._ctx = CellContext(name=context, genes=genes, control=x, target_sum=target_sum)
        self._model = model

    def predict_lfc(self, targets: list[str]) -> np.ndarray:
        pred = self._model.predict(targets, self._ctx)
        mu = self._ctx.mu
        # normlog delta -> natural-log fold change on the linear scale, which is
        # the space the counts sampler multiplies in. The half-count pseudo-count
        # is deliberate: below about one count at this depth a fold change is not
        # identifiable, and shrinking it there is the right behaviour.
        e0 = np.expm1(np.clip(mu, 0.0, None))
        e1 = np.expm1(np.clip(mu[None, :] + pred.delta, 0.0, None))
        eps = 0.5
        lfc = np.log((e1 + eps) / (e0 + eps)).astype(np.float32)

        # ... except on the target gene itself, where the knockdown is the one
        # thing we are sure of. Routing it through the same shrinkage would
        # quietly halve every knockdown the model claims to make -- the same
        # failure the counts pseudo-count caused (docs/05 §3).
        residual = self._model.knockdown
        for i, target in enumerate(targets):
            j = self._ctx.gene_index(target)
            if j is not None and mu[j] > 0.02:
                lfc[i, j] = float(np.log(max(residual.residual_fraction(target), 1e-6)))
        return lfc


if __name__ == "__main__":
    main()
