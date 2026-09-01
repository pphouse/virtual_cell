"""Leave-one-cell-line-out validation and the alpha sweep.

The 2026 task is zero-shot *across cell lines*, so the only honest local
validation is to hold out an entire source line: build the context from that
line's control cells, refit the model on everything else, and score its
measured perturbations.  Holding out perturbations within a line measures a
different and much easier problem, and will happily tell you a model is good
when it cannot transfer at all.

The sweep exists because of one specific tension in the scoring.  `alpha`
scales every predicted delta.  MAE prefers shrinkage (predict nothing and you
tie the baseline); the L1 discrimination score and pearson_delta prefer honest
magnitudes.  Since the aggregate clips each metric's contribution at zero,
there is no symmetric penalty -- so the optimum is usually well above the value
that MAE alone would pick, and it is worth measuring rather than guessing.
"""

from __future__ import annotations

import logging
from dataclasses import replace

import numpy as np

from .context import CellContext
from .library import SignatureLibrary
from .localeval import (
    BulkPair,
    baseline_pair,
    evaluate,
    score_against_baseline,
)
from .model import ContextTransferModel, ModelConfig

logger = logging.getLogger(__name__)


def holdout_library(library: SignatureLibrary, held_out: str) -> SignatureLibrary:
    """Copy of `library` with one cell line removed entirely."""
    lib = SignatureLibrary(genes=library.genes)
    for line in library.lines:
        if line == held_out:
            continue
        lib.baseline[line] = library.baseline[line]
        lib.deltas[line] = dict(library.deltas[line])
        lib.n_cells[line] = dict(library.n_cells.get(line, {}))
        lib.target_sum[line] = library.target_sum.get(line, 1e4)
    return lib


def evaluate_holdout(
    library: SignatureLibrary,
    held_out: str,
    context: CellContext,
    config: ModelConfig | None = None,
    embeddings: dict[str, np.ndarray] | None = None,
    max_targets: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[dict[str, float], dict[str, float], BulkPair]:
    """Score a config on one held-out line.  Returns (raw metrics, scores, pair)."""
    if held_out not in library.lines:
        raise ValueError(f"{held_out!r} is not a source line: {library.lines}")
    targets = library.targets(held_out)
    if max_targets is not None and len(targets) > max_targets:
        rng = rng or np.random.default_rng(0)
        targets = sorted(rng.choice(targets, size=max_targets, replace=False).tolist())

    trimmed = holdout_library(library, held_out)
    model = ContextTransferModel(config).fit(trimmed, embeddings=embeddings)
    pred = model.predict(targets, context)

    gene_pos = np.array([library.gene_index(g) for g in context.genes])
    if (gene_pos < 0).any():
        raise ValueError("context genes must be a subset of the library gene space")
    real_ctrl = library.baseline[held_out][gene_pos]
    real_means = {
        t: real_ctrl + library.deltas[held_out][t][gene_pos]
        for t in targets
        if t in library.deltas[held_out]
    }
    pair = BulkPair(
        perts=list(real_means),
        genes=context.genes,
        pred=context.mu[None, :] + pred.delta[[targets.index(t) for t in real_means]],
        real=np.vstack(list(real_means.values())),
        pred_ctrl=context.mu,
        real_ctrl=real_ctrl,
    )
    metrics = evaluate(pair)
    scores = score_against_baseline(metrics, evaluate(baseline_pair(pair)))
    return metrics, scores, pair


def sweep(
    library: SignatureLibrary,
    held_out: str,
    context: CellContext,
    param: str,
    values: list,
    config: ModelConfig | None = None,
    **kwargs,
) -> list[dict]:
    """Score one config parameter across `values` on a held-out line."""
    base = config or ModelConfig()
    rows = []
    for v in values:
        cfg = replace(base, **{param: v})
        _, scores, _ = evaluate_holdout(library, held_out, context, config=cfg, **kwargs)
        row = {param: v, **scores}
        logger.info("%s=%s -> avg_score %.4f", param, v, row["avg_score"])
        rows.append(row)
    return rows


def best_of(rows: list[dict], param: str) -> object:
    return max(rows, key=lambda r: r["avg_score"])[param]
