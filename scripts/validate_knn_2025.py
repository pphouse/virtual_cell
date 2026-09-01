#!/usr/bin/env python
"""Can a gene feature predict the response to knocking down a gene nobody perturbed?

This is the assumption the whole transfer model rests on.  Only 13 of the 2026
challenge's 300 targets appear in any public CRISPRi screen, so 287 of them have
no signature to transfer and must be reached by similarity: "knocking down A
should look like knocking down B, because A and B look alike".

The 2025 release measures 150 knockdowns in one cell line, which is enough to
test that claim directly.  Hold out one target at a time, predict its signature
as a similarity-weighted average of the other 149, and score the prediction
against what was actually measured.

Three predictors are compared, and the comparison is the point:

* **mean signature** -- ignore the target entirely and predict the average of
  all other knockdowns.  Perturbations share a few stress and growth programmes,
  so this is a genuinely strong baseline and not a straw man.
* **co-expression kNN** -- weight the other targets by how similarly they
  co-vary with the transcriptome across control cells.
* **oracle kNN** -- weight by the true similarity of the measured signatures.
  Unusable in the challenge, but it bounds what any similarity-weighted
  predictor could achieve, so it separates "the features are bad" from "this
  shape of model cannot work".

Scored two ways.  Correlation says whether the shape is right; a discrimination
rank -- is each prediction closest to its own truth among all 150? -- says
whether the prediction is *specific* to that perturbation, which is what the
challenge's discrimination metric measures and what predicting the mean
signature can never achieve.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from vcc2026.coexpression import ControlOnlyConfig, ControlOnlyPredictor
from vcc2026.library import SignatureLibrary
from vcc2026.programs import neighbour_weights

logger = logging.getLogger("knn2025")


def discrimination(pred: np.ndarray, truth: np.ndarray) -> float:
    """Mean normalised rank: 1.0 if every prediction matches its own truth first."""
    n = pred.shape[0]
    scores = np.zeros(n)
    for i in range(n):
        d = np.abs(truth - pred[i][None, :]).sum(axis=1)
        rank = int(np.flatnonzero(np.argsort(d) == i)[0])
        scores[i] = 1.0 - rank / n
    return float(scores.mean())


def row_correlation(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    a = pred - pred.mean(axis=1, keepdims=True)
    b = truth - truth.mean(axis=1, keepdims=True)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nan_to_num(
            np.where(den > 0, (a * b).sum(axis=1), 0.0) / np.where(den > 0, den, 1.0)
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--library", required=True)
    p.add_argument("--screen", required=True)
    p.add_argument("--genes", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=25)
    p.add_argument("--power", type=float, default=3.0)
    p.add_argument("--n-components", type=int, default=50)
    p.add_argument("--string-adj", default=None, help="STRING adjacency .npz to score as well")
    args = p.parse_args()

    genes = pd.read_csv(args.genes).iloc[:, 0].astype(str).to_numpy()
    lib = SignatureLibrary.load(args.library)
    line = lib.lines[0]
    targets = [t for t in lib.targets(line) if lib.gene_index(t) is not None]
    deltas = np.vstack([lib.deltas[line][t] for t in targets])
    pos = np.array([lib.gene_index(t) for t in targets])
    base = lib.baseline[line]

    # Score on expressed genes only, and never on the target genes themselves:
    # the on-target drop is modelled separately and would otherwise dominate
    # every correlation with something the similarity model never predicted.
    keep = base > 0.05
    keep[pos] = False
    truth = deltas[:, keep]
    logger.info("%d targets, scoring on %d genes", len(targets), int(keep.sum()))

    from scripts.validate_on_2025 import load_control_cells  # noqa: E402

    controls = load_control_cells(args.screen, genes)
    model = ControlOnlyPredictor(ControlOnlyConfig(n_components=args.n_components)).fit(
        controls, np.asarray(genes, dtype=object)
    )
    coexpr_sim = np.vstack(
        [
            model.program_correlation(t)[np.array([lib.gene_index(x) for x in targets])]
            for t in targets
        ]
    )

    truth_norm = truth / np.maximum(np.linalg.norm(truth, axis=1, keepdims=True), 1e-9)
    oracle_sim = truth_norm @ truth_norm.T

    candidates = [("coexpression_knn", coexpr_sim)]
    if args.string_adj:
        import scipy.sparse as sp_

        from vcc2026.network import coverage

        adj = sp_.load_npz(args.string_adj)
        logger.info("STRING adjacency %s, gene coverage %.1f%%", adj.shape, 100 * coverage(adj))
        # Only the target rows are needed: similarity is cosine between the
        # neighbourhood profiles of the 150 measured targets.
        rows = np.asarray(adj[pos].todense(), dtype=np.float32)
        rows[np.arange(len(targets)), pos] = 1.0  # self-weight
        rows /= np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-9)
        string_sim = rows @ rows.T
        n_isolated = int((np.abs(adj[pos]).sum(axis=1).A.ravel() == 0).sum())
        logger.info("%d/%d targets have no STRING edge", n_isolated, len(targets))
        candidates.append(("string_knn", string_sim))
    candidates.append(("oracle_knn", oracle_sim))

    n = len(targets)
    results: dict[str, dict] = {}
    for name, sim in candidates:
        pred = np.zeros_like(truth)
        for i in range(n):
            s = sim[i].copy()
            s[i] = -np.inf  # leave-one-out: a target may never vote for itself
            w = neighbour_weights(s, k=args.k, power=args.power)
            if w.sum() <= 0:
                continue
            pred[i] = (w @ truth) / w.sum()
        r = row_correlation(pred, truth)
        results[name] = {
            "mean_r": float(r.mean()),
            "median_r": float(np.median(r)),
            "fraction_positive": float((r > 0).mean()),
            "discrimination": discrimination(pred, truth),
        }
        logger.info("%s: %s", name, results[name])

    mean_pred = np.zeros_like(truth)
    for i in range(n):
        others = np.ones(n, dtype=bool)
        others[i] = False
        mean_pred[i] = truth[others].mean(axis=0)
    r = row_correlation(mean_pred, truth)
    results["mean_signature"] = {
        "mean_r": float(r.mean()),
        "median_r": float(np.median(r)),
        "fraction_positive": float((r > 0).mean()),
        "discrimination": discrimination(mean_pred, truth),
    }
    logger.info("mean_signature: %s", results["mean_signature"])

    results["_meta"] = {"n_targets": n, "n_genes_scored": int(keep.sum()), "k": args.k}
    Path(args.out).write_text(json.dumps(results, indent=2))
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
