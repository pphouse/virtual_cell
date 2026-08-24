#!/usr/bin/env python
"""Rank model configurations without spending a submission.

Each leaderboard round costs about an hour of wall clock -- 25 minutes to build
the prediction, 10 to upload, 10 to 25 to score -- against a daily submission
allowance, with one submission in flight at a time.  Two of the three rounds so
far were spent discovering things that could have been measured locally.

The 2025 public release measures 300 knockdowns in H1 hESC across three
batches, which is enough to hold one batch out and score a prediction for it
built from the other two.  The metrics here are pseudobulk stand-ins for the
scorer's six, close enough in construction to order configurations:

* `pds` -- the real metric ranks a predicted effect against every real effect by
  cosine distance; reproduced exactly, at the pseudobulk level the real one also
  uses.
* `fid` -- direction agreement on the genes the real data moves most, and
  crucially *scaled by yield*: the leaderboard taught that predicting almost
  nothing scores near zero however right the little you predict is.
* `jac` -- overlap of the predicted and real top-moved gene sets.
* `nmae` -- error on the real DE genes, normalised by predicting no change.

The known 2026 anchors (docs/05) are printed alongside so a raw value can be
read as the score it would imply.  Two things it cannot capture: this is
within-cell-line, so it does not test the context transfer at all, and it works
on pseudobulk, so it says nothing about how the emitted cells behave under a
Wilcoxon test.  Use it to order candidates, not to predict a score.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np

from vcc2026.library import SignatureLibrary
from vcc2026.model import ContextTransferModel, ModelConfig

logger = logging.getLogger("offline")

# Solved from two submissions on the vcc2026-val-*-r4 anchor set (docs/05 §1b).
ANCHORS = {
    "pds": (0.5001, 0.9698),
    "fid": (0.5110, 0.8071),
    "jac": (0.0302, 0.3963),
    "nmae": (1.0014, 0.3805),
}


def scaled(metric: str, raw: float) -> float:
    b, r = ANCHORS[metric]
    return (raw - b) / (r - b)


def pds_cosine(pred: np.ndarray, truth: np.ndarray) -> float:
    p = pred / np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-12)
    t = truth / np.maximum(np.linalg.norm(truth, axis=1, keepdims=True), 1e-12)
    sim = p @ t.T
    n = pred.shape[0]
    ranks = np.zeros(n)
    for i in range(n):
        order = np.argsort(-sim[i])
        ranks[i] = 1.0 - float(np.flatnonzero(order == i)[0]) / n
    return float(ranks.mean())


def de_metrics(pred: np.ndarray, truth: np.ndarray, top_n: int = 200, moved: float = 0.05) -> dict:
    n = pred.shape[0]
    fid, jac, nmae = np.zeros(n), np.zeros(n), np.zeros(n)
    agree_only, yield_only = np.zeros(n), np.zeros(n)
    for i in range(n):
        real_top = np.argpartition(-np.abs(truth[i]), top_n - 1)[:top_n]
        called = np.flatnonzero(np.abs(pred[i]) > moved)

        # Direction agreement on the real DE genes, weighted by how many of them
        # the prediction actually claims: right-but-silent scores as low as wrong.
        hit = np.intersect1d(real_top, called, assume_unique=False)
        agree = float((np.sign(pred[i][hit]) == np.sign(truth[i][hit])).mean()) if hit.size else 0.0
        # Recovering half the real DE set is treated as full credit; below that
        # the score falls off linearly, which is the shape the leaderboard showed
        # (a prediction covering ~1 gene scored 0.002 against a 0.511 baseline).
        yield_ = hit.size / top_n
        fid[i] = agree * min(yield_ / 0.5, 1.0)
        # Reported separately because the product hides which half is failing:
        # a right-but-silent prediction and a loud-but-wrong one both score low.
        agree_only[i] = agree if hit.size else np.nan
        yield_only[i] = yield_

        pred_top = (
            np.argpartition(-np.abs(pred[i]), min(top_n, called.size) - 1)[:top_n]
            if called.size >= top_n
            else called
        )
        union = np.union1d(real_top, pred_top).size
        jac[i] = np.intersect1d(real_top, pred_top).size / union if union else 0.0

        err = np.abs(pred[i][real_top] - truth[i][real_top]).mean()
        null = np.abs(truth[i][real_top]).mean()
        nmae[i] = err / null if null > 0 else 1.0
    return {
        "fid": float(fid.mean()),
        "jac": float(jac.mean()),
        "nmae": float(nmae.mean()),
        "direction_agreement": float(np.nanmean(agree_only)),
        "de_yield": float(yield_only.mean()),
    }


def evaluate(lib: SignatureLibrary, held_out: str, config: ModelConfig, features) -> dict:
    from vcc2026.context import CellContext

    trimmed = SignatureLibrary(genes=lib.genes)
    for line in lib.lines:
        if line == held_out:
            continue
        trimmed.baseline[line] = lib.baseline[line]
        trimmed.deltas[line] = dict(lib.deltas[line])
        trimmed.n_cells[line] = dict(lib.n_cells.get(line, {}))
        trimmed.target_sum[line] = lib.target_sum.get(line, 1e4)

    targets = [t for t in lib.targets(held_out) if lib.gene_index(t) is not None]
    truth = np.vstack([lib.deltas[held_out][t] for t in targets])
    pos = np.array([lib.gene_index(t) for t in targets])

    # A pseudo-context standing in for the held-out batch: the model only reads
    # `mu`, and the batch's own control mean is exactly that.
    mu = lib.baseline[held_out]
    ctx = CellContext(
        name=held_out,
        genes=lib.genes,
        control=np.repeat(mu[None, :], 2, axis=0),
        target_sum=lib.target_sum.get(held_out, 1e4),
    )
    model = ContextTransferModel(config).fit(trimmed, features=features)
    pred = model.predict(targets, ctx).delta

    # Score the trans signature only: the on-target drop is modelled separately
    # and would otherwise dominate every number here.
    rows = np.arange(len(targets))
    pred, truth = pred.copy(), truth.copy()
    pred[rows, pos] = 0.0
    truth[rows, pos] = 0.0

    out = {"held_out": held_out, "n_targets": len(targets), "pds": pds_cosine(pred, truth)}
    out.update(de_metrics(pred, truth))
    return out


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--library", required=True)
    p.add_argument("--string-adj", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--sweep", default=None, help="NAME=v1,v2,... a ModelConfig field to vary")
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="fix a ModelConfig field; repeatable, applied before --sweep",
    )
    args = p.parse_args()

    lib = SignatureLibrary.load(args.library)
    features = None
    if args.string_adj:
        import scipy.sparse as sp

        from vcc2026.network import string_features

        adj = sp.load_npz(args.string_adj)
        features = string_features(adj, lib.genes, subset=sorted(lib.targets()))

    base = ModelConfig()
    for spec in args.set:
        name, value = spec.split("=", 1)
        base = replace(base, **{name: type(getattr(base, name))(value)})
    label0 = "+".join(args.set) if args.set else "default"
    variants = [(label0, base)]
    if args.sweep:
        name, values = args.sweep.split("=", 1)
        cast = type(getattr(base, name))
        variants = [(f"{name}={v}", replace(base, **{name: cast(v)})) for v in values.split(",")]

    print(
        f"{'config':28s} {'held out':10s} {'pds':>7s} {'fid':>7s} "
        f"{'jac':>7s} {'nmae':>7s}   scaled sum"
    )
    rows = []
    for label, cfg in variants:
        per = [evaluate(lib, line, cfg, features) for line in lib.lines]
        agg = {m: float(np.mean([r[m] for r in per])) for m in ("pds", "fid", "jac", "nmae")}
        s = sum(scaled(m, agg[m]) for m in agg)
        for r in per:
            print(
                f"{label:28s} {r['held_out']:10s} {r['pds']:7.4f} {r['fid']:7.4f} "
                f"{r['jac']:7.4f} {r['nmae']:7.4f}"
            )
        print(
            f"{label:28s} {'MEAN':10s} {agg['pds']:7.4f} {agg['fid']:7.4f} "
            f"{agg['jac']:7.4f} {agg['nmae']:7.4f}   {s:+.4f}"
        )
        rows.append({"config": label, **agg, "scaled_sum": s})

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
