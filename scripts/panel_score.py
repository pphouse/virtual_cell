#!/usr/bin/env python
"""Score a prediction against several cell lines at once, without submitting.

The leaderboard allows two submissions a day and answers with one number per
context, so a change worth +0.005 cannot be told from noise by submitting it:
the panel it is measured on is 300 perturbations wide and the difference lives
inside the spread.  This builds the missing half of the loop out of public data
-- the 2025 H1 screen and the Replogle K562 and RPE1 screens -- and scores the
same prediction against all three, so a change can be asked the only question
that matters offline: does it point the same way in every line?

Each line contributes its own control cells (which is what the competition hands
you for the test context) and its own differential-expression table computed by
`cell_eval2`.  The source signal for a line is fitted from the *other* lines
only, so nothing a line is scored on was measured in that line.  The four
differential-expression members are the reimplementation in `strategy_search`,
checked against `cell_eval2` by `scripts/verify_local_metrics.py`; `pds` is the
cosine-rank proxy of `offline_score_2025`, computed on the panel-mean-subtracted
profiles, which is the form the submission actually emits.

The three panels are not interchangeable.  H1 is at the competition's exact
shape (400 cells a perturbation) but its knockdowns are about four times
stronger than the 2026 panel's; K562 and RPE1 are at 100 cells a perturbation,
which is the most the public screens support at 300 targets, and their weaker
tables happen to bracket the real one.  So absolute values are not comparable
across lines and are not meant to be.  What is comparable is the sign of a
difference between two arms measured on the same line, and whether that sign
survives the change of line.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_local_context import read_obs_column, read_var_names, stream_rows  # noqa: E402
from local_predict import generic_signal, specific_signal  # noqa: E402
from offline_score_2025 import pds_cosine  # noqa: E402
from strategy_search import evaluate  # noqa: E402

from vcc2026.budget import BudgetConfig, BudgetedPredictor  # noqa: E402
from vcc2026.library import SignatureLibrary  # noqa: E402
from vcc2026.proximity import ProximitySignal, load_gene_positions  # noqa: E402

logger = logging.getLogger("panel")

MEMBERS = ("pds", "fid", "reach", "jac", "nmae")

# Below this a difference is read as the change not touching the member at all.
TIE = 1e-4


@dataclass
class Panel:
    """One cell line's half of the loop: control cells and a measured DE table."""

    name: str
    genes: np.ndarray
    control: sp.csr_matrix
    targets: list[str]
    lfc: np.ndarray
    padj: np.ndarray

    @property
    def sig(self) -> np.ndarray:
        return self.padj < 0.05

    def generic_log2fc(self) -> np.ndarray:
        """The mean knockdown response of this line, with each target's own gene dropped."""
        lfc = self.lfc.copy()
        index = {g: i for i, g in enumerate(self.genes)}
        for row, target in enumerate(self.targets):
            own = index.get(target)
            if own is not None:
                lfc[row, own] = np.nan
        return np.nan_to_num(np.nanmean(lfc, axis=0))


def load_panel(name: str, reference: str, de: str, control_name: str = "non-targeting") -> Panel:
    genes = np.asarray([str(g) for g in read_var_names(reference, None)])
    labels = read_obs_column(reference, "target")
    rows = np.flatnonzero(labels == control_name)
    blocks = [b for _, b in stream_rows(reference, rows, block=8192)]
    control = sp.vstack(blocks, format="csr").astype(np.float32)
    del blocks

    table = pl.read_parquet(de, columns=["target", "feature", "log2_fold_change", "p_adj"])
    targets = sorted(set(table["target"].to_list()))
    ti = {t: i for i, t in enumerate(targets)}
    gi = {g: i for i, g in enumerate(genes)}
    lfc = np.zeros((len(targets), genes.size))
    padj = np.ones((len(targets), genes.size))
    t = table["target"].to_numpy()
    f = table["feature"].to_numpy()
    keep = np.fromiter((x in gi for x in f), bool, f.size)
    r = np.fromiter((ti[x] for x in t[keep]), np.int64, int(keep.sum()))
    c = np.fromiter((gi[x] for x in f[keep]), np.int64, int(keep.sum()))
    lfc[r, c] = table["log2_fold_change"].to_numpy()[keep]
    padj[r, c] = np.nan_to_num(table["p_adj"].to_numpy()[keep], nan=1.0)
    logger.info(
        "%s: %d control cells, %d perturbations, %d genes, mean n_real %.0f",
        name,
        control.shape[0],
        len(targets),
        genes.size,
        float((padj < 0.05).sum(1).mean()),
    )
    return Panel(name=name, genes=genes, control=control, targets=targets, lfc=lfc, padj=padj)


def transfer(sources: list[Panel], genes: np.ndarray) -> np.ndarray:
    """Average the source lines' generic responses onto another line's gene axis.

    Each line is averaged first and the lines are then averaged together, so a
    screen with more targets does not simply outvote a smaller one.  Genes the
    sources never measured stay at zero, which is the honest answer for them.
    """
    index = {g: i for i, g in enumerate(genes)}
    out = np.zeros((len(sources), genes.size))
    for k, src in enumerate(sources):
        response = src.generic_log2fc()
        cols = np.fromiter((index.get(g, -1) for g in src.genes), np.int64, src.genes.size)
        have = cols >= 0
        out[k, cols[have]] = response[have]
    shared = np.count_nonzero(out[0]) if len(sources) == 1 else int((out != 0).all(0).sum())
    logger.info("transfer: %d genes carried from %d source line(s)", shared, len(sources))
    return out.mean(axis=0)


def mask_own_gene(matrix: np.ndarray, panel: Panel) -> np.ndarray:
    """Zero each perturbation's own target gene, as every scored member does."""
    out = matrix.copy()
    index = {g: i for i, g in enumerate(panel.genes)}
    for row, target in enumerate(panel.targets):
        own = index.get(target)
        if own is not None:
            out[row, own] = 0.0
    return out


def sources_for(panel: Panel, lines: list[str]) -> list[str]:
    """Every library line that is not the one being scored.

    A screen split into folds carries the line's name as a prefix (`H1`, `H1val`,
    `H1test`), and all of them have to go: leaving one in would let the model see
    the very perturbations it is being asked to predict.
    """
    return [ln for ln in lines if not ln.startswith(panel.name)]


def score(
    panel: Panel, cfg: BudgetConfig, generic: np.ndarray, positions=None, specific=None
) -> dict:
    proximity = None
    if positions is not None and cfg.proximity_weight:
        proximity = ProximitySignal(panel.genes, positions)
    model = BudgetedPredictor(cfg, generic, specific=specific, proximity=proximity).fit(
        panel.control, panel.genes
    )
    pred = model.predict(panel.targets)
    log2 = pred.log2()

    out = evaluate(pred.call, pred.sign, log2, pred.key, panel.lfc, panel.sig)
    # Every member drops the perturbation's own target gene, and here it has to
    # be dropped by hand: the knockdown itself is the largest entry in the truth
    # and the prediction never spends a call on it, so leaving it in makes the
    # residual profiles point in opposite directions for reasons that have
    # nothing to do with the prediction. Then the panel mean comes off both --
    # `pds` reads only what distinguishes one target from another, so the shared
    # part is noise to it in the truth as much as in the prediction.
    p, t = mask_own_gene(log2, panel), mask_own_gene(panel.lfc, panel)
    out["pds"] = pds_cosine(p - p.mean(0), t - t.mean(0))
    out["n_pred"] = float(pred.call.sum(1).mean())
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--panel",
        action="append",
        required=True,
        metavar="NAME=REFERENCE.h5ad,DE.parquet",
        help="repeat once per cell line",
    )
    p.add_argument("--gene-positions", default=None)
    p.add_argument(
        "--library",
        default=None,
        help="signature library; enables the target-specific arm, fitted on the other lines only",
    )
    p.add_argument("--string-adj", default=None)
    p.add_argument("--specific-shrink", type=float, default=1.0)
    p.add_argument(
        "--generic-from-library",
        action="store_true",
        help="take the generic response from the library rather than from the other panels' "
        "DE tables, which is what the submission does; the two differ in shape as well as "
        "in units, so a mixing weight fitted here transfers only if this is set",
    )
    p.add_argument("--n-calls", type=int, default=8000)
    p.add_argument("--margin", type=float, default=1.35)
    p.add_argument("--top-margin", type=float, default=2.0)
    p.add_argument(
        "--arm",
        action="append",
        default=None,
        metavar="LABEL=FIELD:VALUE[,FIELD:VALUE...]",
        help="a BudgetConfig variant to score; the first is the reference arm",
    )
    p.add_argument("--out", default=None, help="write the table as JSON")
    args = p.parse_args()

    panels = []
    for spec in args.panel:
        name, paths = spec.split("=", 1)
        reference, de = paths.split(",", 1)
        panels.append(load_panel(name, reference, de))

    positions = load_gene_positions(args.gene_positions) if args.gene_positions else None
    lib = SignatureLibrary.load(args.library) if args.library else None
    base = BudgetConfig(n_calls=args.n_calls, margin=args.margin, top_margin=args.top_margin)

    arms = {"base": base}
    for spec in args.arm or []:
        label, fields = spec.split("=", 1)
        over = {}
        for field in fields.split(","):
            key, value = field.split(":", 1)
            over[key] = int(value) if key == "n_calls" else float(value)
        if over.get("specific_weight") and lib is None:
            p.error("--library is required for an arm with a specific weight")
        arms[label] = replace(base, **over)

    rows = []
    if args.generic_from_library and lib is None:
        p.error("--generic-from-library needs --library")

    for panel in panels:
        sources = [q for q in panels if q.name != panel.name]
        if not sources:
            logger.warning("%s has no source line; skipping", panel.name)
            continue
        if args.generic_from_library:
            lines = sources_for(panel, list(lib.baseline))
            generic = generic_signal(lib, lines, panel.genes)
            logger.info("generic response from library lines %s", ",".join(lines))
        else:
            generic = transfer(sources, panel.genes)
        specific = None
        if lib is not None:
            specific = specific_signal(
                lib,
                sources_for(panel, list(lib.baseline)),
                args.string_adj,
                args.specific_shrink,
                panel.targets,
                panel.control,
                panel.genes,
            )
        for label, cfg in arms.items():
            r = score(panel, cfg, generic, positions, specific)
            r.update(line=panel.name, arm=label, source="+".join(q.name for q in sources))
            rows.append(r)
            logger.info("%s / %s done", panel.name, label)

    head = f"{'line':8s} {'arm':14s} {'n_pred':>7s} " + " ".join(f"{m:>7s}" for m in MEMBERS)
    print(head)
    print("-" * len(head))
    for r in rows:
        print(
            f"{r['line']:8s} {r['arm']:14s} {r['n_pred']:7.0f} "
            + " ".join(f"{r[m]:7.4f}" for m in MEMBERS)
        )

    reference = next(iter(arms))
    for label in list(arms)[1:]:
        print(f"\n{label} - {reference}")
        print(f"{'line':8s} " + " ".join(f"{m:>8s}" for m in MEMBERS))
        for panel in panels:
            a = next((r for r in rows if r["line"] == panel.name and r["arm"] == reference), None)
            b = next((r for r in rows if r["line"] == panel.name and r["arm"] == label), None)
            if a and b:
                print(f"{panel.name:8s} " + " ".join(f"{b[m] - a[m]:+8.4f}" for m in MEMBERS))
        # A member the change does not touch agrees with itself trivially, so
        # ties are reported as ties rather than counted as agreement.
        agree, tied = [], []
        for m in MEMBERS:
            deltas = [
                next(r for r in rows if r["line"] == q.name and r["arm"] == label)[m]
                - next(r for r in rows if r["line"] == q.name and r["arm"] == reference)[m]
                for q in panels
            ]
            if all(abs(d) < TIE for d in deltas):
                tied.append(m)
            elif len({np.sign(d) for d in deltas if abs(d) >= TIE}) == 1 and all(
                abs(d) >= TIE for d in deltas
            ):
                agree.append(m)
        print(f"{'sign':8s} agrees on: {', '.join(agree) or 'nothing'}", end="")
        print(f" (unmoved: {', '.join(tied)})" if tied else "")

    if args.out:
        import json

        Path(args.out).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
