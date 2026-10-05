#!/usr/bin/env python
"""Search prediction strategies against a real DE table, without emitting cells.

Three of the four DE metrics are decided by a submission's *call set* -- which
genes it declares as responding -- and the emission only has to realise the set
it is given.  So the choice of set can be searched directly against a reference
DE table computed by the scorer's own code, at a few seconds per candidate
instead of the twenty minutes an emit-and-score round costs.

The four members are reimplemented here exactly as
`docs/vcc2026_metrics/vcc2026-metrics-brief.md` sections 4 to 7 define them, and
`scripts/verify_local_metrics.py` checks this file against `cell_eval2` on an
emitted prediction before any number here is trusted.
"""

from __future__ import annotations

import argparse

import numpy as np


def fidelity(call: np.ndarray, sign: np.ndarray, lfc: np.ndarray, sig: np.ndarray) -> float:
    """Section 4.  `call` and `sig` are boolean (perts x genes); `sign` is the
    predicted direction on the called genes."""
    adjud = lfc != 0.0
    out = []
    for i in range(call.shape[0]):
        n_real = int(sig[i].sum())
        pred = call[i] & adjud[i]
        n_pred = int(pred.sum())
        if n_pred == 0 and n_real == 0:
            continue
        k = int((np.sign(lfc[i][pred]) == sign[i][pred]).sum())
        out.append(k / max(n_pred, n_real) if max(n_pred, n_real) else 0.0)
    return float(np.mean(out)) if out else float("nan")


def jaccard(call: np.ndarray, sig: np.ndarray) -> float:
    """Section 6."""
    inter = (call & sig).sum(1)
    union = (call | sig).sum(1)
    per = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
    return float(per.mean())


def reach(
    call: np.ndarray, score: np.ndarray, sign: np.ndarray, lfc: np.ndarray, sig: np.ndarray,
    purity: float = 0.9,
) -> float:
    """Section 5.  `score` orders the submission's confidence, high first; its
    significant calls are placed ahead of everything else."""
    out = []
    for i in range(call.shape[0]):
        pool = np.flatnonzero(sig[i])
        if pool.size == 0:
            continue
        adjud = lfc[i][pool] != 0.0
        pool = pool[adjud]
        if pool.size == 0:
            out.append(0.0)
            continue
        order = np.lexsort((-score[i][pool], ~call[i][pool]))
        seq = pool[order]
        hit = (np.sign(lfc[i][seq]) == sign[i][seq]).astype(np.int64)
        run = np.cumsum(hit) / np.arange(1, seq.size + 1)
        ok = np.flatnonzero(run >= purity)
        out.append((int(ok[-1]) + 1) / int(sig[i].sum()) if ok.size else 0.0)
    return float(np.mean(out)) if out else float("nan")


def nmae(pred_lfc: np.ndarray, lfc: np.ndarray, sig: np.ndarray, min_gate: int = 10) -> float:
    """Section 7."""
    out = []
    for i in range(lfc.shape[0]):
        gate = sig[i] & np.isfinite(lfc[i])
        if int(gate.sum()) < min_gate:
            continue
        denom = np.abs(lfc[i][gate]).sum()
        if not np.isfinite(denom) or denom <= 0:
            continue
        out.append(float(np.abs(pred_lfc[i][gate] - lfc[i][gate]).sum() / denom))
    return float(np.mean(out)) if out else float("nan")


# Officially measured on the three 2026 val contexts (metrics brief, section 8),
# midpoints of the per-context ranges.  Used only to read a raw value as the score
# it would imply; the ordering of candidates never depends on them.
ANCHORS = {
    "fid": (0.513, 0.813),
    "reach": (0.072, 0.968),
    "jac": (0.029, 0.399),
    "nmae": (1.0013, 0.400),
}


def scaled(metric: str, raw: float) -> float:
    b, r = ANCHORS[metric]
    return (raw - b) / (r - b)


def evaluate(call, sign, pred_lfc, score, lfc, sig) -> dict:
    out = {
        "fid": fidelity(call, sign, lfc, sig),
        "reach": reach(call, score, sign, lfc, sig),
        "jac": jaccard(call, sig),
        "nmae": nmae(pred_lfc, lfc, sig),
    }
    out["sum_scaled"] = sum(scaled(m, out[m]) for m in ("fid", "reach", "jac", "nmae"))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--de", required=True, help="ref60_de.npz from the analysis step")
    args = p.parse_args()
    d = np.load(args.de, allow_pickle=True)
    lfc, padj = d["LFC"], d["PADJ"]
    sig = padj < 0.05
    n_real = sig.sum(1)
    print(
        f"{lfc.shape[0]} perturbations, {lfc.shape[1]} genes, "
        f"n_real median {np.median(n_real):.0f}"
    )


if __name__ == "__main__":
    main()
