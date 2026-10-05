#!/usr/bin/env python
"""Leave-one-cell-line-out benchmark on the simulator, plus the alpha sweep.

Runs without any challenge data.  Its purpose is to show the pipeline end to
end and, more usefully, to make the metric trade-off visible: the same
predictions scored at a range of global magnitude scales.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from synthetic import SimConfig, simulate  # noqa: E402

from vcc2026 import CellContext, ModelConfig, build_signature_library  # noqa: E402
from vcc2026.calibrate import evaluate_holdout  # noqa: E402
from vcc2026.localeval import baseline_pair, evaluate  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

HEADERS = [
    "mae",
    "mae_delta",
    "pearson_delta",
    "discrimination_score_l1",
    "overlap_at_N_proxy",
    "de_direction_match_proxy",
]


def row(label: str, values: dict[str, float]) -> str:
    cells = "  ".join(f"{values.get(h, float('nan')):>10.4f}" for h in HEADERS)
    return f"{label:<26}{cells}"


def main() -> None:
    genes, _, adatas = simulate(SimConfig(seed=1))
    lib = build_signature_library(dict(adatas), genes=genes)

    header = f"{'':<26}" + "  ".join(f"{h[:10]:>10}" for h in HEADERS)

    print("\n=== leave-one-cell-line-out (raw metrics) ===")
    print(header)
    for held_out in lib.lines:
        ctx = CellContext.from_anndata(adatas[held_out], name=held_out, genes=genes)
        metrics, scores, pair = evaluate_holdout(lib, held_out, ctx, config=ModelConfig())
        base = evaluate(baseline_pair(pair))
        print(row(f"{held_out} baseline", base))
        print(row(f"{held_out} CCDT", metrics))
        print(f"{'':<26}avg_score vs baseline = {scores['avg_score']:.4f}\n")

    held_out = lib.lines[-1]
    ctx = CellContext.from_anndata(adatas[held_out], name=held_out, genes=genes)
    print(f"=== alpha sweep on {held_out} (normalised scores, clipped at 0) ===")
    print(f"{'alpha':<26}" + "  ".join(f"{h[:10]:>10}" for h in HEADERS) + "  avg_score")
    for alpha in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]:
        _, scores, _ = evaluate_holdout(lib, held_out, ctx, config=ModelConfig(alpha=alpha))
        cells = "  ".join(f"{scores.get(h, np.nan):>10.4f}" for h in HEADERS)
        print(f"{alpha:<26.2f}{cells}  {scores['avg_score']:.4f}")


if __name__ == "__main__":
    main()
