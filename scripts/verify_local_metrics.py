#!/usr/bin/env python
"""Check this project's DE metric implementations against `cell_eval2`'s own.

`scripts/strategy_search.py` reimplements four of the six scored members so a
candidate call set can be searched in seconds rather than in DE passes.  A
reimplementation is only worth having if it agrees with the thing it stands in
for, and the four public metric functions read DE tables directly -- no matrices,
no cell emission -- so the comparison costs nothing but the tables already on
disk.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from cell_eval2.metrics.de import de_lfc_nmae, de_sig_jaccard
from cell_eval2.metrics.direction import de_direction_fidelity_yield_raw, de_direction_reach

sys.path.insert(0, str(Path(__file__).resolve().parent))

from local_score import tables_to_arrays  # noqa: E402
from strategy_search import evaluate  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--de-real", required=True)
    p.add_argument("--de-pred", required=True)
    args = p.parse_args()

    real = pl.read_parquet(args.de_real)
    pred = pl.read_parquet(args.de_pred)
    kw = dict(de_pred=pred, de_real=real, control="non-targeting")

    official = {
        "fid": de_direction_fidelity_yield_raw(**kw),
        "reach": de_direction_reach(**kw),
        "jac": de_sig_jaccard(**kw),
        "nmae": de_lfc_nmae(**kw),
    }
    official = {
        k: (float(np.mean(list(v.values()))) if isinstance(v, dict) else float(v))
        for k, v in official.items()
    }

    _, _, (rl, rp), (pl_, pp) = tables_to_arrays(real, pred)
    mine = evaluate(pp < 0.05, np.sign(pl_), pl_, -np.log10(np.maximum(pp, 1e-300)), rl, rp < 0.05)

    print(f"{'member':8s} {'cell_eval2':>12s} {'local':>12s} {'diff':>10s}")
    for k in ("fid", "reach", "jac", "nmae"):
        print(f"{k:8s} {official[k]:12.5f} {mine[k]:12.5f} {official[k] - mine[k]:10.5f}")


if __name__ == "__main__":
    main()
