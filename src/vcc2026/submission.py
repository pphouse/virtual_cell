"""Assembling the 2026 submission file.

The format is strict and every rule is checked before upload, so the job here
is to satisfy all of them by construction rather than to fix them up afterwards:

* exactly the official perturbation set, in every context, no extras;
* exactly `cells_per_pert` cells for each one -- one over or under is rejected;
* every cell tagged with the context label it was predicted for;
* raw non-negative integer counts;
* no control cells at all (scoring pairs each context with the held-out
  controls, never yours);
* all genes, in the bundle's order.

The context label deserves the paranoia the CLI guide gives it: it decides
which held-out dataset a cell is compared against, and swapping two contexts
degrades every metric toward chance while looking exactly like a weak model.
So labels are carried from the control file they came from and never
reconstructed.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .challenge import CONTEXT_COL, PERT_COL, Bundle
from .counts import context_mean_proportions, emit_counts
from .writer import SparseH5adWriter

logger = logging.getLogger(__name__)


class LfcPredictor(Protocol):
    """Anything that turns a target list into (P, G) natural-log fold changes."""

    def predict_lfc(self, targets: list[str]) -> np.ndarray: ...


@dataclass
class SubmissionConfig:
    pseudocount: float = 1.0  # calibrated against the real controls; see counts.py
    non_responder_fraction: float = 0.0
    response_shape: float = float("inf")  # inf = every cell gets the full effect
    max_counts_per_cell: int = 1_000_000
    seed: int = 0


def response_scales(n: int, cfg: SubmissionConfig, rng: np.random.Generator) -> np.ndarray:
    """Per-cell multipliers on the log fold change, with mean exactly 1.

    Left at 1 for every cell by default.  Real CRISPRi populations do contain
    escapees, but spreading the effect costs statistical power in the DE test
    that four of the six metrics run, and there is no ground truth for these
    contexts to tune the trade-off against -- so the heterogeneity is a knob to
    move once the leaderboard can answer it, not a default.
    """
    pi0 = float(np.clip(cfg.non_responder_fraction, 0.0, 0.95))
    if pi0 <= 0 and not np.isfinite(cfg.response_shape):
        return np.ones(n)
    responder = rng.random(n) >= pi0
    if not responder.any():
        responder[:] = True
    r = np.zeros(n)
    k = max(cfg.response_shape, 1e-3)
    r[responder] = rng.gamma(shape=k, scale=1.0 / k, size=int(responder.sum()))
    mean = r.mean()
    return r / mean if mean > 0 else np.ones(n)


def build_submission(
    bundle: Bundle,
    predictor_for: Callable[[str, sp.csr_matrix, np.ndarray], LfcPredictor],
    output: str | Path,
    config: SubmissionConfig | None = None,
) -> Path:
    """Predict every perturbation in every context and stream the .h5ad out.

    `predictor_for(context, control_counts, genes)` is called once per context
    and must return something with `predict_lfc`.  It is handed *only* that
    context's control cells, which is the whole information budget the
    challenge allows.
    """
    cfg = config or SubmissionConfig()
    output = Path(output)
    targets = [str(t) for t in bundle.perturbations]
    n_per = bundle.cells_per_pert

    # Built to match what `vcc prep` slims down to, so the file can be packaged
    # straight into a .vcc: plain string columns, positional string index, and
    # nothing else carried along.  `tests/test_vcc_parity.py` holds that claim
    # against the real prep.
    obs = pd.DataFrame(
        {
            PERT_COL: np.asarray(
                np.repeat(targets, n_per).tolist() * len(bundle.contexts), dtype=object
            ),
            CONTEXT_COL: np.repeat(np.asarray(bundle.contexts, dtype=object), len(targets) * n_per),
        },
        index=np.arange(bundle.n_cells).astype(str),
    )

    with SparseH5adWriter(output, obs, bundle.genes) as writer:
        for context in bundle.contexts:
            path = bundle.context_files[context]
            logger.info("context %s: reading %s", context, path.name)
            controls = ad.read_h5ad(path)
            _assert_context(controls, context, path)
            _assert_genes(controls, bundle.genes, path)

            counts = controls.X.tocsr()
            del controls
            mean_p = context_mean_proportions(counts)
            library = np.asarray(counts.sum(axis=1)).ravel()

            predictor = predictor_for(context, counts, bundle.genes)
            lfc = predictor.predict_lfc(targets)
            if lfc.shape != (len(targets), bundle.genes.size):
                raise ValueError(
                    f"predictor returned {lfc.shape}, expected {(len(targets), bundle.genes.size)}"
                )
            logger.info(
                "context %s: %d targets, %d with a non-zero prediction, median |lfc| %.3f",
                context,
                len(targets),
                int((np.abs(lfc).max(axis=1) > 0).sum()),
                float(np.median(np.abs(lfc).max(axis=1))),
            )

            rng = np.random.default_rng(abs(hash((cfg.seed, context))) % (2**32))
            for p, _target in enumerate(targets):
                pick = rng.choice(counts.shape[0], size=n_per, replace=n_per > counts.shape[0])
                base = counts[pick]
                scales = response_scales(n_per, cfg, rng)
                block = emit_counts(
                    base,
                    mean_proportions=mean_p,
                    log_fold_change=(
                        lfc[p][None, :] * scales[:, None]
                        if not np.allclose(scales, 1.0)
                        else lfc[p]
                    ),
                    pseudocount=cfg.pseudocount,
                    rng=rng,
                    library_sizes=library[pick],
                    max_counts_per_cell=cfg.max_counts_per_cell,
                )
                writer.append(block)
                if (p + 1) % 50 == 0:
                    logger.info(
                        "  context %s: %d/%d perturbations (%.2e stored entries so far)",
                        context,
                        p + 1,
                        len(targets),
                        writer.nnz,
                    )
            del counts
    return output


def _assert_context(adata: ad.AnnData, context: str, path: Path) -> None:
    if CONTEXT_COL not in adata.obs:
        raise ValueError(f"{path.name} has no {CONTEXT_COL!r} column")
    labels = set(adata.obs[CONTEXT_COL].astype(str))
    if labels != {context}:
        raise ValueError(
            f"{path.name} is labelled {sorted(labels)} but its filename says {context!r} -- "
            "refusing to guess which is right"
        )


def _assert_genes(adata: ad.AnnData, genes: np.ndarray, path: Path) -> None:
    got = np.asarray(adata.var_names, dtype=str)
    want = np.asarray(genes, dtype=str)
    if not np.array_equal(got, want):
        raise ValueError(f"{path.name} gene order does not match gene_names.csv")
