"""Checking a submission against every rule the scorer enforces, without opening it.

`vcc prep` performs these checks, but it does so by materialising the whole
prediction -- 15.84 GiB resident at this panel's density, per the CLI's own
sizing table -- which is more memory than this machine has.  Every rule is
nevertheless checkable by streaming: the structural ones read `obs` alone, and
the value rules read `X` in chunks.  Peak memory here is one chunk.

The rules, and why each one matters, are in `docs/02-評価指標の解剖.md`.  The one
that cannot be checked by anyone but the author is whether the context labels
are attached to the right cells: `A` and `B` swapped passes every check below
and scores like a broken model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

from .challenge import CONTEXT_COL, CONTROL_LABEL, PERT_COL, Bundle

logger = logging.getLogger(__name__)

MAX_CELLS = 400_000
MAX_NNZ = 4_750_000_000
MAX_COUNTS_PER_CELL = 1_000_000


@dataclass
class ValidationReport:
    n_cells: int = 0
    n_genes: int = 0
    nnz: int = 0
    nnz_per_cell: float = 0.0
    max_counts_per_cell: float = 0.0
    cells_per_context: dict[str, int] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def render(self) -> str:
        lines = [
            f"cells                {self.n_cells:,} (cap {MAX_CELLS:,})",
            f"genes                {self.n_genes:,}",
            f"stored entries       {self.nnz:,} "
            f"({100 * self.nnz / MAX_NNZ:.1f}% of the density cap)",
            f"entries per cell     {self.nnz_per_cell:,.0f}",
            f"max counts per cell  {self.max_counts_per_cell:,.0f} (cap {MAX_COUNTS_PER_CELL:,})",
            f"cells per context    {self.cells_per_context}",
        ]
        lines += (
            ["OK: every documented rule passes"]
            if self.ok
            else [
                "REJECTED:",
                *(f"  - {p}" for p in self.problems),
            ]
        )
        return "\n".join(lines)


def validate_submission(
    path: str | Path, bundle: Bundle, chunk: int = 50_000_000
) -> ValidationReport:
    path = Path(path)
    r = ValidationReport()
    with h5py.File(path, "r") as f:
        r.n_cells, r.n_genes = (int(v) for v in f["X"].attrs["shape"])
        genes = _read_index(f["var"])
        pert = _read_column(f["obs"], PERT_COL)
        context = _read_column(f["obs"], CONTEXT_COL)

        want_genes = np.asarray(bundle.genes, dtype=str)
        if r.n_genes != want_genes.size:
            r.problems.append(f"gene dimension {r.n_genes} != {want_genes.size}")
        elif not np.array_equal(genes, want_genes):
            missing = set(want_genes) - set(genes)
            r.problems.append(
                f"gene set differs (missing {len(missing)})" if missing else "genes out of order"
            )
        if r.n_cells > MAX_CELLS:
            r.problems.append(f"{r.n_cells} cells exceeds the cap of {MAX_CELLS}")

        if pert is None or context is None:
            r.problems.append(f"obs must carry {PERT_COL!r} and {CONTEXT_COL!r}")
            return r

        n_control = int((pert == CONTROL_LABEL).sum())
        if n_control:
            r.problems.append(
                f"{n_control} control cells present; a submission with any is rejected"
            )

        seen = sorted(set(context.tolist()))
        want_contexts = bundle.contexts
        if seen != want_contexts:
            unknown = sorted(set(seen) - set(want_contexts))
            missing = sorted(set(want_contexts) - set(seen))
            if unknown:
                r.problems.append(f"unknown context label(s): {unknown}")
            if missing:
                r.problems.append(f"missing predictions for context(s): {missing}")

        official = set(np.asarray(bundle.perturbations, dtype=str).tolist())
        for ctx in want_contexts:
            mask = context == ctx
            r.cells_per_context[ctx] = int(mask.sum())
            got = pert[mask]
            labels, counts = np.unique(got, return_counts=True)
            extra = sorted(set(labels) - official)
            absent = sorted(official - set(labels))
            if extra:
                r.problems.append(
                    f"context {ctx}: {len(extra)} unexpected perturbation(s), e.g. {extra[:3]}"
                )
            if absent:
                r.problems.append(
                    f"context {ctx}: {len(absent)} perturbation(s) missing, e.g. {absent[:3]}"
                )
            bad = labels[counts != bundle.cells_per_pert]
            if bad.size:
                r.problems.append(
                    f"context {ctx}: {bad.size} perturbation(s) have the wrong number of cells "
                    f"(need exactly {bundle.cells_per_pert}), e.g. {bad[:3].tolist()}"
                )

        data = f["X/data"]
        indptr = f["X/indptr"][:]
        r.nnz = int(data.shape[0])
        r.nnz_per_cell = r.nnz / max(r.n_cells, 1)
        if r.nnz > MAX_NNZ:
            r.problems.append(f"{r.nnz:,} stored entries exceeds the density cap of {MAX_NNZ:,}")

        totals = np.zeros(r.n_cells, dtype=np.float64)
        seen_fractional = seen_negative = seen_nonfinite = 0
        row = 0
        for start in range(0, r.nnz, chunk):
            stop = min(start + chunk, r.nnz)
            block = data[start:stop].astype(np.float64)
            if not np.isfinite(block).all():
                seen_nonfinite += int((~np.isfinite(block)).sum())
                block = np.nan_to_num(block)
            seen_negative += int((block < 0).sum())
            seen_fractional += int((block != np.round(block)).sum())
            # accumulate per-cell totals using the row boundaries in this slice
            while row < r.n_cells and indptr[row + 1] <= stop:
                lo = max(int(indptr[row]), start)
                totals[row] += block[lo - start : int(indptr[row + 1]) - start].sum()
                row += 1
            if row < r.n_cells and indptr[row] < stop:
                lo = max(int(indptr[row]), start)
                totals[row] += block[lo - start :].sum()
            logger.info("  scanned %.0f%% of stored values", 100 * stop / r.nnz)

        if seen_nonfinite:
            r.problems.append(f"{seen_nonfinite} non-finite value(s)")
        if seen_negative:
            r.problems.append(f"{seen_negative} negative value(s); counts cannot be below zero")
        if seen_fractional:
            r.problems.append(
                f"{seen_fractional} fractional value(s); submissions must be raw integer counts"
            )
        r.max_counts_per_cell = float(totals.max()) if r.n_cells else 0.0
        if r.max_counts_per_cell > MAX_COUNTS_PER_CELL:
            r.problems.append(
                f"a cell totals {r.max_counts_per_cell:,.0f} counts, "
                f"over the {MAX_COUNTS_PER_CELL:,} cap"
            )
    return r


def _read_index(group) -> np.ndarray:
    key = group.attrs.get("_index", "_index")
    node = group[key]
    if hasattr(node, "keys") and "values" in node:
        return np.asarray(node["values"][:]).astype(str)
    return np.asarray(node[:]).astype(str)


def _read_column(group, name: str) -> np.ndarray | None:
    if name not in group:
        return None
    node = group[name]
    if hasattr(node, "keys") and "categories" in node:
        cats = np.asarray(node["categories"][:]).astype(str)
        return cats[node["codes"][:]]
    return np.asarray(node[:]).astype(str)
