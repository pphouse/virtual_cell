"""The signature library: every perturbation response we are allowed to train on.

The 2026 rules permit training on any data, so the library is the union of
whatever CRISPRi Perturb-seq we can get -- the 2025 VCC H1 hESC release,
Replogle K562/RPE1/HepG2, and any in-house screens.  Each source contributes
one *(cell line, target gene) -> pseudobulk delta* signature plus that line's
control baseline.

Pseudobulk deltas, not single cells, are the right unit here: cell-eval's
expression metrics (MAE, pearson_delta, the L1 discrimination score) are all
computed on per-perturbation means, and the per-cell spread is re-injected
later by the sampler.  Collapsing early is a ~1000x memory win with no loss on
the metrics that matter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import anndata as ad
import numpy as np

from .context import CONTROL_NAME
from .normalize import as_normlog, is_discrete, median_library_size

logger = logging.getLogger(__name__)


@dataclass
class SignatureLibrary:
    """Pseudobulk perturbation signatures across source cell lines."""

    genes: np.ndarray  # (G,) common gene space
    baseline: dict[str, np.ndarray] = field(
        default_factory=dict
    )  # line -> (G,) normlog control mean
    deltas: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)  # line -> gene -> (G,)
    n_cells: dict[str, dict[str, int]] = field(default_factory=dict)
    target_sum: dict[str, float] = field(default_factory=dict)  # line -> normlog scale

    def __post_init__(self) -> None:
        self.genes = np.asarray(self.genes, dtype=object)
        self._gene_index = {g: i for i, g in enumerate(self.genes)}

    # -- accessors -------------------------------------------------------
    @property
    def lines(self) -> list[str]:
        return sorted(self.deltas)

    def targets(self, line: str | None = None) -> list[str]:
        if line is not None:
            return sorted(self.deltas.get(line, {}))
        seen: set[str] = set()
        for d in self.deltas.values():
            seen.update(d)
        return sorted(seen)

    def gene_index(self, gene: str) -> int | None:
        return self._gene_index.get(gene)

    def lines_with(self, target: str) -> list[str]:
        return [ln for ln in self.lines if target in self.deltas[ln]]

    def delta_matrix(self, line: str) -> tuple[list[str], np.ndarray]:
        """(target genes, (P x G) delta matrix) for one source line."""
        targets = self.targets(line)
        if not targets:
            return [], np.zeros((0, self.genes.size))
        return targets, np.vstack([self.deltas[line][t] for t in targets])

    def weight(self, line: str, target: str, prior_cells: float = 30.0) -> float:
        """Shrinkage weight for a signature: noisy low-n signatures count less."""
        n = float(self.n_cells.get(line, {}).get(target, 0))
        return n / (n + prior_cells)

    # -- construction ----------------------------------------------------
    def add_source(
        self,
        adata: ad.AnnData,
        line: str,
        pert_col: str = "target_gene",
        control_name: str = CONTROL_NAME,
        min_cells: int = 5,
        target_sum: float | None = None,
    ) -> None:
        """Collapse one perturbation screen into pseudobulk deltas.

        Each source keeps its own normlog scale: signatures leave this object
        as fold changes and are re-expressed on the *target* line's scale by
        `transfer.rebase`, so cross-source depth differences never reach the
        prediction.
        """
        if pert_col not in adata.obs:
            raise ValueError(f"{pert_col!r} not in obs: {list(adata.obs.columns)}")
        var_names = np.asarray(adata.var_names, dtype=object)
        keep = np.array([g in self._gene_index for g in var_names])
        if not keep.any():
            raise ValueError(f"source {line} shares no genes with the library gene space")
        col_of = np.array([self._gene_index[g] for g in var_names[keep]])

        scale = target_sum
        if scale is None:
            scale = median_library_size(adata.X) if is_discrete(adata.X) else 1e4
        self.target_sum[line] = float(scale)

        labels = adata.obs[pert_col].astype(str).to_numpy()
        ctrl_mask = labels == control_name
        if not ctrl_mask.any():
            raise ValueError(f"source {line} has no {control_name!r} cells")

        base = np.zeros(self.genes.size)
        base[col_of] = as_normlog(adata[ctrl_mask].X, target_sum=scale)[:, keep].mean(axis=0)
        self.baseline[line] = base
        self.deltas.setdefault(line, {})
        self.n_cells.setdefault(line, {})
        self.n_cells[line][control_name] = int(ctrl_mask.sum())

        for target in np.unique(labels[~ctrl_mask]):
            mask = labels == target
            n = int(mask.sum())
            if n < min_cells:
                continue
            mean = np.zeros(self.genes.size)
            mean[col_of] = as_normlog(adata[mask].X, target_sum=scale)[:, keep].mean(axis=0)
            self.deltas[line][str(target)] = mean - base
            self.n_cells[line][str(target)] = n
        logger.info(
            "library: %s -> %d signatures over %d genes",
            line,
            len(self.deltas[line]),
            int(keep.sum()),
        )

    # -- persistence -----------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        arrays: dict[str, np.ndarray] = {"__genes__": self.genes.astype(str)}
        meta: list[str] = []
        for line in self.lines:
            arrays[f"base::{line}"] = self.baseline[line]
            targets = self.targets(line)
            arrays[f"delta::{line}"] = np.vstack([self.deltas[line][t] for t in targets])
            arrays[f"targets::{line}"] = np.array(targets, dtype=str)
            arrays[f"ncells::{line}"] = np.array(
                [self.n_cells[line].get(t, 0) for t in targets], dtype=np.int64
            )
            arrays[f"tsum::{line}"] = np.array([self.target_sum.get(line, 1e4)])
            meta.append(line)
        arrays["__lines__"] = np.array(meta, dtype=str)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> SignatureLibrary:
        z = np.load(path, allow_pickle=False)
        lib = cls(genes=z["__genes__"].astype(object))
        for line in z["__lines__"].tolist():
            lib.baseline[line] = z[f"base::{line}"]
            targets = z[f"targets::{line}"].tolist()
            mat = z[f"delta::{line}"]
            counts = z[f"ncells::{line}"]
            lib.deltas[line] = {t: mat[i] for i, t in enumerate(targets)}
            lib.n_cells[line] = {t: int(counts[i]) for i, t in enumerate(targets)}
            key = f"tsum::{line}"
            lib.target_sum[line] = float(z[key][0]) if key in z else 1e4
        return lib


def build_signature_library(
    sources: dict[str, ad.AnnData | str],
    genes: np.ndarray,
    pert_col: str = "target_gene",
    control_name: str = CONTROL_NAME,
    min_cells: int = 5,
) -> SignatureLibrary:
    """Build a library from ``{cell_line: h5ad-or-path}`` over a fixed gene space."""
    lib = SignatureLibrary(genes=np.asarray(genes, dtype=object))
    for line, src in sources.items():
        adata = ad.read_h5ad(src) if isinstance(src, str) else src
        lib.add_source(
            adata, line=line, pert_col=pert_col, control_name=control_name, min_cells=min_cells
        )
    return lib
