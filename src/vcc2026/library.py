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


def add_source_streaming(
    lib: SignatureLibrary,
    path: str | Path,
    line: str,
    pert_col: str = "target_gene",
    control_name: str = CONTROL_NAME,
    min_cells: int = 5,
    block: int = 4096,
    target_sum: float | None = None,
) -> None:
    """Pseudobulk a screen too large to load, straight off disk.

    The 2025 release is 221k cells by 18k genes with 1.9e9 stored entries --
    around 15 GB on disk and more than that in memory.  Nothing here needs the
    cells themselves, only one mean per perturbation, so the file is streamed in
    row blocks and collapsed as it goes.  Peak memory is one block plus the
    (perturbations x genes) accumulator: a few hundred MB rather than tens of GB.
    """
    import h5py

    path = Path(path)
    with h5py.File(path, "r") as f:
        encoding = f["X"].attrs.get("encoding-type")
        dense = isinstance(f["X"], h5py.Dataset)
        if not dense and encoding != "csr_matrix":
            raise ValueError(f"{path.name}: unsupported X encoding {encoding!r}")
        n_cells, n_src_genes = (
            (int(v) for v in f["X"].shape) if dense else (int(v) for v in f["X"].attrs["shape"])
        )
        src_genes = _read_string_index(f["var"])
        labels = _read_categorical(f["obs"], pert_col)
        if labels is None:
            raise ValueError(f"{path.name}: no {pert_col!r} in obs")

        keep = np.array([g in lib._gene_index for g in src_genes])
        if not keep.any():
            raise ValueError(f"{path.name}: no shared genes with the library")
        col_of = np.full(n_src_genes, -1, dtype=np.int64)
        col_of[keep] = [lib._gene_index[g] for g in src_genes[keep]]

        if dense:
            data = indices = None
            ptr = None
            probe = f["X"][: min(n_cells, 2000)]
            totals = probe.sum(axis=1)
        else:
            data, indices, indptr = f["X/data"], f["X/indices"], f["X/indptr"]
            ptr = indptr[:]
            step = max(min(n_cells, 20000) // 2000, 1)
            totals = np.array(
                [data[ptr[i] : ptr[i + 1]].sum() for i in range(0, min(n_cells, 20000), step)]
            )

        if target_sum is None:
            target_sum = float(np.median(totals[totals > 0])) if totals.size else 1e4
        lib.target_sum[line] = float(target_sum)
        logger.info(
            "%s: %d cells x %d genes (%d shared), normlog target sum %.0f",
            line,
            n_cells,
            n_src_genes,
            int(keep.sum()),
            target_sum,
        )

        uniq = np.unique(labels)
        row_of = {t: i for i, t in enumerate(uniq)}
        acc = np.zeros((uniq.size, lib.genes.size), dtype=np.float64)
        counts = np.zeros(uniq.size, dtype=np.int64)

        keep_cols = np.flatnonzero(col_of >= 0)
        dest_cols = col_of[keep_cols]

        for start in range(0, n_cells, block):
            stop = min(start + block, n_cells)
            groups = np.array([row_of[t] for t in labels[start:stop]])

            if dense:
                # Dense and gzip-compressed (the scPerturb releases): one row
                # block at a time, restricted to the shared genes.
                chunk = np.asarray(f["X"][start:stop], dtype=np.float64)
                totals = chunk.sum(axis=1)
                totals[totals == 0] = 1.0
                normed = np.log1p(chunk[:, keep_cols] * (target_sum / totals)[:, None])
                for g in np.unique(groups):
                    sel = groups == g
                    acc[g, dest_cols] += normed[sel].sum(axis=0)
            else:
                lo, hi = int(ptr[start]), int(ptr[stop])
                d = data[lo:hi].astype(np.float64)
                idx = indices[lo:hi]
                rows = np.repeat(np.arange(stop - start), np.diff(ptr[start : stop + 1]))
                totals = np.zeros(stop - start)
                np.add.at(totals, rows, d)
                totals[totals == 0] = 1.0
                d = np.log1p(d * (target_sum / totals[rows]))
                good = col_of[idx] >= 0
                np.add.at(acc, (groups[rows[good]], col_of[idx[good]]), d[good])

            np.add.at(counts, groups, 1)
            if (start // block) % 10 == 0:
                logger.info("  %s: %d/%d cells", line, stop, n_cells)

    means = acc / np.maximum(counts, 1)[:, None]
    if control_name not in row_of:
        raise ValueError(f"{path.name}: no {control_name!r} cells")
    base = means[row_of[control_name]]
    lib.baseline[line] = base
    lib.deltas.setdefault(line, {})
    lib.n_cells.setdefault(line, {})
    lib.n_cells[line][control_name] = int(counts[row_of[control_name]])
    for target, i in row_of.items():
        if target == control_name or counts[i] < min_cells:
            continue
        lib.deltas[line][str(target)] = means[i] - base
        lib.n_cells[line][str(target)] = int(counts[i])
    logger.info("%s: %d signatures", line, len(lib.deltas[line]))


def _read_string_index(group) -> np.ndarray:
    key = group.attrs.get("_index", "_index")
    node = group[key]
    if hasattr(node, "keys") and "values" in node:
        return np.asarray(node["values"][:], dtype=object).astype(str)
    return np.asarray(node[:]).astype(str)


def _read_categorical(group, name: str) -> np.ndarray | None:
    if name not in group:
        return None
    node = group[name]
    if hasattr(node, "keys") and "categories" in node:
        cats = np.asarray(node["categories"][:]).astype(str)
        return cats[node["codes"][:]]
    return np.asarray(node[:]).astype(str)
