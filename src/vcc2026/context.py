"""The *cell context*: everything we are allowed to know about a target cell line.

For the 2026 challenge this is exactly two things per held-out line:

    1. a pool of unperturbed (non-targeting) cells, and
    2. the list of genes that will be knocked down, with a cell count each.

No measured perturbation response for that line is ever available -- that is
what makes the task zero-shot.  `CellContext` is deliberately the *only* object
the predictor is allowed to read at inference time, so that any accidental
leakage of held-out responses becomes a type error rather than a silent bug.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import anndata as ad
import numpy as np

from .normalize import as_normlog, median_library_size

CONTROL_NAME = "non-targeting"


def _looks_like_counts(x) -> bool:
    from .normalize import is_discrete

    return is_discrete(x)


@dataclass
class CellContext:
    """Unperturbed state of one cell line, in normlog space."""

    name: str
    genes: np.ndarray  # (G,) gene symbols, the submission gene order
    control: np.ndarray  # (n_ctrl, G) normlog control cells
    target_sum: float = 1e4  # library size the normlog scale was built on

    _mu: np.ndarray | None = field(default=None, repr=False)
    _var: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.genes = np.asarray(self.genes, dtype=object)
        if self.control.shape[1] != self.genes.size:
            raise ValueError(
                f"control has {self.control.shape[1]} genes, gene list has {self.genes.size}"
            )
        self._index = {g: i for i, g in enumerate(self.genes)}

    @property
    def mu(self) -> np.ndarray:
        """Mean normlog control profile -- the baseline every metric is scored against."""
        if self._mu is None:
            self._mu = self.control.mean(axis=0)
        return self._mu

    @property
    def var(self) -> np.ndarray:
        """Per-gene variance across control cells (used for heterogeneity modelling)."""
        if self._var is None:
            self._var = self.control.var(axis=0)
        return self._var

    @property
    def n_control(self) -> int:
        return self.control.shape[0]

    def gene_index(self, gene: str) -> int | None:
        return self._index.get(gene)

    def expressed_mask(self, min_mu: float = 0.05) -> np.ndarray:
        """Genes with non-trivial baseline expression in this line.

        A gene that is off in this context cannot be knocked *down*, and a
        prediction that moves it is pure metric loss.
        """
        return self.mu > min_mu

    @classmethod
    def from_anndata(
        cls,
        adata: ad.AnnData,
        name: str,
        pert_col: str = "target_gene",
        control_name: str = CONTROL_NAME,
        genes: np.ndarray | None = None,
        target_sum: float | None = None,
    ) -> CellContext:
        """Build a context from an h5ad holding (at least) the control cells.

        `target_sum=None` estimates the scale from these control cells' own
        median library size -- the statistic cell-eval will apply to the real
        held-out data, and the only estimate of it we are given.
        """
        if pert_col in adata.obs:
            mask = (adata.obs[pert_col].astype(str) == control_name).to_numpy()
            if not mask.any():
                raise ValueError(f"no cells with {pert_col} == {control_name!r} in {name}")
            adata = adata[mask]
        var_names = np.asarray(adata.var_names, dtype=object)
        scale = target_sum
        if scale is None:
            scale = median_library_size(adata.X) if _looks_like_counts(adata.X) else 1e4
        x = as_normlog(adata.X, target_sum=scale)
        if genes is not None:
            genes = np.asarray(genes, dtype=object)
            pos = {g: i for i, g in enumerate(var_names)}
            missing = [g for g in genes if g not in pos]
            if missing:
                raise ValueError(
                    f"{len(missing)} submission genes missing from {name}, e.g. {missing[:5]}"
                )
            x = x[:, np.array([pos[g] for g in genes])]
            var_names = genes
        return cls(name=name, genes=var_names, control=x, target_sum=float(scale))

    def subsample(self, n: int, rng: np.random.Generator) -> CellContext:
        """Cheap thinning of the control pool (development / memory control)."""
        if n >= self.n_control:
            return self
        idx = rng.choice(self.n_control, size=n, replace=False)
        return CellContext(
            name=self.name,
            genes=self.genes,
            control=self.control[idx],
            target_sum=self.target_sum,
        )
