"""The on-target term: CRISPRi knocks down the gene it targets.

This is the single most reliable thing we know about the experiment, and it is
worth modelling separately from the trans effects because it behaves
differently: the size of the drop is a property of the guide and the promoter,
not of the downstream biology.

It is also worth knowing what it does *not* buy.  cell-eval's L1 discrimination
score (PDS) excludes the target gene column by construction, so a perfect
on-target prediction contributes nothing there.  It pays off in MAE, in
pearson_delta, and in the DE metrics, where the target gene is usually the
single most significant hit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .context import CellContext
from .library import SignatureLibrary
from .transfer import EPS_PSEUDOCOUNT, apply_fold_change, fold_change

DEFAULT_KNOCKDOWN = 0.65  # typical CRISPRi residual-expression literature value


@dataclass
class KnockdownModel:
    """Per-gene knockdown efficiency, learned from the library where possible."""

    per_gene: dict[str, float]
    default: float = DEFAULT_KNOCKDOWN

    @classmethod
    def fit(
        cls,
        lib: SignatureLibrary,
        default: float = DEFAULT_KNOCKDOWN,
        min_baseline: float = 0.1,
    ) -> KnockdownModel:
        """Median observed on-target fold change per gene across source lines."""
        obs: dict[str, list[float]] = {}
        for line in lib.lines:
            mu = lib.baseline[line]
            for target, delta in lib.deltas[line].items():
                j = lib.gene_index(target)
                if j is None or mu[j] < min_baseline:
                    continue  # gene not measurable in this line: uninformative
                fc = float(fold_change(mu[j : j + 1], delta[j : j + 1])[0])
                obs.setdefault(target, []).append(np.clip(fc, 0.02, 1.5))
        per_gene = {g: float(np.median(v)) for g, v in obs.items()}
        pooled = float(np.median(list(per_gene.values()))) if per_gene else 1.0 - default
        return cls(per_gene=per_gene, default=float(np.clip(1.0 - pooled, 0.0, 1.0)))

    def residual_fraction(self, gene: str) -> float:
        """Fraction of baseline expression remaining after knockdown."""
        return self.per_gene.get(gene, 1.0 - self.default)

    def apply(self, ctx: CellContext, target: str, delta: np.ndarray) -> np.ndarray:
        """Overwrite the target-gene entry of `delta` with the on-target drop."""
        j = ctx.gene_index(target)
        if j is None:
            return delta
        out = delta.copy()
        fc = np.array([self.residual_fraction(target)])
        out[j] = float(apply_fold_change(ctx.mu[j : j + 1], fc, eps=EPS_PSEUDOCOUNT)[0])
        return out
