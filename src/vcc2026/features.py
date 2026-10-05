"""Gene features used to predict the response to knocking down an *unseen* target.

Most target genes in the held-out set will never have been perturbed in any
public screen, so the signature library cannot be queried directly.  We need a
similarity over genes that says "knocking down A should look like knocking down
B", built only from things we are allowed to see.

Two sources, either or both:

* **Context co-expression** -- correlation of the target gene with every
  high-variance gene *across the target line's own control cells*.  Free,
  always available, and context-specific: genes in one complex or one pathway
  co-vary, and that co-variation is what a knockdown collapses.
* **External embeddings** -- protein-language (ESM2), GO/STRING, or any
  pretrained gene embedding, supplied as an ``.npz``.  Both 2025 podium teams
  reported protein embeddings as a consistent win, so they are worth the
  dependency, but the package runs without them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .context import CellContext

logger = logging.getLogger(__name__)


def load_embeddings(path: str | Path) -> dict[str, np.ndarray]:
    """Load ``{gene: vector}`` from an npz with ``genes`` and ``embeddings`` arrays."""
    z = np.load(path, allow_pickle=False)
    genes = z["genes"].astype(str)
    emb = z["embeddings"]
    if emb.shape[0] != genes.size:
        raise ValueError(f"embeddings {emb.shape} do not match {genes.size} genes")
    return {g: emb[i] for i, g in enumerate(genes)}


def _l2_normalise(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    norm[norm == 0] = 1.0
    return x / norm


@dataclass
class GeneFeatures:
    """Row-normalised feature vectors for genes, in a fixed gene order."""

    genes: np.ndarray
    matrix: np.ndarray  # (n_genes, d), L2-normalised rows

    def __post_init__(self) -> None:
        self.genes = np.asarray(self.genes, dtype=object)
        self._index = {g: i for i, g in enumerate(self.genes)}

    def has(self, gene: str) -> bool:
        return gene in self._index

    def vector(self, gene: str) -> np.ndarray | None:
        i = self._index.get(gene)
        return None if i is None else self.matrix[i]

    def similarity(self, gene: str, others: list[str]) -> np.ndarray:
        """Cosine similarity of `gene` against `others` (0 where unknown)."""
        v = self.vector(gene)
        if v is None:
            return np.zeros(len(others))
        idx = np.array([self._index.get(g, -1) for g in others])
        out = np.zeros(len(others))
        known = idx >= 0
        if known.any():
            out[known] = self.matrix[idx[known]] @ v
        return out


def coexpression_features(
    ctx: CellContext,
    n_hvg: int = 2000,
    min_cells: int = 50,
) -> GeneFeatures:
    """Correlation of every gene with the target line's high-variance genes."""
    if ctx.n_control < min_cells:
        logger.warning(
            "only %d control cells in %s -- co-expression features will be noisy",
            ctx.n_control,
            ctx.name,
        )
    var = ctx.var
    k = min(n_hvg, int((var > 0).sum()))
    if k < 2:
        return GeneFeatures(genes=ctx.genes, matrix=np.zeros((ctx.genes.size, 1)))
    hvg = np.argpartition(-var, k - 1)[:k]

    x = ctx.control
    xc = x - x.mean(axis=0, keepdims=True)
    sd = xc.std(axis=0)
    sd[sd == 0] = 1.0
    xz = xc / sd
    n = x.shape[0]
    # (G x k) correlation of all genes against the HVG panel
    corr = (xz.T @ xz[:, hvg]) / max(n - 1, 1)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return GeneFeatures(genes=ctx.genes, matrix=_l2_normalise(corr))


def embedding_features(embeddings: dict[str, np.ndarray], genes: np.ndarray) -> GeneFeatures:
    """Wrap an external embedding dict as `GeneFeatures` over `genes`."""
    genes = np.asarray(genes, dtype=object)
    dim = len(next(iter(embeddings.values()))) if embeddings else 1
    mat = np.zeros((genes.size, dim))
    for i, g in enumerate(genes):
        v = embeddings.get(g)
        if v is not None:
            mat[i] = v
    return GeneFeatures(genes=genes, matrix=_l2_normalise(mat))


def combine(parts: list[tuple[GeneFeatures, float]]) -> GeneFeatures:
    """Concatenate feature blocks with per-block weights, then re-normalise rows."""
    parts = [(f, w) for f, w in parts if w > 0 and f.matrix.shape[1] > 0]
    if not parts:
        raise ValueError("no feature blocks to combine")
    genes = parts[0][0].genes
    blocks = []
    for f, w in parts:
        if not np.array_equal(np.asarray(f.genes), np.asarray(genes)):
            raise ValueError("feature blocks must share a gene order")
        blocks.append(_l2_normalise(f.matrix) * w)
    return GeneFeatures(genes=genes, matrix=_l2_normalise(np.hstack(blocks)))
