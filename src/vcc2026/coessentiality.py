"""Gene similarity from how knockouts kill cells, rather than from annotation.

STRING says two genes are related when a database says so.  DepMap says it when
knocking each of them out changes fitness the same way across a thousand cell
lines -- a measurement, made by the same kind of experiment this challenge is
about, and one that does not depend on anyone having curated the pair.  Genes in
one complex have near-identical fitness profiles even where no interaction is
annotated, which is exactly the case STRING's neighbourhood profile handles
worst.

The similarity is the correlation of two genes' gene-effect profiles across cell
lines.  Each profile is centred and L2-normalised here, so the cosine the model
takes is that correlation.  Lines that did not screen a gene leave NaN; those
entries become 0 after centring, which is the neutral contribution.

This is the one representation left to try.  The co-expression of a context's
own control cells was measured and carries nothing (`coexpression.py`), and
every knob on the STRING representation -- confidence threshold, neighbour
count, neighbour weighting, low-rank width -- was swept against `pds` at the
competition's panel size and none of them agreed in sign across two cell lines
(`docs/08` §17).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .features import GeneFeatures

logger = logging.getLogger(__name__)


def read_gene_effect(path: str | Path, subset: list[str] | None = None) -> GeneFeatures:
    """Read DepMap's `CRISPRGeneEffect.csv` into cosine-ready gene features.

    The file is cell lines by genes, with columns named `SYMBOL (ENTREZ)`; the
    symbol is what the rest of this project keys on.  Only `subset` is kept when
    given, which is what makes this cheap: the library's targets and the panel's
    are a few thousand of the eighteen thousand columns.
    """
    import pandas as pd

    want = None if subset is None else set(map(str, subset))
    frame = pd.read_csv(path, index_col=0)
    symbols = np.array([c.split(" (")[0] for c in frame.columns], dtype=str)
    keep = np.ones(symbols.size, dtype=bool) if want is None else np.isin(symbols, list(want))
    x = frame.to_numpy(dtype=np.float32).T[keep]
    genes = symbols[keep]

    # A gene screened in no line, or in one, carries no correlation; drop it
    # rather than let a zero vector answer every similarity query with 0.
    seen = np.isfinite(x).sum(axis=1)
    ok = seen >= 2
    x, genes = x[ok], genes[ok]

    centred = np.where(np.isfinite(x), x, np.nan)
    mean = np.nanmean(centred, axis=1, keepdims=True)
    centred = np.nan_to_num(centred - mean, nan=0.0)
    norms = np.linalg.norm(centred, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    logger.info(
        "co-essentiality features: %d genes over %d cell lines (%d columns dropped)",
        genes.size,
        x.shape[1],
        int((~ok).sum()),
    )
    return GeneFeatures(genes=np.asarray(genes, dtype=object), matrix=centred / norms)


def load_features(path: str | Path, subset: list[str] | None = None) -> GeneFeatures:
    """Read the cached npz written by `scripts/build_coessentiality.py`."""
    z = np.load(path, allow_pickle=False)
    genes, matrix = z["genes"].astype(str), z["matrix"]
    if subset is not None:
        keep = np.isin(genes, list(map(str, subset)))
        genes, matrix = genes[keep], matrix[keep]
    logger.info("co-essentiality features: %d of the requested genes are covered", genes.size)
    return GeneFeatures(genes=np.asarray(genes, dtype=object), matrix=matrix)


def blend(a: GeneFeatures, b: GeneFeatures, weight: float) -> GeneFeatures:
    """One similarity from two, over the genes both cover.

    The two feature spaces are unrelated -- a STRING neighbourhood profile and a
    fitness profile across cell lines -- so they are concatenated rather than
    added, with `weight` scaling the second block before the rows are
    re-normalised.  A cosine over the result is then the weighted sum of the two
    cosines, which is what "use both" should mean.  A gene either source cannot
    place contributes a zero block rather than being dropped.
    """
    genes = np.asarray(sorted(set(map(str, a.genes)) | set(map(str, b.genes))), dtype=str)
    left = np.zeros((genes.size, a.matrix.shape[1]), dtype=np.float32)
    right = np.zeros((genes.size, b.matrix.shape[1]), dtype=np.float32)
    ai = {str(g): i for i, g in enumerate(a.genes)}
    bi = {str(g): i for i, g in enumerate(b.genes)}
    for k, g in enumerate(genes):
        if g in ai:
            left[k] = a.matrix[ai[g]]
        if g in bi:
            right[k] = b.matrix[bi[g]]
    both = np.hstack([left, float(weight) * right])
    norms = np.linalg.norm(both, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    logger.info(
        "blended features: %d genes, %d from both sources, weight %.2f",
        genes.size,
        int(sum(1 for g in genes if g in ai and g in bi)),
        weight,
    )
    return GeneFeatures(genes=np.asarray(genes, dtype=object), matrix=both / norms)


def neighbourhood(features: GeneFeatures, k: int = 50) -> GeneFeatures:
    """Re-express co-essentiality the way STRING is used: who your partners are.

    STRING's gain over raw edges came from comparing *neighbourhood profiles*
    rather than direct links -- two genes are alike when they interact with the
    same partners, whether or not they interact with each other.  The same
    transform applies to a fitness profile: keep each gene's `k` most correlated
    partners, zero the rest, and compare those sparse partner sets.  It throws
    away the long tail of weak correlations, which is where a 1,178-line profile
    is least trustworthy.
    """
    sim = features.matrix @ features.matrix.T
    np.fill_diagonal(sim, 0.0)
    n = sim.shape[0]
    keep = min(k, n - 1)
    cut = np.partition(sim, -keep, axis=1)[:, -keep][:, None]
    rows = np.where(sim >= cut, sim, 0.0).astype(np.float32)
    np.fill_diagonal(rows, 1.0)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    logger.info("co-essentiality neighbourhood: %d genes, top %d partners each", n, keep)
    return GeneFeatures(genes=features.genes, matrix=rows / norms)
