"""Gene similarity from a functional interaction network.

The co-expression features computed from a context's own control cells turned
out to carry no usable signal about which knockdowns resemble which: on the 150
measured 2025 signatures they predict a held-out signature *worse* than simply
predicting the average of all the others, and their discrimination score sits at
chance (see `docs/05-実測でわかったこと.md`).  The model shape is not the problem
-- an oracle similarity built from the true signatures reaches r=0.47 and
discrimination 0.75 on the same harness -- so the bottleneck is the gene
representation.

STRING is the cheapest representation to try that encodes what actually matters
here: whether two genes sit in the same complex or pathway, which is what makes
their knockdowns look alike.  It is a download rather than a model, needs no GPU,
and maps directly onto gene symbols.

The similarity used is not the raw edge weight but the *neighbourhood* profile:
two genes are similar when they interact with the same partners, whether or not
they interact with each other.  Members of one complex share partners even where
no direct edge is annotated, and the profile degrades gracefully for genes with
sparse annotation instead of dropping to zero.
"""

from __future__ import annotations

import gzip
import logging
from pathlib import Path

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)


def load_string_network(
    links_path: str | Path,
    info_path: str | Path,
    genes: np.ndarray,
    min_score: int = 400,
) -> sp.csr_matrix:
    """Symmetric gene-by-gene STRING adjacency over `genes`, scores in [0, 1].

    `min_score` is STRING's own confidence scale (400 = medium, 700 = high).
    Below medium the network is dominated by text-mining noise.
    """
    genes = np.asarray(genes, dtype=str)
    pos = {g: i for i, g in enumerate(genes)}

    protein_to_gene: dict[str, int] = {}
    with gzip.open(info_path, "rt") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            i = pos.get(parts[1])
            if i is not None:
                protein_to_gene[parts[0]] = i
    logger.info("STRING: %d proteins map into the gene space", len(protein_to_gene))

    rows, cols, vals = [], [], []
    kept = skipped = 0
    with gzip.open(links_path, "rt") as f:
        next(f)
        for line in f:
            a, b, score = line.split()
            s = int(score)
            if s < min_score:
                continue
            i = protein_to_gene.get(a)
            j = protein_to_gene.get(b)
            if i is None or j is None or i == j:
                skipped += 1
                continue
            rows.append(i)
            cols.append(j)
            vals.append(s / 1000.0)
            kept += 1
    logger.info("STRING: %d edges kept at score >= %d (%d unmapped)", kept, min_score, skipped)

    adj = sp.csr_matrix(
        (np.asarray(vals, dtype=np.float32), (rows, cols)), shape=(genes.size, genes.size)
    )
    adj = adj.maximum(adj.T)
    return adj


def neighbourhood_features(adj: sp.csr_matrix, self_weight: float = 1.0) -> np.ndarray:
    """Row-normalised neighbourhood profiles; cosine between rows is the similarity.

    Adding self-weight keeps directly-interacting genes similar even when their
    partner sets barely overlap, which matters for the small complexes where the
    profile alone is thin.
    """
    mat = adj.tolil(copy=True)
    if self_weight:
        mat.setdiag(self_weight)
    dense = np.asarray(mat.tocsr().todense(), dtype=np.float32)
    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return dense / norms


def coverage(adj: sp.csr_matrix) -> float:
    """Fraction of genes with at least one edge -- how far the network reaches."""
    return float((np.diff(adj.indptr) > 0).mean())
