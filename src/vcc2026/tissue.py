"""Gene similarity from where in the body a gene is expressed.

The third similarity this project tries, after STRING's annotation-derived
neighbourhood profile and DepMap's fitness profile (`coessentiality.py`).  Each
answers "are these two genes related?" from a different kind of evidence, and
the blend of the first two was worth +0.0062 of Overall on two seeds
(`docs/08` 22), so a third axis is worth a screening run.

GTEx publishes the median TPM of every gene in each of 68 tissues.  Two genes
whose expression rises and falls together across those tissues tend to sit in
the same programme -- and unlike the co-expression measured *within* one
context's control cells, which carries no signal at all (`coexpression.py`),
this varies over whole tissues rather than over the noise of individual cells.

The profile is log1p'd before centring, because TPM spans five orders of
magnitude and an uncentred correlation would otherwise be dominated by whether
both genes happen to be highly expressed.  Centring makes a flat, ubiquitously
expressed gene a zero vector, which contributes nothing to any similarity --
the honest answer for a housekeeping gene.
"""

from __future__ import annotations

import gzip
import logging
from pathlib import Path

import numpy as np

from .features import GeneFeatures

logger = logging.getLogger(__name__)


def read_median_tpm(path: str | Path, subset: list[str] | None = None) -> GeneFeatures:
    """Read GTEx's `gene_median_tpm.gct.gz` into cosine-ready gene features.

    The GCT header is three lines -- a version, the dimensions, then the column
    names -- and the first two data columns are the Ensembl id and the symbol.
    Symbols are what the rest of this project keys on, and GTEx repeats a few of
    them across ids; the copy with the larger total expression wins, which picks
    the real locus over a readthrough or a patch scaffold.
    """
    want = None if subset is None else set(map(str, subset))
    rows: dict[str, np.ndarray] = {}
    with gzip.open(path, "rt") as fh:
        fh.readline()
        fh.readline()
        header = fh.readline().rstrip("\n").split("\t")
        n_tissue = len(header) - 2
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            symbol = parts[1]
            if want is not None and symbol not in want:
                continue
            values = np.asarray(parts[2:], dtype=np.float32)
            prior = rows.get(symbol)
            if prior is None or values.sum() > prior.sum():
                rows[symbol] = values

    if not rows:
        raise ValueError(f"no requested gene found in {path}")
    genes = np.array(sorted(rows), dtype=str)
    x = np.log1p(np.vstack([rows[g] for g in genes]))
    centred = x - x.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centred, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    logger.info("tissue features: %d genes over %d tissues", genes.size, n_tissue)
    return GeneFeatures(genes=np.asarray(genes, dtype=object), matrix=centred / norms)
