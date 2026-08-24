"""Writing and validating a challenge submission.

The submission is an ``.h5ad`` that ``cell-eval prep`` converts into the
``.vcc`` tarball the leaderboard accepts.  Everything ``prep`` checks is
checked here first, because a rejected upload during the final window is an
expensive way to discover a column name.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from .context import CONTROL_NAME

logger = logging.getLogger(__name__)

PERT_COL = "target_gene"
CELLTYPE_COL = "celltype"


def read_gene_list(path: str | Path) -> np.ndarray:
    """Read the challenge gene list (single-column, no header) in file order."""
    df = pd.read_csv(path, header=None)
    return df.iloc[:, 0].astype(str).to_numpy()


def read_pert_counts(path: str | Path) -> dict[str, int]:
    """Read ``pert_counts_*.csv`` into ``{target_gene: n_cells}``.

    The file names how many cells to emit per perturbation; emitting a
    different number is the most common reason a submission is rejected.
    """
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    pert = cols.get("target_gene") or cols.get("perturbation") or df.columns[0]
    count = cols.get("n_cells") or cols.get("count") or df.columns[-1]
    return {str(k): int(v) for k, v in zip(df[pert], df[count], strict=False)}


def validate(
    adata: ad.AnnData,
    genes: np.ndarray | None = None,
    max_cells: int = 200_000,
    control_name: str = CONTROL_NAME,
) -> None:
    """Fail loudly on everything `cell-eval prep` would reject."""
    problems: list[str] = []
    if PERT_COL not in adata.obs:
        problems.append(f"obs is missing {PERT_COL!r}")
    elif control_name not in set(adata.obs[PERT_COL].astype(str)):
        problems.append(f"no {control_name!r} cells present")
    if genes is not None:
        got = np.asarray(adata.var_names, dtype=str)
        want = np.asarray(genes, dtype=str)
        if got.size != want.size:
            problems.append(f"gene dimension {got.size} != expected {want.size}")
        elif set(got) != set(want):
            missing = sorted(set(want) - set(got))[:5]
            problems.append(f"gene set differs, e.g. missing {missing}")
        elif not np.array_equal(got, want):
            problems.append("genes present but out of order (prep will reorder; fix upstream)")
    if adata.n_obs > max_cells:
        problems.append(f"{adata.n_obs} cells exceeds the {max_cells} cap")
    if not np.isfinite(adata.X.data if hasattr(adata.X, "data") else adata.X).all():
        problems.append("X contains non-finite values")
    if problems:
        raise ValueError("submission validation failed:\n  - " + "\n  - ".join(problems))
    logger.info("submission OK: %d cells x %d genes", adata.n_obs, adata.n_vars)


def write_submission(
    adata: ad.AnnData,
    path: str | Path,
    genes: np.ndarray | None = None,
    gene_list_path: str | Path | None = None,
    run_prep: bool = True,
    max_cells: int = 200_000,
) -> Path:
    """Validate, write the h5ad, and (optionally) run ``cell-eval prep``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    validate(adata, genes=genes, max_cells=max_cells)
    adata.write_h5ad(path)
    logger.info("wrote %s", path)

    if not run_prep:
        return path
    if shutil.which("cell-eval") is None:
        logger.warning("cell-eval not on PATH; skipping prep (pip install cell-eval)")
        return path
    if gene_list_path is None:
        raise ValueError("gene_list_path is required to run cell-eval prep")
    out = path.with_suffix(".prep.vcc")
    cmd = [
        "cell-eval",
        "prep",
        "-i",
        str(path),
        "-g",
        str(gene_list_path),
        "-o",
        str(out),
        "--pert-col",
        PERT_COL,
    ]
    if CELLTYPE_COL in adata.obs:
        cmd += ["--celltype-col", CELLTYPE_COL]
    logger.info("running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out
