"""The 2026 challenge bundle: what `vcc datasets download controls` gives you.

Everything here is read from the bundle rather than hard-coded, because the
final phase ships a *different* bundle -- three new cell lines under the labels
D/E/F, with their own control files -- and the one mistake the scorer cannot
detect for you is a context label attached to the wrong cells.  Carrying
validation labels into a final submission scores your model against the wrong
cell lines and looks like a weak model, not a bug.  So the labels are always
taken from the control filenames and the `context` column inside them, never
assumed.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PERT_COL = "target_gene"
CONTEXT_COL = "context"
CONTROL_LABEL = "non-targeting"
CELLS_PER_PERT = 400

_CONTEXT_FILE = re.compile(r"^context_([A-Za-z0-9]+)\.h5ad$")


@dataclass
class Bundle:
    """Paths and metadata for one phase's controls bundle."""

    root: Path
    genes: np.ndarray  # (G,) submission gene order
    perturbations: np.ndarray  # (P,) official target list
    context_files: dict[str, Path]
    cells_per_pert: int = CELLS_PER_PERT
    manifest: dict | None = None

    @property
    def contexts(self) -> list[str]:
        return sorted(self.context_files)

    @property
    def n_cells(self) -> int:
        return len(self.contexts) * self.perturbations.size * self.cells_per_pert

    @classmethod
    def load(cls, root: str | Path) -> Bundle:
        root = Path(root)
        genes = pd.read_csv(root / "gene_names.csv").iloc[:, 0].astype(str).to_numpy()
        perts = pd.read_csv(root / "pert_counts.csv").iloc[:, 0].astype(str).to_numpy()

        context_files: dict[str, Path] = {}
        for path in sorted(root.glob("context_*.h5ad")):
            m = _CONTEXT_FILE.match(path.name)
            if m:
                context_files[m.group(1)] = path
        if not context_files:
            raise FileNotFoundError(f"no context_*.h5ad in {root}")

        manifest = None
        cells_per_pert = CELLS_PER_PERT
        manifest_path = root / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            cells_per_pert = int(manifest.get("cells_per_pert", CELLS_PER_PERT))
            declared = manifest.get("contexts")
            if declared and sorted(declared) != sorted(context_files):
                raise ValueError(
                    f"manifest declares contexts {sorted(declared)} but the bundle has "
                    f"{sorted(context_files)} -- do not guess, re-download the bundle"
                )
        logger.info(
            "bundle %s: %d genes, %d perturbations, contexts %s, %d cells/pert",
            root,
            genes.size,
            perts.size,
            sorted(context_files),
            cells_per_pert,
        )
        return cls(
            root=root,
            genes=genes,
            perturbations=perts,
            context_files=context_files,
            cells_per_pert=cells_per_pert,
            manifest=manifest,
        )

    def check_targets_predictable(self) -> np.ndarray:
        """Which official targets are inside the submission gene space."""
        return np.isin(self.perturbations, self.genes)
