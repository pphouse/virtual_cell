"""The one signal in this model that does not have to survive a context switch.

Everything else here carries a measured transcriptional response from one cell
line to another, and the measurements say that barely works: the direction of a
generic knockdown response is 0.53 to 0.56 accurate across a line, and no
hyperparameter moves it.  This is different in kind.  CRISPRi silences by
tethering a KRAB domain, which deposits H3K9me3 that spreads from the guide, and
where it spreads is a property of the *reagent* and of local sequence rather than
of which cell line the experiment was done in.  So the genes sitting next to a
target go down when it is knocked down, and gene coordinates are the same
everywhere.

Measured on the 2025 H1 screen at the competition's depth, over 60 knockdowns
(`scripts/proximity_effect.py`):

| distance   | pairs | P(significant) | P(down) | mean log2FC |
|------------|-------|----------------|---------|-------------|
| < 10 kb    |     8 | 0.500          | 0.375   | -0.376      |
| 10-25 kb   |    17 | 0.353          | 0.824   | -1.040      |
| 25-50 kb   |    34 | 0.265          | 0.647   | -0.357      |
| 50-100 kb  |    60 | 0.217          | 0.633   | -0.173      |
| 100-200 kb |   119 | 0.118          | 0.571   | -0.041      |
| 200-500 kb |   302 | 0.136          | 0.517   | -0.018      |
| 0.5-1 Mb   |   474 | 0.095          | 0.508   |  0.003      |
| other chr  |  602k | 0.096          | 0.498   | -0.001      |

Against a background rate of 0.096, the effect peaks at 10 to 25 kb and is spent
by about 200 kb.

That scale is why published TAD calls do not help, which was tested rather than
assumed (`scripts/tad_boundaries.py`).  Counting the Rao 2014 Arrowhead domain
boundaries between a target and its neighbour does NOT separate the pairs that
feel the effect from the ones that do not: within 100 kb, GM12878 gives P(down)
0.600 with no boundary between and 0.680 with one or more, and IMR90 gives 0.605
against 0.808.  Crossing a boundary leaves the effect intact.  Those domains
average a quarter of a megabase and this effect lives an order of magnitude
below that, so a boundary is simply the wrong ruler.  A smooth exponential is
used instead, and `pds` is flat in the decay length anywhere between 20 and
100 kb, so the choice is not delicate.

It is a small set -- a median of six genes per perturbation within 500 kb --
which is why it cannot move the DE members, whose denominators run to a
thousand.  It moves `pds_cosine`, which ranks predictions against each other by
cosine and so pays for coordinates belonging to one target and nothing else.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# The decay length of the pull toward silence. Fitted by eye to the table above
# rather than by least squares: the effect is over by 500 kb and half gone by
# 100, and pds is flat between 1e5 and 2e5 so the choice is not delicate.
DECAY_BP = 1.0e5


def load_gene_positions(path: str | Path) -> dict[str, tuple[str, float]]:
    """Read a {symbol: [chromosome, midpoint]} table."""
    raw = json.loads(Path(path).read_text())
    return {str(k): (str(v[0]), float(v[1])) for k, v in raw.items()}


class ProximitySignal:
    """A signed, distance-decayed pull toward silence around each target."""

    def __init__(
        self,
        genes: np.ndarray,
        positions: dict[str, tuple[str, float]],
        decay_bp: float = DECAY_BP,
        floor: float = 0.02,
    ) -> None:
        self.genes = np.asarray(genes)
        self.decay_bp = float(decay_bp)
        self.floor = float(floor)
        self._positions = positions
        self._index = {str(g): i for i, g in enumerate(self.genes)}
        self._chrom = np.array([positions.get(str(g), ("", 0.0))[0] for g in self.genes])
        self._pos = np.array([positions.get(str(g), ("", np.nan))[1] for g in self.genes])
        placed = self._chrom != ""
        logger.info(
            "proximity: %d of %d genes placed on the genome, decay %.0f kb",
            int(placed.sum()),
            self.genes.size,
            self.decay_bp / 1e3,
        )

    def weights(self, target: str) -> np.ndarray:
        """Non-positive per-gene weights; zero for the target's own gene."""
        out = np.zeros(self.genes.size)
        where = self._positions.get(str(target))
        if where is None:
            return out
        chrom, centre = where
        same = self._chrom == chrom
        if not same.any():
            return out
        decay = np.exp(-np.abs(self._pos[same] - centre) / self.decay_bp)
        decay[decay < self.floor] = 0.0
        out[same] = -decay
        own = self._index.get(str(target))
        if own is not None:
            out[own] = 0.0
        return out
