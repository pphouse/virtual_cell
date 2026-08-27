"""Predict a *call set*, because that is most of what the scorer reads.

Three of the four differential-expression members are decided as much by how
many genes a submission declares as by which ones.  Fidelity divides the correct
calls by `max(n_pred, n_real)`, the Jaccard divides the agreement by the union,
and reach walks the reference's own significant genes in the order the
submission's confidence puts them.  A prediction that is directionally right but
declares ten thousand genes for a perturbation that moved two scores near zero
on all three -- which is what the third leaderboard round measured, at
`de_wilcoxon_sig_jaccard` 0.0255 against a mean-response baseline of 0.029.

So the prediction is built the other way round: rank the genes, take a budget of
them, and give the rest a fold change of exactly one so the emitted cells pass
the control's own distribution through untouched and the Wilcoxon test cannot
call them.  The ranking is a propensity -- how likely this gene is to be called
for *any* knockdown, which is mostly how well the assay can see it -- times the
size of the response, generic or target-specific.  The propensity needs no
source screen at all: it comes from the context's own control cells.

The magnitude assigned to a called gene is the fold change at which a gene of
that expression becomes significant, times a margin.  That is a narrow ledge to
stand on: too small and the call does not happen, too large and
`de_wilcoxon_lfc_nmae` charges for the overshoot.  It is also, measured on the
2025 H1 screen at the competition's depth, close to the median real effect --
the significance threshold and the typical true response are the same size, so
the two requirements do not actually conflict.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)

LN2 = float(np.log(2.0))

# Measured on the 2025 H1 screen cut to the competition's shape (400 cells per
# perturbation against 18,400 controls, median 20,000 UMI): the |log2 fold
# change| at which a gene of a given control expression is called significant
# half the time is 1.314 * cpm ** -0.456 over the range the gate admits.
# `scripts/significance_threshold.py` reproduces the fit.
THRESHOLD_SCALE = 1.314
THRESHOLD_EXPONENT = -0.456

# Least squares on [1, log10 cpm, log10 fano, their product] against the
# fraction of the 60 reference perturbations calling each gene significant
# (Spearman 0.63 against the measured propensity).  The relationship is a
# property of the assay rather than of H1, which is why it is carried across.
PROPENSITY_COEF = (-1.196823, 0.275022, 0.716629, -0.136090)


@dataclass
class BudgetConfig:
    n_calls: int = 8000
    margin: float = 1.35
    top_margin: float = 2.0
    min_cpm: float = 5.0
    max_abs_log2fc: float = 1.5
    min_abs_log2fc: float = 0.05
    # A target-specific signal, when there is one, is worth more than the generic
    # response: measured K562+RPE1 -> H1 (a different lab and a different line,
    # which is the transfer the challenge asks for), directional accuracy over
    # the top 50 ranked genes is 0.611 against the generic's 0.562, and it still
    # leads at every depth.  It is also the only part of the prediction that
    # differs between perturbations, so it is the only part `pds_cosine` can see.
    #
    # Mixing the generic response back in was measured on the leaderboard and it
    # LOSES: at a generic weight of 2 the fidelity gain the panel promised was
    # real (+0.022) and pds fell by 0.100, because a dominant shared component
    # makes every target declare nearly the same genes and pds is a retrieval
    # task over 300 of them.  Weights are ratios -- each component is put on unit
    # scale in `_score` before it is weighted -- so 2 meant twice the generic,
    # whatever units it arrived in.  Measured on K562 at the competition's panel
    # size, pds falls monotonically in that ratio: 0.559 at 0, 0.559 at 0.25,
    # 0.554 at 0.5, 0.540 at 1, 0.528 at 2.  Keep it at 0.
    specific_weight: float = 0.0
    generic_weight: float = 1.0
    # Relative to the mean |signal|, how hard to pull a target's genomic
    # neighbours toward silence. pds rises from 0.689 to 0.705 between 0 and 20
    # and is flat above that, so the value is a plateau rather than a peak.
    proximity_weight: float = 0.0
    # How hard to let a target's own ranking score set the size of its calls, not
    # just their signs.  0 keeps the size a function of expression alone.
    magnitude_gamma: float = 0.0
    max_magnitude_scale: float = 3.0
    propensity_coef: tuple[float, float, float, float] = field(default=PROPENSITY_COEF)
    seed: int = 0


def control_statistics(counts: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Mean CPM and Fano factor per gene over a block of control cells."""
    lib = np.asarray(counts.sum(axis=1)).ravel()
    scale = sp.diags(1e6 / np.maximum(lib, 1.0))
    cpm = (scale @ counts).tocsc()
    n = counts.shape[0]
    mean = np.asarray(cpm.sum(axis=0)).ravel() / n
    second = np.asarray(cpm.multiply(cpm).sum(axis=0)).ravel() / n
    var = np.maximum(second - mean**2, 0.0)
    return mean, var / np.maximum(mean, 1e-9)


def propensity(mean_cpm: np.ndarray, fano: np.ndarray, coef) -> np.ndarray:
    """How often a gene of these control statistics is called for *some* knockdown."""
    lm = np.log10(mean_cpm + 1e-3)
    lf = np.log10(fano + 1e-9)
    a, b, c, d = coef
    return np.clip(a + b * lm + c * lf + d * lm * lf, 1e-4, 1.0)


def significance_magnitude(mean_cpm: np.ndarray, margin: float) -> np.ndarray:
    """|log2 fold change| a gene of this expression needs to be called."""
    with np.errstate(divide="ignore"):
        thr = THRESHOLD_SCALE * np.power(np.maximum(mean_cpm, 1e-6), THRESHOLD_EXPONENT)
    return margin * thr


@dataclass
class BudgetedPrediction:
    """What a budgeted prediction asserts, in the four forms the scorer reads."""

    lfc: np.ndarray
    """Natural-log fold change, zero outside the call set."""
    key: np.ndarray
    """Ranking score; the order the submission's p-values will come out in."""
    sign: np.ndarray
    """Predicted direction, defined everywhere, read only on the call set."""
    call: np.ndarray
    """The genes the submission declares as responding."""

    def log2(self) -> np.ndarray:
        return self.lfc / LN2


class BudgetedPredictor:
    """A per-target natural-log fold change with a fixed number of non-zero entries."""

    def __init__(
        self, cfg: BudgetConfig, generic_log2fc: np.ndarray, specific=None, proximity=None
    ) -> None:
        self.cfg = cfg
        self.generic = np.asarray(generic_log2fc, dtype=np.float64)
        self.specific = specific
        self.proximity = proximity

    def fit(self, counts: sp.csr_matrix, genes: np.ndarray) -> BudgetedPredictor:
        self.genes = np.asarray(genes)
        self._generic_scale = float(np.mean(np.abs(self.generic))) or 1.0
        self.mean_cpm, self.fano = control_statistics(counts)
        self.tested = self.mean_cpm > self.cfg.min_cpm
        self.prop = propensity(self.mean_cpm, self.fano, self.cfg.propensity_coef)
        self.magnitude = np.clip(
            significance_magnitude(self.mean_cpm, self.cfg.margin),
            self.cfg.min_abs_log2fc,
            self.cfg.max_abs_log2fc,
        )
        self._index = {str(g): i for i, g in enumerate(self.genes)}
        logger.info(
            "context: %d of %d genes above %.1f CPM; median call magnitude %.3f log2",
            int(self.tested.sum()),
            self.genes.size,
            self.cfg.min_cpm,
            float(np.median(self.magnitude[self.tested])),
        )
        return self

    def _score(
        self, row_specific: np.ndarray | None, target: str, specific_scale: float = 1.0
    ) -> tuple[np.ndarray, np.ndarray]:
        # Each component is put on unit scale before it is weighted, so a weight
        # is a mixing ratio rather than an accident of what units the component
        # happens to arrive in -- the generic response measured as a log2 fold
        # change is six times the size of the same response as a library delta,
        # and a ratio fitted in one would be wrong by that factor in the other.
        # A component is scaled by *its own* global mean rather than per target,
        # which is the whole point of mixing: a target the transfer model has
        # little to say about keeps its small specific signal and lets the
        # generic response carry the ranking.
        signal = self.cfg.generic_weight * self.generic / self._generic_scale
        if row_specific is not None and self.cfg.specific_weight:
            signal = signal + self.cfg.specific_weight * row_specific / specific_scale
        if self.proximity is not None and self.cfg.proximity_weight:
            # Scaled against the signal it is joining, so the weight means the
            # same thing whatever the transfer model happens to be emitting.
            unit = float(np.mean(np.abs(signal))) or 1.0
            signal = signal + self.cfg.proximity_weight * unit * self.proximity.weights(target)
        return self.prop * np.abs(signal) * self.tested, np.sign(signal)

    def predict(self, targets: list[str]) -> BudgetedPrediction:
        """The call set, its direction, its confidence ordering and its magnitude.

        The emitter only needs the fold changes, but every DE member reads one of
        the other three as well -- the Jaccard reads the set, fidelity reads the
        direction, reach reads the ordering -- so they are returned together and
        the offline validator scores exactly what the submission would express.
        """
        specific = None
        specific_scale = 1.0
        if self.specific is not None:
            specific = np.asarray(self.specific.predict_log2fc(targets), dtype=np.float64)
            specific_scale = float(np.mean(np.abs(specific))) or 1.0
        shape = (len(targets), self.genes.size)
        out = np.zeros(shape, dtype=np.float32)
        key = np.zeros(shape, dtype=np.float32)
        direction = np.zeros(shape, dtype=np.float32)
        call = np.zeros(shape, dtype=bool)
        for i, target in enumerate(targets):
            score, sign = self._score(
                None if specific is None else specific[i], target, specific_scale
            )
            own = self._index.get(str(target))
            if own is not None:
                # Never scored -- every member drops the perturbation's own target
                # gene -- so spending a call on it would waste one.
                score[own] = -1.0
            key[i] = score
            direction[i] = sign
            order = np.argsort(-score)
            take = order[: min(self.cfg.n_calls, order.size)]
            take = take[score[take] > 0.0]
            call[i, take] = True
            # Reach walks the reference's responding genes in the order the
            # submission's own p-values put them, so confidence has to be
            # *expressed* -- a flat magnitude leaves that ordering to expression
            # level and the metric reads it as no confidence at all. The calls
            # are graded from `top_margin` at the head down to `margin` at the
            # tail, which keeps the weakest call above its significance
            # threshold while giving the strongest the smallest p-value.
            if take.size:
                ramp = np.linspace(self.cfg.top_margin, self.cfg.margin, take.size)
                scale = ramp / self.cfg.margin
                if self.cfg.magnitude_gamma:
                    # The magnitude assigned to a call is otherwise a function of
                    # the gene's expression alone, so every target emits the same
                    # sizes and differs only in signs.  `pds` ranks a predicted
                    # profile against 300 real ones, and shared coordinates carry
                    # no information for that.  Scaling by the target's own signal
                    # -- never below 1, so a call stays above its significance
                    # threshold -- puts target-specific structure into the sizes.
                    rel = score[take] / max(float(np.median(score[take])), 1e-12)
                    scale = scale * np.clip(
                        np.power(rel, self.cfg.magnitude_gamma),
                        1.0,
                        self.cfg.max_magnitude_scale,
                    )
                out[i, take] = (
                    sign[take] * self.magnitude[take] * scale * LN2
                ).astype(np.float32)
        return BudgetedPrediction(lfc=out, key=key, sign=direction, call=call)

    def predict_lfc(self, targets: list[str]) -> np.ndarray:
        """Natural-log fold change per target, zero everywhere outside the budget."""
        return self.predict(targets).lfc
