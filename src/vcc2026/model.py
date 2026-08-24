"""Context-Conditioned Delta Transfer (CCDT).

The predictor is deliberately *not* an end-to-end network.  The 2025 challenge
was won by hybrids of classical statistics and learned components, and the
first-place team's own conclusion was that pure deep models did not reliably
beat statistical baselines on this task.  The structure below keeps every term
separately inspectable and separately calibratable:

    delta(g, c*) = alpha * M(g, c*) * [ b * D(g, c*) + (1 - b) * K(g, c*) ]
                   with the target-gene entry replaced by the on-target model

    D  direct transfer   -- rebase the measured signature of g from every source
                            line into c*, weight by line similarity and by how
                            many cells backed each measurement
    K  program kNN       -- for targets never perturbed anywhere (the majority),
                            predict program loadings from gene features and
                            expand them through the low-rank basis
    b  confidence        -- how much measured signal actually backs D
    M  magnitude         -- per-perturbation effect size, the term the L1
                            discrimination score is most sensitive to
    alpha global scale   -- the one knob that trades MAE against PDS /
                            pearson_delta; tuned on a held-out cell line
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .context import CellContext
from .features import GeneFeatures, coexpression_features, combine, embedding_features
from .knockdown import KnockdownModel
from .library import SignatureLibrary
from .programs import ProgramBasis, neighbour_weights
from .transfer import line_weights, rebase

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    # program basis
    n_components: int = 30
    max_library_targets: int = 4000
    # feature space
    n_hvg: int = 2000
    coexpr_weight: float = 1.0
    embedding_weight: float = 1.0
    # neighbour kernel
    n_neighbours: int = 25
    neighbour_power: float = 3.0
    # transfer
    line_temperature: float = 0.15
    shrink_prior_cells: float = 30.0
    fc_clip: float = 8.0
    # blending / calibration
    trans_similarity_floor: float = 0.0
    coverage_aware_knn: bool = True
    confidence_prior: float = 0.25
    magnitude_gamma: float = 0.5
    shared_shrink: float = 0.0
    fallback_scale: float = 0.35
    alpha: float = 1.0
    min_target_baseline: float = 0.05
    expression_gate: float = 0.05
    seed: int = 0


@dataclass
class Prediction:
    """Predicted deltas plus the per-target provenance needed to debug them."""

    targets: list[str]
    delta: np.ndarray  # (P, G) normlog delta relative to the context baseline
    confidence: np.ndarray  # (P,) how much measured signature backed each target
    magnitude: np.ndarray  # (P,) L2 norm of the final delta
    observed: np.ndarray  # (P,) bool, target present in the signature library
    fallback: np.ndarray  # (P,) bool, no usable neighbours -- mean signature used


class ContextTransferModel:
    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self.library: SignatureLibrary | None = None
        self.knockdown: KnockdownModel | None = None
        self.embeddings: dict[str, np.ndarray] | None = None
        self.features: GeneFeatures | None = None
        self._library_targets: list[str] = []

    # -- training --------------------------------------------------------
    def fit(
        self,
        library: SignatureLibrary,
        embeddings: dict[str, np.ndarray] | None = None,
        features: GeneFeatures | None = None,
    ) -> ContextTransferModel:
        """Training is cheap: the library *is* most of the model.

        Only the on-target knockdown efficiencies are estimated here.  Every
        context-dependent quantity -- the rebased signatures, the program basis,
        the feature space -- is built at prediction time, because all of them
        depend on the target line's baseline, which we only see then.
        """
        self.library = library
        self.knockdown = KnockdownModel.fit(library)
        self.embeddings = embeddings
        # An explicit feature space overrides the context-derived one entirely.
        # Co-expression computed from control cells was measured to carry no
        # signal about which knockdowns resemble which (docs/05), so in practice
        # this is how a usable gene representation gets in.
        self.features = features
        counts: dict[str, float] = {}
        for line in library.lines:
            for t in library.deltas[line]:
                counts[t] = counts.get(t, 0.0) + library.weight(line=line, target=t)
        ranked = sorted(counts, key=lambda t: -counts[t])
        self._library_targets = ranked[: self.config.max_library_targets]
        logger.info(
            "fitted: %d source lines, %d usable target signatures, default knockdown %.2f",
            len(library.lines),
            len(self._library_targets),
            self.knockdown.default,
        )
        return self

    # -- per-context assembly -------------------------------------------
    def _consensus(self, ctx: CellContext) -> tuple[list[str], np.ndarray, np.ndarray]:
        """Rebase every library signature into `ctx` and pool across source lines.

        Pooling is per *gene*, not per signature, because sources do not measure
        the same genes.  The public Replogle releases carry 7,938 of this panel's
        18,533; averaging them against a source that carries 18,077 as though the
        missing 10,600 had been measured as zero drags every prediction toward
        zero exactly there.  Measured: adding 2,057 K562 signatures that way cost
        more than they brought, taking the offline DE-set overlap from 0.147 to
        0.035.  A gene is averaged only over the sources that saw it.
        """
        assert self.library is not None
        cfg = self.config
        lib = self.library
        weights = line_weights(ctx.mu, lib.baseline, temperature=cfg.line_temperature)
        logger.debug("line weights for %s: %s", ctx.name, weights)

        gene_pos = np.array([lib.gene_index(g) for g in ctx.genes])
        known = gene_pos >= 0
        targets = self._library_targets
        pos_of = {t: i for i, t in enumerate(targets)}

        acc = np.zeros((len(targets), ctx.genes.size))
        wsum_gene = np.zeros((len(targets), ctx.genes.size), dtype=np.float32)
        wsum = np.zeros(len(targets))

        for line, lw in weights.items():
            if lw <= 1e-4:
                continue
            present = [t for t in targets if t in lib.deltas[line]]
            if not present:
                continue
            block = np.vstack([lib.deltas[line][t] for t in present])
            mu_src = np.zeros(ctx.genes.size)
            mu_src[known] = lib.baseline[line][gene_pos[known]]
            src = np.zeros((len(present), ctx.genes.size))
            src[:, known] = block[:, gene_pos[known]]
            rebased = rebase(src, mu_src, ctx.mu, fc_clip=cfg.fc_clip)
            w = np.array(
                [
                    lw * lib.weight(line=line, target=t, prior_cells=cfg.shrink_prior_cells)
                    for t in present
                ]
            )
            rows = np.array([pos_of[t] for t in present])
            acc[rows] += rebased * w[:, None]
            measured = np.zeros(ctx.genes.size, dtype=bool)
            measured[known] = lib.baseline[line][gene_pos[known]] != 0
            wsum_gene[np.ix_(rows, np.flatnonzero(measured))] += w[:, None].astype(np.float32)
            wsum[rows] += w

        nz = wsum_gene > 0
        acc[nz] /= wsum_gene[nz]
        acc[~nz] = 0.0
        return targets, acc, wsum, nz

    def _feature_space(self, ctx: CellContext) -> GeneFeatures:
        if self.features is not None:
            return self.features
        cfg = self.config
        blocks: list[tuple[GeneFeatures, float]] = []
        if cfg.coexpr_weight > 0:
            blocks.append((coexpression_features(ctx, n_hvg=cfg.n_hvg), cfg.coexpr_weight))
        if self.embeddings and cfg.embedding_weight > 0:
            blocks.append((embedding_features(self.embeddings, ctx.genes), cfg.embedding_weight))
        return combine(blocks)

    # -- prediction ------------------------------------------------------
    def predict(self, targets: list[str], ctx: CellContext) -> Prediction:
        if self.library is None or self.knockdown is None:
            raise RuntimeError("call fit() before predict()")
        cfg = self.config

        lib_targets, consensus, wsum, covered = self._consensus(ctx)
        usable = wsum > 0
        if usable.sum() < 2:
            raise RuntimeError(
                "no library signature survived transfer into this context -- "
                "check that the source and target gene spaces overlap"
            )
        basis = ProgramBasis.fit(
            [t for t, ok in zip(lib_targets, usable, strict=False) if ok],
            consensus[usable],
            n_components=cfg.n_components,
        )
        features = self._feature_space(ctx)
        train_targets = basis.targets
        usable_rows = np.flatnonzero(usable)
        train_mag = np.linalg.norm(consensus[usable], axis=1)
        row_of = {t: i for i, t in enumerate(train_targets)}
        wsum_of = dict(zip(lib_targets, wsum, strict=False))
        consensus_row_of = {t: i for i, t in enumerate(lib_targets)}

        gate = self._expression_gate(ctx)
        out = np.zeros((len(targets), ctx.genes.size))
        conf = np.zeros(len(targets))
        observed = np.zeros(len(targets), dtype=bool)
        fallback = np.zeros(len(targets), dtype=bool)

        for i, g in enumerate(targets):
            sim = features.similarity(g, train_targets)
            if g in row_of:  # never let a target vote for itself
                sim[row_of[g]] = -np.inf
            nw = neighbour_weights(sim, k=cfg.n_neighbours, power=cfg.neighbour_power)
            if cfg.trans_similarity_floor > 0 and np.nanmax(sim) < cfg.trans_similarity_floor:
                # No neighbour close enough to learn from. The context mean is
                # the honest answer here: it scores exactly 0, where a guess
                # assembled from unrelated signatures scores below it.
                out[i] = self.knockdown.apply(ctx, g, np.zeros(ctx.genes.size))
                conf[i] = 0.0
                observed[i] = g in wsum_of and wsum_of[g] > 0
                continue
            if nw.sum() <= 0:
                # No feature signal at all -- typically a gene with zero variance
                # across this line's control cells.  Predicting a flat zero would
                # be the control baseline, which scores at the bottom of the L1
                # discrimination ranking; a damped mean signature is a strictly
                # better guess and is flagged so it can be audited.
                nw = np.ones(len(train_targets))
                fallback[i] = True
            if cfg.coverage_aware_knn:
                # Average each gene only over the neighbours that measured it.
                # Averaging in the basis' loading space instead treats a gene a
                # source never measured as a measured zero, which drags the
                # prediction toward zero exactly where coverage is thin -- the
                # public Replogle releases carry 7,938 of this panel's 18,533.
                d_knn = _covered_average(consensus, covered, nw, usable_rows)
            else:
                d_knn = basis.reconstruct(basis.predict_loadings(nw))
            if fallback[i]:
                d_knn = d_knn * cfg.fallback_scale

            w_obs = float(wsum_of.get(g, 0.0))
            observed[i] = w_obs > 0
            b = w_obs / (w_obs + cfg.confidence_prior) if w_obs > 0 else 0.0
            conf[i] = b
            if observed[i]:
                d_direct = basis.denoise(consensus[consensus_row_of[g]])
                d = b * d_direct + (1.0 - b) * d_knn
            else:
                d = d_knn
            d = d * gate
            d = self._scale_magnitude(d, g, nw, train_mag, ctx)
            d = self.knockdown.apply(ctx, g, d)
            out[i] = cfg.alpha * d

        if cfg.shared_shrink > 0 and len(targets) > 2:
            # Averaging neighbouring signatures pulls every prediction toward
            # the programme they share, so predictions end up far more alike
            # than real signatures are (measured on context A: median pairwise
            # cosine 0.29 against 0.015 for the measured H1 signatures). The
            # discrimination metric ranks a predicted effect against every real
            # effect, so that shared mass is what makes perturbations
            # indistinguishable. Subtracting part of it trades DE agreement,
            # which the shared programme genuinely carries, for specificity.
            shared = out.mean(axis=0, keepdims=True)
            out = out - cfg.shared_shrink * shared

        return Prediction(
            targets=list(targets),
            delta=out,
            confidence=conf,
            magnitude=np.linalg.norm(out, axis=1),
            observed=observed,
            fallback=fallback,
        )

    # -- helpers ---------------------------------------------------------
    def _expression_gate(self, ctx: CellContext) -> np.ndarray:
        """Damp movement of genes that are effectively off in this context.

        `rebase` already collapses these for the direct term, but the program
        reconstruction can put mass anywhere, and a predicted change on a gene
        that reads zero in every real cell is a pure false positive in the DE
        metrics.
        """
        m0 = self.config.expression_gate
        if m0 <= 0:
            return np.ones(ctx.genes.size)
        return ctx.mu / (ctx.mu + m0)

    def _scale_magnitude(
        self,
        delta: np.ndarray,
        target: str,
        nw: np.ndarray,
        train_mag: np.ndarray,
        ctx: CellContext,
    ) -> np.ndarray:
        """Nudge the effect size towards what neighbouring perturbations show.

        Blending two signatures shrinks the result towards their common part,
        so a kNN-reconstructed delta is systematically *smaller* than a real
        one.  Left uncorrected every predicted effect lands near the weak end of
        the real effect distribution, and the L1 discrimination score matches
        each prediction to whichever real perturbation happens to be weakest.
        `magnitude_gamma` controls how far back towards the neighbour-implied
        size we push: 0 disables it, 1 forces it exactly.
        """
        gamma = self.config.magnitude_gamma
        cur = float(np.linalg.norm(delta))
        if gamma <= 0 or cur <= 0 or nw.sum() <= 0:
            return delta

        want = float((nw @ train_mag) / nw.sum())
        j = ctx.gene_index(target)
        if j is not None:
            # a target that is barely expressed here cannot produce a full effect
            m0 = max(self.config.min_target_baseline, 1e-6)
            want *= float(ctx.mu[j] / (ctx.mu[j] + m0))
        if want <= 0:
            return delta
        return delta * (want / cur) ** gamma


def _covered_average(
    consensus: np.ndarray,
    covered: np.ndarray,
    weights: np.ndarray,
    rows: np.ndarray,
) -> np.ndarray:
    """Weighted mean over neighbours, per gene, skipping sources that never saw it."""
    idx = np.flatnonzero(weights)
    if idx.size == 0:
        return np.zeros(consensus.shape[1])
    src = rows[idx]
    w = weights[idx][:, None]
    mask = covered[src]
    num = (consensus[src] * w * mask).sum(axis=0)
    den = (w * mask).sum(axis=0)
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)
