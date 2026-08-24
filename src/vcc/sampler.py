"""Turning predicted pseudobulk deltas back into single cells.

cell-eval scores two different kinds of thing, and they want opposite things
from this step:

* the expression metrics (MAE, pearson_delta, the L1 discrimination score) are
  computed on *per-perturbation means*, so they are completely indifferent to
  how the spread is distributed -- as long as the mean is preserved exactly;
* the DE metrics run a real differential-expression test on the emitted cells,
  so they depend entirely on the within-perturbation spread.  Emit identical
  cells and every gene is significant at p ~ 0; emit too much noise and nothing
  is.

The sampler therefore builds each cell as ``control_cell + r_i * delta`` where
the ``r_i`` are drawn from a mixture -- a non-responder spike at zero (CRISPRi
escapees are real and are why perturbed populations are bimodal) plus a Gamma
for the responders -- and are then renormalised so that ``mean(r) == 1``
exactly.  That makes the pseudobulk mean *exactly* the predicted delta,
regardless of the heterogeneity settings, so the two families of metrics can be
tuned independently instead of fighting each other.
"""

from __future__ import annotations

from dataclasses import dataclass

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .context import CONTROL_NAME, CellContext
from .model import Prediction
from .normalize import renormalise_rows


@dataclass
class SamplerConfig:
    non_responder_fraction: float = 0.15
    response_shape: float = 4.0  # Gamma shape; smaller = more heterogeneous responders
    holdout_controls: float = 0.5  # fraction of the control pool reserved for the control arm
    mean_match_iters: int = 12
    renormalise: bool = False
    seed: int = 0


class CellSampler:
    def __init__(self, config: SamplerConfig | None = None) -> None:
        self.config = config or SamplerConfig()

    def response_scales(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Per-cell response multipliers with mean exactly 1."""
        cfg = self.config
        pi0 = float(np.clip(cfg.non_responder_fraction, 0.0, 0.95))
        responder = rng.random(n) >= pi0
        r = np.zeros(n)
        k = max(cfg.response_shape, 1e-3)
        n_resp = int(responder.sum())
        if n_resp == 0:  # degenerate draw: fall back to all-responders
            responder[:] = True
            n_resp = n
        r[responder] = rng.gamma(shape=k, scale=1.0 / k, size=n_resp)
        mean = r.mean()
        return r / mean if mean > 0 else np.ones(n)

    @staticmethod
    def _match_mean(x: np.ndarray, target: np.ndarray, iters: int) -> np.ndarray:
        """Set each column's mean to `target` exactly, keeping every value >= 0.

        Expression cannot go below zero, and the clip lands hardest on exactly
        the genes with the largest true effect -- so a naive
        ``control + r * delta`` biases the pseudobulk mean towards zero right
        where the metrics care most.  A shift cannot fix that: shifting a
        column down re-clips and leaves the mean too high whenever a few cells
        carry all the mass.

        So the correction is directional.  Raising a mean is a shift, which
        preserves the shape of the distribution and cannot break
        non-negativity.  Lowering one is a rescale, which is exact and also
        non-negative -- and shrinking the spread along with the mean is what a
        real knockdown does anyway.  Both are one-step exact; the loop only
        exists to mop up float error.
        """
        np.clip(x, 0.0, None, out=x)
        target = np.clip(target, 0.0, None)
        for _ in range(max(iters, 1)):
            cur = x.mean(axis=0)
            resid = target - cur
            if np.abs(resid).max() < 1e-9:
                break
            up = resid > 0
            if up.any():
                x[:, up] += resid[up]
            down = (resid < 0) & (cur > 0)
            if down.any():
                x[:, down] *= target[down] / cur[down]
            # columns that must fall but are already all-zero cannot move
            np.clip(x, 0.0, None, out=x)
        return x

    def sample(
        self,
        ctx: CellContext,
        prediction: Prediction,
        n_cells: dict[str, int],
        n_control_cells: int | None = None,
    ) -> ad.AnnData:
        """Emit the submission AnnData: predicted cells plus a control arm."""
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)

        n_ctrl = ctx.n_control
        perm = rng.permutation(n_ctrl)
        n_hold = round(float(np.clip(cfg.holdout_controls, 0.0, 0.9)) * n_ctrl)
        n_hold = min(max(n_hold, 1), n_ctrl - 1)
        control_arm_idx = perm[:n_hold]
        basis_idx = perm[n_hold:]

        blocks: list[np.ndarray] = []
        labels: list[np.ndarray] = []

        for i, target in enumerate(prediction.targets):
            n = int(n_cells.get(target, 0))
            if n <= 0:
                continue
            pick = rng.choice(basis_idx, size=n, replace=n > basis_idx.size)
            base = ctx.control[pick]
            r = self.response_scales(n, rng)
            x = base + r[:, None] * prediction.delta[i][None, :]
            # Anchor on the *full* control pool mean, not the resampled subset's.
            # Resampling a few hundred cells out of the pool injects sampling
            # noise straight into the pseudobulk mean that MAE and pearson_delta
            # are computed from -- noise that has nothing to do with the model
            # and that the control-mean baseline does not pay.
            target_mean = np.clip(ctx.mu + prediction.delta[i], 0.0, None)
            x = self._match_mean(x, target_mean, cfg.mean_match_iters)
            if cfg.renormalise:
                x = renormalise_rows(x, ctx.target_sum)
            blocks.append(x.astype(np.float32))
            labels.append(np.full(n, target, dtype=object))

        n_ctrl_out = n_control_cells if n_control_cells is not None else control_arm_idx.size
        pick = rng.choice(
            control_arm_idx, size=n_ctrl_out, replace=n_ctrl_out > control_arm_idx.size
        )
        ctrl_block = ctx.control[pick].copy()
        # Same argument for the control arm: every delta metric subtracts this
        # mean, so any noise in it is charged to all P perturbations at once.
        ctrl_block = self._match_mean(ctrl_block, np.clip(ctx.mu, 0.0, None), cfg.mean_match_iters)
        blocks.append(ctrl_block.astype(np.float32))
        labels.append(np.full(n_ctrl_out, CONTROL_NAME, dtype=object))

        x = np.vstack(blocks)
        obs = pd.DataFrame(
            {
                "target_gene": pd.Categorical(np.concatenate(labels).astype(str)),
                "celltype": pd.Categorical(np.full(x.shape[0], ctx.name)),
            },
            index=np.arange(x.shape[0]).astype(str),
        )
        return ad.AnnData(
            X=sp.csr_matrix(x),
            obs=obs,
            var=pd.DataFrame(index=pd.Index(np.asarray(ctx.genes, dtype=str))),
        )
