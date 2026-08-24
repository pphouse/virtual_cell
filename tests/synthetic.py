"""A small multi-cell-line CRISPRi simulator, used to exercise the pipeline.

It is built to have the structure the model assumes and nothing more: a shared
low-rank programme basis, target genes that load on those programmes in
clusters, and cell lines that differ in which genes they express.  Cells are
drawn from a latent factor model over the *same* basis, so that the
co-expression features computed from control cells alone carry real
information about which knockdowns should look alike -- which is the
assumption the zero-shot transfer rests on.
"""

from __future__ import annotations

from dataclasses import dataclass

import anndata as ad
import numpy as np
import pandas as pd

CONTROL = "non-targeting"


@dataclass
class SimConfig:
    n_genes: int = 400
    n_programs: int = 8
    n_targets: int = 60
    n_lines: int = 4
    n_cells_per_pert: int = 40
    n_control_cells: int = 600
    depth: int = 4000
    effect_scale: float = 1.2
    on_target_fc: float = 0.3
    program_noise: float = 0.25
    seed: int = 0


def simulate(cfg: SimConfig | None = None):
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed)
    G, K, P, L = cfg.n_genes, cfg.n_programs, cfg.n_targets, cfg.n_lines

    genes = np.array([f"G{i:04d}" for i in range(G)], dtype=object)
    targets = [f"G{i:04d}" for i in rng.choice(G, size=P, replace=False)]
    cluster = rng.integers(0, K, size=P)

    # programme basis: each programme owns a random block of genes
    basis = np.zeros((K, G))
    for k in range(K):
        idx = rng.choice(G, size=G // K, replace=False)
        basis[k, idx] = rng.normal(0, 1, size=idx.size)
        basis[k] /= np.linalg.norm(basis[k])
    # make each target gene load on its own programme, so co-expression is informative
    for p, t in enumerate(targets):
        basis[cluster[p], list(genes).index(t)] += 0.5
    basis /= np.linalg.norm(basis, axis=1, keepdims=True)

    loadings = np.zeros((P, K))
    for p in range(P):
        loadings[p, cluster[p]] = rng.uniform(0.6, 1.4)
        loadings[p] += rng.normal(0, cfg.program_noise, size=K)

    # per-line baseline expression rates, with line-specific silencing
    base_rate = rng.lognormal(0.0, 1.0, size=G)
    lines = {}
    for line in range(L):
        rate = base_rate * rng.lognormal(0.0, 0.5, size=G)
        rate[rng.random(G) < 0.15] = 1e-4  # genes off in this line
        lines[f"line{line}"] = rate

    def draw(rate: np.ndarray, n: int) -> np.ndarray:
        z = rng.normal(0, 1, size=(n, K))
        mod = np.exp(0.35 * (z @ basis))
        lam = rate[None, :] * mod
        lam = cfg.depth * lam / lam.sum(axis=1, keepdims=True)
        return rng.poisson(lam).astype(np.float32)

    out = {}
    for name, rate in lines.items():
        blocks = [draw(rate, cfg.n_control_cells)]
        labels = [np.full(cfg.n_control_cells, CONTROL, dtype=object)]
        for p, t in enumerate(targets):
            fc = np.exp(-cfg.effect_scale * (loadings[p] @ basis))
            fc[list(genes).index(t)] = cfg.on_target_fc
            blocks.append(draw(rate * fc, cfg.n_cells_per_pert))
            labels.append(np.full(cfg.n_cells_per_pert, t, dtype=object))
        x = np.vstack(blocks)
        obs = pd.DataFrame(
            {"target_gene": pd.Categorical(np.concatenate(labels).astype(str))},
            index=np.arange(x.shape[0]).astype(str),
        )
        out[name] = ad.AnnData(X=x, obs=obs, var=pd.DataFrame(index=pd.Index(genes.astype(str))))
    return genes, targets, out
