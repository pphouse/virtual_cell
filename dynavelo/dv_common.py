"""Shared helpers to run the unmodified DynaVelo repo on current torch / pandas."""
import os
import sys
import numpy as np
import scanpy as sc
import torch

ROOT = '/home/azureuser/vcc'
OUT = f'{ROOT}/results'
DV = f'{ROOT}/DynaVelo/dynavelo'
DATASET, SAMPLE = 'PBMC', 'cluster0_3'

# DynaVelo writes to ../checkpoints and ../log relative to the working directory
os.chdir(DV)
sys.path.insert(0, DV)

# torch>=2.7 dropped the `verbose` kwarg that dynavelo/utils.py passes
_RLROP = torch.optim.lr_scheduler.ReduceLROnPlateau
class _ReduceLROnPlateau(_RLROP):
    def __init__(self, *args, verbose=None, **kwargs):
        super().__init__(*args, **kwargs)
torch.optim.lr_scheduler.ReduceLROnPlateau = _ReduceLROnPlateau

import models  # noqa: E402
import torchdiffeq  # noqa: E402
from models import MultiomeDataset as _MultiomeDataset, DynaVelo  # noqa: E402

# Upstream uses odeint_adjoint. Direct back-propagation through dopri5 gives the same loss/gradients
# (verified: identical losses) and is ~5x faster on this machine; the state is only 100-d so memory is fine.
models.odeint = torchdiffeq.odeint


class MultiomeDataset(_MultiomeDataset):
    """Same as upstream, but with positional cell-type lookup (pandas>=2 label-indexing change)."""

    def __init__(self, adata_rna, adata_atac, use_weights=True):
        super().__init__(adata_rna, adata_atac, use_weights=use_weights)
        if use_weights:
            ct = adata_rna.obs['celltype'].astype(str).values
            self._w = np.array([self.weights_dict[c] for c in ct], dtype=np.float32)

    def __getitem__(self, idx):
        x = torch.tensor(self.x_rna[idx, :], dtype=torch.float32)
        y = torch.tensor(self.x_atac[idx, :], dtype=torch.float32)
        vx = torch.tensor(self.vx[idx, :] * self.velocity_genes_mask[idx, :], dtype=torch.float32)
        v_mask = torch.tensor(self.velocity_genes_mask[idx, :], dtype=torch.float32)
        w = torch.tensor(self._w[idx] if self.use_weights else 1., dtype=torch.float32)
        return x, y, vx, v_mask, w


def load_inputs():
    adata_rna = sc.read_h5ad(f'{OUT}/adata_rna.h5ad')
    adata_atac = sc.read_h5ad(f'{OUT}/adata_motif.h5ad')
    assert all(adata_rna.obs_names == adata_atac.obs_names)
    if not isinstance(adata_rna.X, np.ndarray):
        adata_rna.X = adata_rna.X.toarray()
    adata_rna.X = adata_rna.X.astype(np.float32)
    adata_atac.X = np.asarray(adata_atac.X, dtype=np.float32)
    return adata_rna, adata_atac


def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
