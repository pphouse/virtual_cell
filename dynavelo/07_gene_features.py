"""Gene features for the VCC transfer model from the trained DynaVelo model.

For every requested gene g, nudge its input expression by -EPS on the PBMC cells matched to
context A and record the change in decoded expression (a finite-difference Jacobian column,
averaged over the context-A-weighted cells).  Unlike the knock-down-to-minimum of
04_perturb_contextA.py this does not scale with how highly g is expressed, so lowly expressed
targets get a direction too.

The shared response (every gene moves the naive<->effector axis) is removed by centring across
genes and dropping the leading component before the rows are L2-normalised.

Output: results/dynavelo_gene_features.npz with `genes` and `matrix` (the format
vcc2026.coessentiality.load_features reads).
"""
import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.optim as optim
from dv_common import ROOT, OUT, DATASET, SAMPLE, DynaVelo, set_seed, get_device

EPS = 1.0
BATCH = 128
N_COMP = int(sys.argv[1]) if len(sys.argv) > 1 else 50
DROP = int(sys.argv[2]) if len(sys.argv) > 2 else 1

set_seed(0)
device = get_device()
adata_rna = sc.read_h5ad(f'{OUT}/adata_rna_pred.h5ad')
adata_atac = sc.read_h5ad(f'{OUT}/adata_motif_pred.h5ad')
X = np.asarray(adata_rna.X, dtype=np.float32)
Y = np.asarray(adata_atac.X, dtype=np.float32)
genes = adata_rna.var_names.values

model = DynaVelo(x_dim=X.shape[1], y_dim=Y.shape[1], device=device, dataset_name=DATASET, sample_name=SAMPLE).to(device)
model.load(optim.Adam(model.parameters(), lr=1e-3))
model.mode = 'evaluation-fixed'
model.eval()


@torch.no_grad()
def decode(x, y):
    out = []
    for c in np.array_split(np.arange(x.shape[0]), max(1, int(np.ceil(x.shape[0] / BATCH)))):
        xp = model(torch.from_numpy(x[c]).to(device), torch.from_numpy(y[c]).to(device))[0]
        out.append(xp.cpu().numpy())
    return np.concatenate(out)


m = pd.read_csv(f'{ROOT}/data/context_A_to_cluster0_3_umap2d_nearest_barcodes.csv')
cnt = m['nearest_pbmc_cell'].value_counts()
cnt = cnt[cnt.index.isin(adata_rna.obs_names)]
idx = adata_rna.obs_names.get_indexer(cnt.index)
order = np.argsort(idx); idx = idx[order]
w = cnt.values[order].astype(np.float64); w /= w.sum()
xA, yA = X[idx], Y[idx]
base = decode(xA, yA)

raw_path = f'{OUT}/dynavelo_unit_response.npz'
wanted_file = f'{ROOT}/data/vcc_target_genes.txt'
wanted = [l.strip() for l in open(wanted_file) if l.strip()] if os.path.exists(wanted_file) else list(genes)
todo = [g for g in wanted if g in set(genes)]
print(f'{len(todo)} of {len(wanted)} requested genes are in the model', flush=True)
R = np.zeros([len(todo), len(genes)], np.float32)
for i, g in enumerate(todo):
    x = xA.copy()
    gi = np.where(genes == g)[0].item()
    x[:, gi] -= EPS
    R[i] = w @ ((decode(x, yA) - base) / -EPS)   # d decoded expression / d x_g
    if i % 50 == 0:
        print(i, g, float(np.linalg.norm(R[i])), flush=True)
np.savez_compressed(raw_path, genes=np.array(todo), response=R, var=genes)

Rc = R - R.mean(0, keepdims=True)
U, S, Vt = np.linalg.svd(Rc, full_matrices=False)
print('variance explained (centred):', np.round((S ** 2 / (S ** 2).sum())[:8], 3))
keep = slice(DROP, DROP + N_COMP)
F = U[:, keep] * S[keep]
F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-12
np.savez(f'{OUT}/dynavelo_gene_features.npz', genes=np.array(todo), matrix=F.astype(np.float32))
print('saved', F.shape)
