"""In-silico knock-down with the trained DynaVelo model, summarised for VCC context A.

Same perturbation rule as DynaVelo.predict_perturbation (expression of the gene, and the motif z-score
if the gene is a TF with a motif, are set to their minimum over all cells; mode 'evaluation-fixed'),
but aggregated on the fly so that every model gene can be perturbed without a
[cells x genes x perturbations] tensor.

Part A: every model gene, on the PBMC cells that are nearest neighbours of context-A cells
        (weighted by how many context-A cells map to each PBMC cell).
Part B: TFs with motifs, on all cells -> per-cell-type summaries.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dv_common import (ROOT, OUT, DATASET, SAMPLE, MultiomeDataset, DynaVelo, set_seed, get_device)

BATCH = 128
set_seed(0)
device = get_device()

adata_rna = sc.read_h5ad(f'{OUT}/adata_rna_pred.h5ad')
adata_atac = sc.read_h5ad(f'{OUT}/adata_motif_pred.h5ad')
X = np.asarray(adata_rna.X, dtype=np.float32)
Y = np.asarray(adata_atac.X, dtype=np.float32)
genes = adata_rna.var_names.values
tfs = adata_atac.var['TF'].values
x_min, y_min = X.min(0), Y.min(0)

model = DynaVelo(x_dim=X.shape[1], y_dim=Y.shape[1], device=device, dataset_name=DATASET, sample_name=SAMPLE).to(device)
model.load(optim.Adam(model.parameters(), lr=1e-3))
model.mode = 'evaluation-fixed'
model.eval()


@torch.no_grad()
def forward(x, y):
    """x, y: numpy [n, .] -> dict of numpy outputs (x_pred, vx, vy, t)."""
    n = x.shape[0]
    chunks = np.array_split(np.arange(n), max(1, int(np.ceil(n / BATCH))))
    out = {k: [] for k in ['x_pred', 'y_pred', 'vx', 'vy', 't']}
    for c in chunks:
        xp, yp, vx, vy, _, _, _, t, _ = model(torch.from_numpy(x[c]).to(device), torch.from_numpy(y[c]).to(device))
        for k, v in zip(out, [xp, yp, vx, vy, t]):
            out[k].append(v.cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def perturb(x, y, gene):
    x = x.copy(); y = y.copy()
    gi = np.where(genes == gene)[0].item()
    x[:, gi] = x_min[gi]
    if gene in tfs:
        ti = np.where(tfs == gene)[0].item()
        y[:, ti] = y_min[ti]
    return forward(x, y)


# ---------------------------------------------------------------- context A <-> PBMC mapping
m = pd.read_csv(f'{ROOT}/data/context_A_to_cluster0_3_umap2d_nearest_barcodes.csv')
cnt = m['nearest_pbmc_cell'].value_counts()
present = cnt.index.isin(adata_rna.obs_names)
print(f'context A cells: {len(m)}; unique PBMC neighbours: {len(cnt)}; '
      f'kept after QC: {present.sum()} covering {cnt[present].sum()} context-A cells', flush=True)
cnt = cnt[present]
idxA = adata_rna.obs_names.get_indexer(cnt.index)
order = np.argsort(idxA); idxA = idxA[order]
wA = cnt.values[order].astype(np.float64); wA /= wA.sum()
adata_rna.obs['n_contextA_cells'] = 0
adata_rna.obs.iloc[idxA, adata_rna.obs.columns.get_loc('n_contextA_cells')] = cnt.values[order]

# ---------------------------------------------------------------- Part A: all genes on context-A matched cells
xA, yA = X[idxA], Y[idxA]
base = forward(xA, yA)
frac_expr_A = (xA > 0).T @ wA
pert_genes = list(genes)
n_p = len(pert_genes)
d_vx = np.zeros([n_p, len(genes)], np.float32)   # change in RNA velocity
d_x = np.zeros([n_p, len(genes)], np.float32)    # change in decoded expression (log1p space)
d_vy = np.zeros([n_p, len(tfs)], np.float32)     # change in motif velocity
d_y = np.zeros([n_p, len(tfs)], np.float32)      # change in decoded motif accessibility
d_t = np.zeros(n_p, np.float32)                  # change in latent time
t0 = time.time()
for i, g in enumerate(pert_genes):
    p = perturb(xA, yA, g)
    d_vx[i] = wA @ (p['vx'] - base['vx'])
    d_x[i] = wA @ (p['x_pred'] - base['x_pred'])
    d_vy[i] = wA @ (p['vy'] - base['vy'])
    d_y[i] = wA @ (p['y_pred'] - base['y_pred'])
    d_t[i] = wA @ (p['t'] - base['t'])
    if i % 200 == 0:
        print(f'[A] {i}/{n_p} {g} ({time.time() - t0:.0f}s)', flush=True)

obs = pd.DataFrame(index=pd.Index(pert_genes, name='perturbed_gene'))
obs['gene_id'] = adata_rna.var['gene_ids'].values
obs['is_motif_tf'] = np.isin(pert_genes, tfs)
obs['frac_expressing_contextA_matched'] = frac_expr_A
obs['mean_logexpr_contextA_matched'] = xA.T @ wA
obs['delta_latent_time'] = d_t
obs['delta_vx_norm'] = np.linalg.norm(d_vx, axis=1)
obs['delta_x_norm'] = np.linalg.norm(d_x, axis=1)
res = ad.AnnData(X=d_x, obs=obs, var=adata_rna.var[['gene_ids']].copy())
res.layers['delta_velocity'] = d_vx
res.obsm['delta_motif_velocity'] = pd.DataFrame(d_vy, index=obs.index, columns=tfs)
res.obsm['delta_motif_access'] = pd.DataFrame(d_y, index=obs.index, columns=tfs)
res.var['baseline_x_pred'] = wA @ base['x_pred']
res.var['baseline_x_obs'] = xA.T @ wA
res.var['baseline_velocity'] = wA @ base['vx']
res.uns['description'] = ('DynaVelo in-silico knock-down, weighted mean over PBMC cells matched to VCC context A. '
                          'X = change in decoded log1p expression; layers[delta_velocity] = change in RNA velocity.')
res.write_h5ad(f'{OUT}/contextA_dynavelo_perturbation.h5ad')

# top responders per perturbation (excluding the perturbed gene itself)
rows = []
for i, g in enumerate(pert_genes):
    d = d_x[i].copy(); d[i] = 0
    o = np.argsort(d)
    rows.append(dict(perturbed_gene=g, is_motif_tf=obs['is_motif_tf'].iloc[i], frac_expressing=frac_expr_A[i],
                     self_delta_x=d_x[i, i], delta_x_norm_others=np.linalg.norm(d), delta_latent_time=d_t[i],
                     top_down=';'.join(f'{genes[j]}({d[j]:+.3f})' for j in o[:8]),
                     top_up=';'.join(f'{genes[j]}({d[j]:+.3f})' for j in o[::-1][:8])))
pd.DataFrame(rows).sort_values('delta_x_norm_others', ascending=False).to_csv(
    f'{OUT}/contextA_perturbation_summary.csv', index=False)
print(f'[A] done in {(time.time() - t0) / 60:.1f} min', flush=True)

# ---------------------------------------------------------------- Part B: motif TFs on all cells, by cell type
if '--skip-b' not in sys.argv:
    base_all = forward(X, Y)
    ct = adata_rna.obs['celltype'].astype(str).values
    cts = sorted(set(ct))
    tf_list = [g for g in tfs if g in set(genes)]
    recs = []
    d_t_cell = np.zeros([X.shape[0], len(tf_list)], np.float32)
    t0 = time.time()
    for i, g in enumerate(tf_list):
        p = perturb(X, Y, g)
        d_t_cell[:, i] = p['t'] - base_all['t']
        dv = p['vx'] - base_all['vx']
        cos = (p['vx'] * base_all['vx']).sum(1) / (np.linalg.norm(p['vx'], axis=1) * np.linalg.norm(base_all['vx'], axis=1) + 1e-12)
        for c in cts:
            k = ct == c
            recs.append(dict(perturbed_tf=g, celltype=c, n_cells=int(k.sum()), delta_latent_time=d_t_cell[k, i].mean(),
                             delta_vx_norm=np.linalg.norm(dv[k], axis=1).mean(), velocity_cosine_to_baseline=cos[k].mean()))
        if i % 20 == 0:
            print(f'[B] {i}/{len(tf_list)} {g} ({time.time() - t0:.0f}s)', flush=True)
    pd.DataFrame(recs).to_csv(f'{OUT}/tf_perturbation_by_celltype.csv', index=False)
    adata_rna.obsm['delta_latent_time_tf_ko'] = pd.DataFrame(d_t_cell, index=adata_rna.obs_names, columns=tf_list)
    adata_rna.write_h5ad(f'{OUT}/adata_rna_pred.h5ad')
    print(f'[B] done in {(time.time() - t0) / 60:.1f} min', flush=True)
