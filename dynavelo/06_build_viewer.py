"""Collect the DynaVelo results into one JSON blob and inline it into the viewer template.

    python 06_build_viewer.py <template.html> <out.html>
"""
import json
import sys
import glob
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc

ROOT = '/home/azureuser/vcc'
OUT = f'{ROOT}/results'
template, out_html = sys.argv[1], sys.argv[2]


def r(a, n=3):
    return np.round(np.asarray(a, dtype=float), n).tolist()


# ---- cells
rna = sc.read_h5ad(f'{OUT}/adata_rna_pred.h5ad')
v = rna.layers['velocity']; m = ~np.isnan(v).any(0); vp = rna.layers['vx_pred_mean']
cos = (v[:, m] * vp[:, m]).sum(1) / (np.linalg.norm(v[:, m], axis=1) * np.linalg.norm(vp[:, m], axis=1) + 1e-12)
cts = list(rna.obs['celltype'].cat.categories)
o = rna.obs
cells = dict(
    x=r(rna.obsm['X_umap'][:, 0], 2), y=r(rna.obsm['X_umap'][:, 1], 2),
    ct=o['celltype'].cat.codes.astype(int).tolist(),
    lt=r(o['latent_time_mean']), lts=r(o['latent_time_scvelo']), ltsd=r(o['latent_time_std']),
    cos=r(cos), nA=o['n_contextA_cells'].astype(int).tolist(),
)
o2 = o.assign(cos=cos)
g = o2.groupby('celltype', observed=True)
ct_table = [dict(celltype=c, n=int(g.size()[c]), lt=round(float(g['latent_time_mean'].mean()[c]), 3),
                 lts=round(float(g['latent_time_scvelo'].mean()[c]), 3), cos=round(float(g['cos'].mean()[c]), 3),
                 nA=int(g['n_contextA_cells'].sum()[c]), nAcells=int((o2['n_contextA_cells'][o2['celltype'] == c] > 0).sum()))
            for c in cts]

# ---- training log
log = pd.read_csv(glob.glob(f'{ROOT}/DynaVelo/log/PBMC_cluster0_3_*.log')[0], sep='\t')
train = dict(epoch=log['Epoch'].astype(int).tolist(), loss_train=r(log['loss_train'], 0), loss_test=r(log['loss_test'], 0),
             vel_cos=r(-log['loss_vel'], 4), nll_x=r(log['nll_x'], 0), nll_y=r(log['nll_y'], 0))

# ---- perturbations (context A weighted)
p = ad.read_h5ad(f'{OUT}/contextA_dynavelo_perturbation.h5ad')
D = np.array(p.X, dtype=np.float64)
self_dx = np.diag(D).copy()
np.fill_diagonal(D, 0)
genes = list(p.var_names)
norm = np.linalg.norm(D, axis=1)
k = np.argsort(-norm)[:1000]
U, S, Vt = np.linalg.svd(D[k], full_matrices=False)
if Vt[0][genes.index('LEF1')] > 0:  # orient PC1 so that + = naive programme down
    Vt[0] *= -1
scores = D @ Vt[:2].T
var_exp = (S ** 2 / (S ** 2).sum())[:8]
oload = np.argsort(Vt[0])
pc1 = dict(neg=[[genes[j], round(float(Vt[0][j]), 3)] for j in oload[:15]],
           pos=[[genes[j], round(float(Vt[0][j]), 3)] for j in oload[::-1][:15]])
top = []
for i in range(len(genes)):
    oo = np.argsort(D[i])
    top.append([[int(j) for j in oo[:8]], [round(float(D[i, j]) * 1e3, 2) for j in oo[:8]],
                [int(j) for j in oo[::-1][:8]], [round(float(D[i, j]) * 1e3, 2) for j in oo[::-1][:8]]])
cosine_pc1 = (D @ Vt[0]) / (norm + 1e-12)
pert = dict(
    gene=genes, tf=p.obs['is_motif_tf'].astype(int).tolist(),
    frac=r(p.obs['frac_expressing_contextA_matched']), expr=r(p.obs['mean_logexpr_contextA_matched']),
    norm=r(norm * 1e3, 2), self=r(self_dx * 1e3, 2), dt=r(p.obs['delta_latent_time'] * 1e3, 2),
    pc1=r(scores[:, 0] * 1e3, 2), pc2=r(scores[:, 1] * 1e3, 2), cpc1=r(cosine_pc1, 2), top=top,
)
Pn = D[k] / np.linalg.norm(D[k], axis=1, keepdims=True)
C = np.abs(Pn @ Pn.T)
stats = dict(
    n_cells=int(rna.n_obs), n_genes=int(rna.n_vars), n_tfs=int(p.obsm['delta_motif_velocity'].shape[1]),
    n_vel_genes=int(m.sum()), epochs=int(log['Epoch'].max()), vel_cos=round(float(cos.mean()), 3),
    lt_spearman=round(float(pd.Series(o['latent_time_mean']).corr(o['latent_time_scvelo'], method='spearman')), 3),
    nA_total=18400, nA_kept=int(o['n_contextA_cells'].sum()), nA_pbmc=int((o['n_contextA_cells'] > 0).sum()),
    max_abs_dx=round(float(np.abs(np.array(p.X)).max()), 4), var_exp=r(var_exp, 3),
    mean_abs_cos=round(float(C[np.triu_indices(len(k), 1)].mean()), 3),
    corr_norm_expr=round(float(np.corrcoef(norm, p.obs['mean_logexpr_contextA_matched'])[0, 1]), 3),
    self_neg_frac=round(float((self_dx < 0).mean()), 3), self_mean=round(float(self_dx.mean()), 5),
)

# ---- TF knock-down by cell type
t = pd.read_csv(f'{OUT}/tf_perturbation_by_celltype.csv')
pv = t.pivot(index='perturbed_tf', columns='celltype', values='delta_latent_time')[cts]
tf_top = pv.abs().max(1).sort_values(ascending=False).index[:30]
tfko = dict(tfs=list(tf_top), cts=cts, dlt=[r(pv.loc[x].values * 1e3, 2) for x in tf_top])

# ---- motif velocity
mot = sc.read_h5ad(f'{OUT}/adata_motif_pred.h5ad')
vy = pd.DataFrame(mot.layers['vy_pred_mean'], columns=mot.var_names)
vy['ct'] = o['celltype'].values
mv = vy.groupby('ct', observed=True).mean().loc[cts]
mtop = mv.abs().max(0).sort_values(ascending=False).index[:25]
motif = dict(tfs=list(mtop), cts=cts, v=[r(mv[x].values, 3) for x in mtop])

data = dict(cts=cts, cells=cells, ct_table=ct_table, train=train, pert=pert, pc1=pc1, stats=stats, tfko=tfko, motif=motif)
blob = json.dumps(data, separators=(',', ':'), ensure_ascii=False)
html = open(template, encoding='utf-8').read().replace('/*__DATA__*/null', blob)
open(out_html, 'w', encoding='utf-8').write(html)
json.dump(stats, open(f'{OUT}/viewer_stats.json', 'w'), indent=1)
print('wrote', out_html, round(len(html) / 1e6, 2), 'MB'); print(json.dumps(stats)); print(pd.DataFrame(ct_table)); print(pc1)
