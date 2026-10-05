"""QC of the DynaVelo predictions: reconstruction, velocity agreement, latent time, plots."""
import numpy as np, pandas as pd, scanpy as sc, scvelo as scv
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
OUT = '/home/azureuser/vcc/results'
r = sc.read_h5ad(f'{OUT}/adata_rna_pred.h5ad'); a = sc.read_h5ad(f'{OUT}/adata_motif_pred.h5ad')
X = np.asarray(r.X); xp = r.layers['x_pred_mean']
print('RNA recon: pearson(all entries)=%.3f  per-gene-mean pearson=%.3f' % (pearsonr(X.ravel(), xp.ravel())[0], pearsonr(X.mean(0), xp.mean(0))[0]))
Y = np.asarray(a.X); yp = a.layers['y_pred_mean']
print('motif recon: pearson(all entries)=%.3f' % pearsonr(Y.ravel(), yp.ravel())[0])
pg = np.array([pearsonr(Y[:, j], yp[:, j])[0] for j in range(Y.shape[1])]); print('motif per-TF pearson median=%.3f' % np.median(pg))
v = r.layers['velocity']; m = ~np.isnan(v).any(0); vp = r.layers['vx_pred_mean']
cos = (v[:, m] * vp[:, m]).sum(1) / (np.linalg.norm(v[:, m], axis=1) * np.linalg.norm(vp[:, m], axis=1) + 1e-12)
print('velocity genes used for supervision:', m.sum(), ' cosine(scVelo, DynaVelo) mean=%.3f median=%.3f' % (cos.mean(), np.median(cos)))
r.obs['vel_cosine'] = cos
print('latent time: spearman(DynaVelo, scVelo)=%.3f' % spearmanr(r.obs['latent_time_mean'], r.obs['latent_time_scvelo'])[0])
print(r.obs.groupby('celltype', observed=True)[['latent_time_mean', 'latent_time_std', 'latent_time_scvelo', 'vel_cosine']].mean().round(3))
print('latent_time_mean range', r.obs['latent_time_mean'].min(), r.obs['latent_time_mean'].max())
print('|vx_pred| mean', np.abs(vp).mean(), ' uncertainty (std/|mean|) median', np.median(r.layers['vx_pred_std'].mean(0) / (np.abs(vp).mean(0) + 1e-9)))
vy = a.layers['vy_pred_mean']
top = pd.Series(np.abs(vy).mean(0), index=a.var_names).sort_values(ascending=False)
print('TFs with largest |motif velocity|:', ', '.join(top.index[:15]))
# plots
r.layers['velocity_scvelo'] = r.layers['velocity'].copy()
r.layers['velocity_dv'] = vp
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
sc.pl.umap(r, color='celltype', ax=ax[0], show=False, frameon=False)
sc.pl.umap(r, color='latent_time_mean', ax=ax[1], show=False, frameon=False, cmap='viridis', title='DynaVelo latent time')
sc.pl.umap(r, color='latent_time_scvelo', ax=ax[2], show=False, frameon=False, cmap='viridis', title='scVelo latent time')
plt.savefig(f'{OUT}/figures/dynavelo_latent_time.png', dpi=120, bbox_inches='tight'); plt.close()
try:
    scv.tl.velocity_graph(r, vkey='velocity_dv', xkey='Ms', sqrt_transform=False, n_jobs=4)
    scv.pl.velocity_embedding_stream(r, vkey='velocity_dv', basis='umap', color='celltype', show=False, legend_loc='right margin', title='DynaVelo RNA velocity')
    plt.savefig(f'{OUT}/figures/dynavelo_stream.png', dpi=120, bbox_inches='tight'); plt.close()
except Exception as e: print('stream plot failed', repr(e))
