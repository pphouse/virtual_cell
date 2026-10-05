"""RNA preprocessing + scVelo (dynamical) for DynaVelo, following
DynaVelo/preprocessing/joint_rna_motif_analysis.ipynb as closely as possible.

Output: results/adata_rna.h5ad  (X = log1p normalised, layers['velocity'], obs['latent_time'], obs['celltype'])
"""
import os
import sys
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
import scipy.sparse as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pyjaspar import jaspardb

ROOT = '/home/azureuser/vcc'
OUT = f'{ROOT}/results'
os.makedirs(f'{OUT}/figures', exist_ok=True)
N_HVG = 2000
N_TOP_EXPR = 2000
N_JOBS = 4

adata = sc.read_h5ad(f'{ROOT}/data/cluster0_and_3_rna.h5ad')
adata.var_names_make_unique()

# spliced / unspliced counts from velocyto
with h5py.File(f'{ROOT}/data/cluster0_and_3_velocyto.loom') as f:
    cells = f['col_attrs/CellID'][:].astype(str)
    acc = f['row_attrs/Accession'][:].astype(str)
    S = sp.csr_matrix(f['layers/spliced'][:].T.astype(np.float32))
    U = sp.csr_matrix(f['layers/unspliced'][:].T.astype(np.float32))
ci = pd.Index(cells).get_indexer(adata.obs_names)
gi = pd.Index(acc).get_indexer(adata.var['gene_ids'])
assert (ci >= 0).all() and (gi >= 0).all()
adata.layers['spliced'] = S[ci][:, gi]
adata.layers['unspliced'] = U[ci][:, gi]
adata.layers['counts'] = adata.X.copy()
print('spliced/unspliced totals:', adata.layers['spliced'].sum(), adata.layers['unspliced'].sum())

adata.obs['celltype'] = adata.obs['cell_type'].astype(str)
# a cell type with a single cell breaks the 1/frequency cell weights used by DynaVelo
keep = adata.obs['celltype'].map(adata.obs['celltype'].value_counts()) >= 10
adata = adata[keep.values].copy()
adata.obs['celltype'] = adata.obs['celltype'].astype('category')

sc.pp.filter_cells(adata, min_counts=1000)
adata = adata[:, ~adata.var_names.isin(['MALAT1'])].copy()
print('after cell filtering:', adata.shape)

scv.pp.filter_and_normalize(adata, min_shared_counts=10)
sc.pp.log1p(adata)
adata.layers['X_log'] = adata.X.copy()
sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, flavor='seurat')
sc.pp.pca(adata, n_comps=30, use_highly_variable=True)
sc.pp.neighbors(adata, n_neighbors=30, use_rep='X_pca')
sc.tl.umap(adata)
adata.var['hvg'] = adata.var['highly_variable']
del adata.var['highly_variable']

# ---- gene set fed to the model: HVG + TFs with a JASPAR motif + highly expressed genes (+ optional user list)
jdb = jaspardb(release='JASPAR2024')
motifs = jdb.fetch_motifs(collection='CORE', tax_group=['vertebrates'])
tf_names = sorted({m.name.upper() for m in motifs if '::' not in m.name})
frac_expr = np.asarray((adata.layers['counts'] > 0).mean(0)).ravel()
mean_expr = np.asarray(adata.X.mean(0)).ravel()
adata.var['frac_expr'] = frac_expr
is_tf = adata.var_names.isin(tf_names) & (frac_expr > 0.02)
top_expr = np.zeros(adata.n_vars, bool)
top_expr[np.argsort(-mean_expr)[:N_TOP_EXPR]] = True
extra = np.zeros(adata.n_vars, bool)
extra_file = f'{ROOT}/data/vcc_target_genes.txt'
if os.path.exists(extra_file):
    wanted = [l.strip() for l in open(extra_file) if l.strip()]
    extra = adata.var_names.isin(wanted) | adata.var['gene_ids'].isin(wanted).values
    print(f'user gene list: {len(wanted)} requested, {extra.sum()} pass expression filters')
adata.var['is_tf'] = is_tf
adata.var['top_expr'] = top_expr
sel = adata.var['hvg'].values | is_tf | top_expr | extra
print(f'model genes: {sel.sum()} (hvg {adata.var["hvg"].sum()}, motif TFs {is_tf.sum()}, top expressed {top_expr.sum()})')
adata = adata[:, sel].copy()

# ---- scVelo dynamical model
scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
scv.tl.recover_dynamics(adata, use_raw=False, n_jobs=N_JOBS)
scv.tl.velocity(adata, mode='dynamical', filter_genes=False, use_highly_variable=False)
vg = adata.var_names[adata.var['velocity_genes'].values]
print('velocity genes:', len(vg))
scv.tl.velocity_graph(adata, gene_subset=list(vg), xkey='Ms', sqrt_transform=False, n_jobs=N_JOBS)
scv.tl.velocity_confidence(adata)
scv.tl.latent_time(adata)
print(adata.obs.groupby('celltype', observed=True)[['latent_time', 'velocity_confidence', 'velocity_length']].mean())

# DynaVelo masks genes whose velocity is NaN -> keep NaN for non-fitted genes
v = adata.layers['velocity']
print('genes with finite velocity:', int((~np.isnan(v)).all(0).sum()), '/', adata.n_vars)

try:
    scv.pl.velocity_embedding_stream(adata, basis='umap', color='celltype', show=False, legend_loc='right margin')
    plt.savefig(f'{OUT}/figures/scvelo_stream_celltype.png', dpi=130, bbox_inches='tight'); plt.close()
    scv.pl.scatter(adata, basis='umap', color='latent_time', color_map='viridis', show=False)
    plt.savefig(f'{OUT}/figures/scvelo_latent_time.png', dpi=130, bbox_inches='tight'); plt.close()
except Exception as e:
    print('plot failed:', repr(e))

adata.write_h5ad(f'{OUT}/adata_rna.h5ad')
print('saved', adata)
