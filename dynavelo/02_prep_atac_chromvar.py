"""chromVAR TF-motif deviation z-scores (pychromvar + JASPAR2024 CORE vertebrates) for DynaVelo.

Output: results/adata_motif.h5ad  (cells x motifs; X = kNN-smoothed z-scores, layers['z_raw'], var['TF'])
"""
import os
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import pychromvar as pc
from pyjaspar import jaspardb
from sklearn.decomposition import TruncatedSVD

ROOT = '/home/azureuser/vcc'
OUT = f'{ROOT}/results'
GENOME = f'{ROOT}/ref/hg38.fa'
MIN_FRAC_CELLS = 0.03
N_NEIGHBORS = 30

rna = sc.read_h5ad(f'{OUT}/adata_rna.h5ad', backed='r')
cells = rna.obs_names.copy()
rna_genes = set(rna.var_names)

adata = sc.read_h5ad(f'{ROOT}/data/cluster0_and_3_atac.h5ad')[cells].copy()
adata.X = sp.csr_matrix(adata.X)
chrom = adata.var_names.str.split(':').str[0]
std = chrom.isin([f'chr{i}' for i in range(1, 23)] + ['chrX'])
frac = np.asarray((adata.X > 0).mean(0)).ravel()
adata = adata[:, std & (frac >= MIN_FRAC_CELLS)].copy()
print('peaks kept:', adata.n_vars, 'cells:', adata.n_obs)
adata.layers['counts'] = adata.X.copy()

# ---- LSI embedding of ATAC (for kNN smoothing of the per-cell z-scores, cf. ArchR imputation used by the authors)
X = adata.X
tf = X.multiply(1 / np.asarray(X.sum(1))).tocsr()
idf = np.log1p(X.shape[0] / np.asarray((X > 0).sum(0)).ravel())
tfidf = tf.multiply(idf).tocsr()
tfidf.data = np.log1p(tfidf.data * 1e4)
lsi = TruncatedSVD(n_components=30, random_state=0).fit_transform(tfidf)
depth = np.log10(np.asarray(X.sum(1)).ravel())
print('corr(LSI1, depth) =', np.corrcoef(lsi[:, 0], depth)[0, 1])
adata.obsm['X_lsi'] = lsi[:, 1:]  # first component tracks sequencing depth

# ---- chromVAR
pc.add_peak_seq(adata, genome_file=GENOME, delimiter=':|-')
pc.add_gc_bias(adata)
pc.get_bg_peaks(adata, niterations=50, n_jobs=4)

jdb = jaspardb(release='JASPAR2024')
motifs = [m for m in jdb.fetch_motifs(collection='CORE', tax_group=['vertebrates']) if '::' not in m.name]
motifs = [m for m in motifs if m.name.upper() in rna_genes]
print('motifs of TFs present in the RNA gene set:', len(motifs))
pc.match_motif(adata, motifs=motifs)
dev = pc.compute_deviations(adata, n_jobs=4)
dev.obs = adata.obs.copy()
print(dev)

Z = np.nan_to_num(np.asarray(dev.X, dtype=np.float32))
dev.var['motif_id'] = [m.matrix_id for m in motifs]
dev.var['TF'] = [m.name.upper() for m in motifs]
dev.var['n_peaks'] = np.asarray(adata.varm['motif_match'].sum(0)).ravel()
dev.var['z_var'] = Z.var(0)

# one motif per TF: keep the most variable one
keep = dev.var.sort_values('z_var', ascending=False).drop_duplicates('TF').index
dev = dev[:, dev.var_names.isin(keep)].copy()
Z = np.nan_to_num(np.asarray(dev.X, dtype=np.float32))
dev.layers['z_raw'] = Z

# kNN smoothing on the ATAC LSI graph
dev.obsm['X_lsi'] = adata.obsm['X_lsi']
sc.pp.neighbors(dev, n_neighbors=N_NEIGHBORS, use_rep='X_lsi', key_added='lsi')
C = dev.obsp['lsi_connectivities'].tocsr().astype(np.float32)
C = C + sp.eye(C.shape[0], dtype=np.float32, format='csr')
C = sp.diags(1 / np.asarray(C.sum(1)).ravel()) @ C
dev.X = np.asarray(C @ Z, dtype=np.float32)
dev.var['z_smooth_var'] = dev.X.var(0)
dev.obs['celltype'] = rna.obs['celltype'].values
dev.var_names = dev.var['TF'].values

print(dev.var.sort_values('z_var', ascending=False).head(25)[['motif_id', 'n_peaks', 'z_var', 'z_smooth_var']])
print('X range', dev.X.min(), dev.X.max(), 'std', dev.X.std())
dev.write_h5ad(f'{OUT}/adata_motif.h5ad')
print('saved', dev)
