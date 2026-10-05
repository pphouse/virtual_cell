"""Train DynaVelo on the PBMC cluster 0/3 multiome data and predict velocities / latent time.
Mirrors DynaVelo/examples/DynaVelo_Demo.ipynb (default hyper-parameters)."""
import sys
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dv_common import (OUT, DATASET, SAMPLE, MultiomeDataset, DynaVelo, load_inputs, set_seed, get_device)

MAX_EPOCH = int(sys.argv[1]) if len(sys.argv) > 1 else 200
N_SAMPLES = 20

set_seed(0)
device = get_device()
adata_rna, adata_atac = load_inputs()
print(adata_rna.shape, adata_atac.shape, device, flush=True)

dataset = MultiomeDataset(adata_rna, adata_atac)
N_test = int(0.1 * len(dataset))
idx_random = np.random.permutation(len(dataset))
idx_test, idx_train = idx_random[:N_test], idx_random[N_test:]
dataset_train = MultiomeDataset(adata_rna[idx_train], adata_atac[idx_train])
dataset_test = MultiomeDataset(adata_rna[idx_test], adata_atac[idx_test])

dataloader_train = DataLoader(dataset_train, batch_size=128, shuffle=True, num_workers=0, drop_last=True)
dataloader_test = DataLoader(dataset_test, batch_size=128, shuffle=False, num_workers=0)
dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)

model = DynaVelo(x_dim=adata_rna.shape[1], y_dim=adata_atac.shape[1], device=device,
                 dataset_name=DATASET, sample_name=SAMPLE).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

t0 = time.time()
model.fit(dataloader_train, dataloader_test, optimizer, max_epoch=MAX_EPOCH)
print(f'training took {(time.time() - t0) / 60:.1f} min', flush=True)

model.load(optimizer)  # best checkpoint (early stopping)
model.mode = 'evaluation-sample'
adata_rna_pred, adata_atac_pred = model.evaluate(adata_rna, adata_atac, dataloader, n_samples=N_SAMPLES)
adata_rna_pred.write_h5ad(f'{OUT}/adata_rna_pred.h5ad')
adata_atac_pred.write_h5ad(f'{OUT}/adata_motif_pred.h5ad')
print('saved predictions', flush=True)
