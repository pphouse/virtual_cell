"""Check whether the DynaVelo gene features say anything about which knock-downs look alike,
then write co-essentiality + DynaVelo blends for the VCC transfer model.

Run with the virtual_cell venv:
    virtual_cell/.venv/bin/python scripts/08_blend_and_validate.py <weight> [<weight> ...]

Validation: over the Replogle K562 genome-wide signatures of the challenge targets, compare the
feature cosine of every target pair with the correlation of their measured signatures.
"""
import sys
import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr
from vcc2026.coessentiality import blend, load_features
from vcc2026.features import GeneFeatures
from vcc2026.library import SignatureLibrary
from vcc2026.network import string_features

SP = '/home/azureuser/vcc/vcc_sp'
DV = '/home/azureuser/vcc/results/dynavelo_gene_features.npz'

lib = SignatureLibrary.load(f'{SP}/lib_gwps300.npz')
sig = lib.deltas['K562gwps']
targets = sorted(sig)
D = np.stack([np.asarray(sig[t], dtype=np.float64) for t in targets])
D = D - D.mean(0, keepdims=True)          # remove the response every knock-down shares
D /= np.linalg.norm(D, axis=1, keepdims=True) + 1e-12
truth = D @ D.T
iu = np.triu_indices(len(targets), 1)


def unit(m):
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)


def report(name, feats):
    pos = {str(g): i for i, g in enumerate(feats.genes)}
    have = np.array([t in pos and np.linalg.norm(feats.matrix[pos[t]]) > 0 for t in targets])
    idx = np.flatnonzero(have)
    F = unit(np.stack([feats.matrix[pos[targets[i]]] for i in idx]).astype(np.float64))
    S = F @ F.T
    T = truth[np.ix_(idx, idx)]
    k = np.triu_indices(len(idx), 1)
    rho = spearmanr(S[k], T[k])[0]
    np.fill_diagonal(S, -np.inf)
    top = np.argsort(-S, axis=1)[:, :5]
    nn = np.mean([T[i, top[i]].mean() for i in range(len(idx))])
    Tn = T.copy(); np.fill_diagonal(Tn, np.nan)
    print(f'{name:28s} targets {len(idx):3d}  spearman {rho:+.4f}  true sim of 5 nearest {nn:+.4f} '
          f'(random {np.nanmean(Tn):+.4f}, oracle {np.mean(np.sort(np.nan_to_num(Tn, nan=-1), axis=1)[:, -5:]):+.4f})')
    return set(targets[i] for i in idx)


coess = load_features(f'{SP}/depmap/coess.npz')
z = np.load(DV)
dv = GeneFeatures(genes=np.asarray(z['genes'].astype(str), dtype=object), matrix=z['matrix'])
string = string_features(sp.load_npz(f'{SP}/string/adj_400.npz'), lib.genes, subset=targets)

report('STRING', string)
report('co-essentiality', coess)
c_dv = report('DynaVelo', dv)
# same target subset for a fair read of the blends
sub = lambda f: GeneFeatures(genes=np.asarray([g for g in f.genes if str(g) in c_dv], dtype=object),  # noqa: E731
                             matrix=np.stack([f.matrix[i] for i, g in enumerate(f.genes) if str(g) in c_dv]))
report('STRING (DynaVelo subset)', sub(string))
report('co-ess (DynaVelo subset)', sub(coess))
report('STRING+coess', blend(sub(string), sub(coess), 1.0))
for w in map(float, sys.argv[1:]):
    b = blend(coess, dv, w)
    report(f'STRING+[coess|{w}*DynaVelo]', blend(sub(string), sub(b), 1.0))
    out = f'{SP}/depmap/coess_dv{w:g}.npz'
    np.savez(out, genes=np.asarray(b.genes, dtype=str), matrix=b.matrix.astype(np.float32))
    print('  wrote', out, b.matrix.shape)
