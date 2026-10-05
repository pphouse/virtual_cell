"""Streaming writer for a submission-sized sparse .h5ad.

A complete 2026 submission is 300 perturbations x 400 cells x 3 contexts =
360,000 cells over 18,533 genes.  Built from real control cells it carries
roughly 6,000 stored entries per cell, so about 2.2e9 nonzeros -- comfortably
under the scorer's 4.75e9 density cap, but around 17 GB as raw CSR arrays.
That is too big to hold in memory *and* too big to write uncompressed on a
normal disk allowance, and both failure modes strike an hour into sampling.

So the matrix is never fully materialised, and it is never stored raw.  `obs`
and `var` are written once through AnnData (they are small, and this guarantees
the on-disk encoding is exactly what AnnData will read back), then `X` is
replaced by resizable, gzip-compressed CSR datasets that each block of cells is
appended to as it is generated.  Counts compress hard -- most nonzero values
are 1 or 2, and CSR indices ascend within each row -- which is the same reason
the CLI guide warns that a 25 MB .vcc can need more memory to score than a
3.3 GB one: file size says nothing about the density cap.
"""

from __future__ import annotations

import logging
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)


class SparseH5adWriter:
    """Append blocks of cells to a CSR .h5ad without holding the matrix."""

    def __init__(
        self,
        path: str | Path,
        obs: pd.DataFrame,
        var_names: np.ndarray,
        dtype: str = "float32",
        chunk: int = 1 << 20,
        compression: str | None = "gzip",
        compression_opts: int | None = 4,
    ) -> None:
        self.path = Path(path)
        self.n_obs = len(obs)
        self.n_vars = int(np.asarray(var_names).size)
        self._dtype = dtype
        self._chunk = chunk
        self._rows = 0
        self._nnz = 0

        self.path.parent.mkdir(parents=True, exist_ok=True)
        skeleton = ad.AnnData(
            X=sp.csr_matrix((self.n_obs, self.n_vars), dtype=np.float32),
            obs=obs,
            var=pd.DataFrame(index=pd.Index(np.asarray(var_names, dtype=str))),
        )
        skeleton.write_h5ad(self.path)
        del skeleton

        self._f = h5py.File(self.path, "a")
        del self._f["X"]
        g = self._f.create_group("X")
        g.attrs["encoding-type"] = "csr_matrix"
        g.attrs["encoding-version"] = "0.1.0"
        g.attrs["shape"] = np.array([self.n_obs, self.n_vars], dtype=np.int64)
        kw: dict = {"chunks": (chunk,)}
        if compression:
            kw["compression"] = compression
            if compression_opts is not None:
                kw["compression_opts"] = compression_opts
        g.create_dataset("data", shape=(0,), maxshape=(None,), dtype=dtype, **kw)
        g.create_dataset("indices", shape=(0,), maxshape=(None,), dtype="int32", **kw)
        self._indptr = np.zeros(self.n_obs + 1, dtype=np.int64)
        self._g = g

    def append(self, block: sp.csr_matrix) -> None:
        if block.shape[1] != self.n_vars:
            raise ValueError(f"block has {block.shape[1]} genes, expected {self.n_vars}")
        if self._rows + block.shape[0] > self.n_obs:
            raise ValueError("more cells appended than the writer was sized for")
        block.sort_indices()

        n = block.nnz
        for name, values in (("data", block.data), ("indices", block.indices)):
            ds = self._g[name]
            ds.resize((self._nnz + n,))
            ds[self._nnz :] = values
        self._indptr[self._rows + 1 : self._rows + block.shape[0] + 1] = (
            self._nnz + block.indptr[1:]
        )
        self._rows += block.shape[0]
        self._nnz += n

    def close(self) -> None:
        if self._rows != self.n_obs:
            raise ValueError(f"wrote {self._rows} of {self.n_obs} cells")
        self._g.create_dataset("indptr", data=self._indptr, dtype="int64")
        self._f.close()
        logger.info(
            "wrote %s: %d cells x %d genes, %d stored entries (%.0f/cell)",
            self.path,
            self.n_obs,
            self.n_vars,
            self._nnz,
            self._nnz / max(self.n_obs, 1),
        )

    @property
    def nnz(self) -> int:
        return self._nnz

    def __enter__(self) -> SparseH5adWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            self._f.close()
