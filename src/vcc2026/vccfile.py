"""Packaging a `.vcc` without loading the prediction into memory.

`vcc prep` is the reference implementation and the authority on the format.  It
is also unusable on a machine this size: it reads the whole prediction into one
SciPy matrix, and the CLI's own `sizing.py` records the measurement -- a
2.08e9-nonzero prediction is 15.84 GiB resident, before the second copy the read
path holds.  Our prediction is 2.06e9 nonzeros.  On a 15 GB box prep is
OOM-killed, and the temporary *uncompressed* h5ad it writes on the way (another
~16 GB) does not fit on the disk allowance either.

Reducing the prediction's density to fit would mean emitting shallower cells
than the real controls, which costs real DE power for a purely local reason.
So the file is packaged directly instead.

The format is small and fully specified by prep itself: an **uncompressed tar
whose single member is `pred.h5ad.zst`**, the member metadata normalised
(uid/gid 0, empty names, mtime 0) so identical predictions produce identical
archives.  Inside is the "minimal" AnnData -- CSR float32 `X`, an `obs` holding
only `target_gene` and `context` with a positional string index, and a `var`
carrying nothing but the gene names.

`build_submission` already writes exactly that minimal form, so packaging is a
streaming zstd of a file we have plus a tar header.  Parity with the real prep
is not assumed: `tests/test_vcc_parity.py` builds a miniature submission, runs
both this and `vcc prep`, and asserts the two archives decode to the same
AnnData.
"""

from __future__ import annotations

import logging
import tarfile
from pathlib import Path

logger = logging.getLogger(__name__)

PRED_MEMBER = "pred.h5ad.zst"


def write_vcc(h5ad_path: str | Path, output: str | Path, level: int = 3) -> Path:
    """Package a prep-minimal `.h5ad` into a `.vcc`, streaming throughout."""
    import zstandard as zstd

    h5ad_path, output = Path(h5ad_path), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    zst_path = output.with_suffix(output.suffix + ".tmp.zst")

    cctx = zstd.ZstdCompressor(level=level, threads=-1)
    with open(h5ad_path, "rb") as src, open(zst_path, "wb") as dst:
        cctx.copy_stream(src, dst)
    logger.info(
        "zstd: %.2f GB -> %.2f GB",
        h5ad_path.stat().st_size / 1e9,
        zst_path.stat().st_size / 1e9,
    )

    def _normalize(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo:
        # Same normalisation prep applies: without it the archive records the
        # submitter's uid, username and local mtime.
        tarinfo.uid = tarinfo.gid = 0
        tarinfo.uname = tarinfo.gname = ""
        tarinfo.mtime = 0
        return tarinfo

    with tarfile.open(output, "w") as tar:
        tar.add(zst_path, arcname=PRED_MEMBER, filter=_normalize)
    zst_path.unlink()
    logger.info("wrote %s (%.2f GB)", output, output.stat().st_size / 1e9)
    return output


def read_vcc(path: str | Path, out_dir: str | Path) -> Path:
    """Unpack a `.vcc` back to an `.h5ad` (used by the parity test)."""
    import zstandard as zstd

    path, out_dir = Path(path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "r") as tar:
        member = tar.getmember(PRED_MEMBER)
        extracted = out_dir / PRED_MEMBER
        with tar.extractfile(member) as src, open(extracted, "wb") as dst:
            while chunk := src.read(1 << 20):
                dst.write(chunk)
    h5ad = out_dir / "pred.h5ad"
    dctx = zstd.ZstdDecompressor()
    with open(extracted, "rb") as src, open(h5ad, "wb") as dst:
        dctx.copy_stream(src, dst)
    extracted.unlink()
    return h5ad


def rewrite_obs(path: str | Path, obs) -> None:
    """Replace an .h5ad's `obs` group in place, without touching `X`.

    Used to bring an already-built prediction onto prep's exact minimal obs
    encoding.  Rebuilding the file would take a quarter of an hour of sampling
    to change a few hundred kilobytes; `obs` is independent of `X` on disk, so
    it is swapped directly.  The replacement group is produced by AnnData
    itself, so the encoding is whatever AnnData writes rather than a hand-rolled
    guess at the spec.
    """
    import tempfile

    import anndata as ad
    import h5py
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    path = Path(path)
    with tempfile.TemporaryDirectory() as tmp:
        donor = Path(tmp) / "obs.h5ad"
        ad.AnnData(
            X=sp.csr_matrix((len(obs), 1), dtype=np.float32),
            obs=obs,
            var=pd.DataFrame(index=pd.Index(["_"])),
        ).write_h5ad(donor)
        with h5py.File(path, "a") as dst, h5py.File(donor, "r") as src:
            n = int(dst["X"].attrs["shape"][0])
            if len(obs) != n:
                raise ValueError(f"obs has {len(obs)} rows, X has {n}")
            del dst["obs"]
            src.copy(src["obs"], dst, name="obs")
    logger.info("rewrote obs in %s (%d rows, columns %s)", path, len(obs), list(obs.columns))
