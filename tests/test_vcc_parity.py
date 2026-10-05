"""Prove the streaming packager produces what `vcc prep` produces.

The real prediction is too dense for `vcc prep` to open on this machine, so it
is packaged directly (see `vcc2026/vccfile.py`).  That is only safe if the
result is the same file prep would have written, so the claim is tested rather
than asserted: build a miniature submission, run both paths, and compare the
AnnData that comes back out of each archive.

Skipped when the official CLI is not importable -- there is nothing to compare
against, and a green run would be meaningless.
"""

import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pytest

from vcc2026.vccfile import PRED_MEMBER, read_vcc, write_vcc

pytest.importorskip("zstandard")


def _cli() -> str | None:
    """The official CLI, which lives in its own venv to avoid a name collision."""
    for candidate in (
        Path(sys.prefix).parent / ".venv-cli" / "bin" / "vcc",
        Path.cwd() / ".venv-cli" / "bin" / "vcc",
    ):
        if candidate.exists():
            return str(candidate)
    return None


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """A miniature but structurally complete submission, plus its bundle."""
    import tests.test_submission_format as fixtures
    from tests.test_submission_format import CELLS_PER_PERT  # noqa: F401
    from vcc2026.coexpression import ControlOnlyConfig, ControlOnlyPredictor
    from vcc2026.submission import SubmissionConfig, build_submission

    root = tmp_path_factory.mktemp("bundle")
    fixtures_bundle = _make_bundle(root, fixtures)
    out = tmp_path_factory.mktemp("out")
    h5ad = out / "submission.h5ad"
    build_submission(
        fixtures_bundle,
        lambda c, x, g: ControlOnlyPredictor(ControlOnlyConfig(n_components=6)).fit(x, g),
        h5ad,
        SubmissionConfig(seed=3),
    )
    return fixtures_bundle, h5ad, out


def _make_bundle(root, fixtures):
    import json

    import pandas as pd
    import scipy.sparse as sp

    from vcc2026.challenge import CONTEXT_COL, PERT_COL, Bundle

    rng = np.random.default_rng(11)
    genes = np.array([f"G{i:04d}" for i in range(fixtures.N_GENES)], dtype=object)
    perts = genes[rng.choice(fixtures.N_GENES, size=fixtures.N_PERTS, replace=False)]
    pd.Series(genes.astype(str)).to_frame("gene_name").to_csv(root / "gene_names.csv", index=False)
    pd.Series(perts.astype(str)).to_frame("target_gene").to_csv(
        root / "pert_counts.csv", index=False
    )
    for ctx in fixtures.CONTEXTS:
        rate = rng.lognormal(0.0, 0.6, size=(fixtures.N_CONTROL, fixtures.N_GENES))
        rate = 3000 * rate / rate.sum(axis=1, keepdims=True)
        obs = pd.DataFrame(
            {
                PERT_COL: pd.Categorical(["non-targeting"] * fixtures.N_CONTROL),
                CONTEXT_COL: pd.Categorical([ctx] * fixtures.N_CONTROL),
            },
            index=np.arange(fixtures.N_CONTROL).astype(str),
        )
        ad.AnnData(
            X=sp.csr_matrix(rng.poisson(rate).astype(np.float32)),
            obs=obs,
            var=pd.DataFrame(index=pd.Index(genes.astype(str))),
        ).write_h5ad(root / f"context_{ctx}.h5ad")
    (root / "manifest.json").write_text(
        json.dumps({"contexts": fixtures.CONTEXTS, "cells_per_pert": fixtures.CELLS_PER_PERT})
    )
    return Bundle.load(root)


def test_streaming_package_round_trips(built, tmp_path):
    bundle, h5ad, out = built
    vcc = write_vcc(h5ad, out / "mine.vcc")
    back = read_vcc(vcc, tmp_path / "mine")
    original = ad.read_h5ad(h5ad)
    restored = ad.read_h5ad(back)
    assert restored.shape == original.shape
    assert (restored.X != original.X).nnz == 0
    assert list(restored.obs.columns) == list(original.obs.columns)


def test_matches_the_official_prep(built, tmp_path):
    cli = _cli()
    if cli is None:
        pytest.skip("official vcc CLI not installed; nothing to compare against")
    bundle, h5ad, out = built

    theirs = out / "theirs.vcc"
    result = subprocess.run(
        [
            cli,
            "prep",
            str(h5ad),
            "-g",
            str(bundle.root / "gene_names.csv"),
            "--perts",
            str(bundle.root / "pert_counts.csv"),
            "--cells-per-pert",
            str(bundle.cells_per_pert),
            "--contexts",
            ",".join(bundle.contexts),
            "-o",
            str(theirs),
            "-f",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    mine = write_vcc(h5ad, out / "mine2.vcc")
    a = ad.read_h5ad(read_vcc(mine, tmp_path / "a"))
    b = ad.read_h5ad(read_vcc(theirs, tmp_path / "b"))

    assert a.shape == b.shape
    assert (a.X != b.X).nnz == 0
    assert a.X.dtype == b.X.dtype
    assert np.array_equal(np.asarray(a.var_names), np.asarray(b.var_names))
    assert np.array_equal(np.asarray(a.obs_names), np.asarray(b.obs_names))
    for col in b.obs.columns:
        assert col in a.obs.columns, f"prep keeps {col!r}; we dropped it"
        assert np.array_equal(a.obs[col].astype(str).to_numpy(), b.obs[col].astype(str).to_numpy())
    assert set(a.obs.columns) == set(b.obs.columns)


def test_archive_layout_matches(built, tmp_path):
    import tarfile

    cli = _cli()
    if cli is None:
        pytest.skip("official vcc CLI not installed")
    _bundle, h5ad, out = built
    mine = write_vcc(h5ad, out / "mine3.vcc")
    with tarfile.open(mine) as tar:
        names = tar.getnames()
        info = tar.getmember(PRED_MEMBER)
    assert names == [PRED_MEMBER]
    assert info.uid == 0 and info.gid == 0 and info.mtime == 0
    assert info.uname == "" and info.gname == ""
