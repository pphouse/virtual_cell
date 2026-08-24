"""The submission format rules, checked on a miniature bundle.

Every one of these is a rule the scorer enforces, and most of them reject the
whole upload rather than degrading gracefully -- so they are worth a test that
runs in a second rather than a discovery during the final submission window.
"""

import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from vcc2026.challenge import CONTEXT_COL, PERT_COL, Bundle
from vcc2026.coexpression import ControlOnlyConfig, ControlOnlyPredictor
from vcc2026.counts import calibrate_pseudocount, context_mean_proportions, emit_counts
from vcc2026.submission import SubmissionConfig, build_submission, response_scales

N_GENES = 120
N_PERTS = 6
N_CONTROL = 300
CELLS_PER_PERT = 20
CONTEXTS = ["A", "B", "C"]


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    """A structurally faithful miniature of the real controls bundle."""
    root = tmp_path_factory.mktemp("bundle")
    rng = np.random.default_rng(0)
    genes = np.array([f"G{i:04d}" for i in range(N_GENES)], dtype=object)
    perts = genes[rng.choice(N_GENES, size=N_PERTS, replace=False)]

    pd.Series(genes.astype(str)).to_frame("gene_name").to_csv(root / "gene_names.csv", index=False)
    pd.Series(perts.astype(str)).to_frame("target_gene").to_csv(
        root / "pert_counts.csv", index=False
    )

    for ctx in CONTEXTS:
        latent = rng.normal(size=(N_CONTROL, 4))
        loading = rng.normal(size=(4, N_GENES))
        rate = np.exp(1.2 + 0.4 * (latent @ loading))
        rate = 3000 * rate / rate.sum(axis=1, keepdims=True)
        x = sp.csr_matrix(rng.poisson(rate).astype(np.float32))
        obs = pd.DataFrame(
            {
                PERT_COL: pd.Categorical(["non-targeting"] * N_CONTROL),
                CONTEXT_COL: pd.Categorical([ctx] * N_CONTROL),
            },
            index=np.arange(N_CONTROL).astype(str),
        )
        ad.AnnData(X=x, obs=obs, var=pd.DataFrame(index=pd.Index(genes.astype(str)))).write_h5ad(
            root / f"context_{ctx}.h5ad"
        )
    (root / "manifest.json").write_text(
        json.dumps({"contexts": CONTEXTS, "cells_per_pert": CELLS_PER_PERT})
    )
    return Bundle.load(root)


@pytest.fixture(scope="module")
def submission(bundle, tmp_path_factory):
    out = tmp_path_factory.mktemp("out") / "submission.h5ad"

    def predictor_for(context, counts, genes):
        return ControlOnlyPredictor(ControlOnlyConfig(n_components=8, trans_beta=0.25)).fit(
            counts, genes
        )

    build_submission(bundle, predictor_for, out, SubmissionConfig(seed=1))
    return ad.read_h5ad(out)


def test_bundle_reads_labels_from_the_files(bundle):
    assert bundle.contexts == CONTEXTS
    assert bundle.perturbations.size == N_PERTS
    assert bundle.genes.size == N_GENES
    assert bundle.cells_per_pert == CELLS_PER_PERT
    assert bundle.check_targets_predictable().all()


def test_mislabelled_context_file_is_refused(bundle, tmp_path):
    """A context file whose contents disagree with its name must not be guessed at."""
    bad = ad.read_h5ad(bundle.context_files["A"])
    bad.obs[CONTEXT_COL] = pd.Categorical(["B"] * bad.n_obs)
    path = tmp_path / "context_A.h5ad"
    bad.write_h5ad(path)
    (tmp_path / "gene_names.csv").write_text((bundle.root / "gene_names.csv").read_text())
    (tmp_path / "pert_counts.csv").write_text((bundle.root / "pert_counts.csv").read_text())
    broken = Bundle.load(tmp_path)
    with pytest.raises(ValueError, match="refusing to guess"):
        build_submission(
            broken,
            lambda c, x, g: ControlOnlyPredictor(ControlOnlyConfig()).fit(x, g),
            tmp_path / "out.h5ad",
        )


def test_every_context_is_present_exactly_once(submission, bundle):
    counts = submission.obs[CONTEXT_COL].astype(str).value_counts()
    assert sorted(counts.index) == CONTEXTS
    for ctx in CONTEXTS:
        assert counts[ctx] == N_PERTS * CELLS_PER_PERT


def test_exact_perturbation_set_and_cell_counts(submission, bundle):
    official = set(bundle.perturbations.astype(str))
    for ctx in CONTEXTS:
        sub = submission.obs[submission.obs[CONTEXT_COL].astype(str) == ctx]
        got = sub[PERT_COL].astype(str)
        assert set(got) == official, "perturbation set must match the official list exactly"
        counts = got.value_counts()
        assert (counts == CELLS_PER_PERT).all(), "one cell over or under is rejected"


def test_no_control_cells(submission):
    assert (submission.obs[PERT_COL].astype(str) == "non-targeting").sum() == 0


def test_raw_non_negative_integer_counts(submission):
    data = submission.X.data
    assert np.allclose(data, np.round(data)), "fractional values are rejected outright"
    assert (data >= 0).all()
    assert np.isfinite(data).all()


def test_within_the_scorer_caps(submission):
    per_cell = np.asarray(submission.X.sum(axis=1)).ravel()
    assert per_cell.max() <= 1_000_000
    assert submission.n_obs <= 400_000
    assert submission.X.nnz <= 4_750_000_000
    # a dense array is over the density cap on its own, so sparsity is not optional
    assert submission.X.nnz < submission.n_obs * submission.n_vars


def test_gene_space_matches_the_bundle(submission, bundle):
    assert np.array_equal(np.asarray(submission.var_names, dtype=str), bundle.genes.astype(str))


def test_on_target_knockdown_is_predicted_down(bundle):
    controls = ad.read_h5ad(bundle.context_files["A"])
    model = ControlOnlyPredictor(ControlOnlyConfig(trans_beta=0.0)).fit(
        controls.X.tocsr(), bundle.genes
    )
    targets = [str(t) for t in bundle.perturbations]
    lfc = model.predict_lfc(targets)
    pos = {g: i for i, g in enumerate(bundle.genes)}
    for p, t in enumerate(targets):
        assert lfc[p, pos[t]] < 0
        assert np.count_nonzero(lfc[p]) == 1, "trans_beta=0 must move nothing else"


def test_unexpressed_target_falls_back_to_no_change(bundle):
    """The context mean scores 0; a confident guess about a silent gene can score below it."""
    controls = ad.read_h5ad(bundle.context_files["A"])
    x = controls.X.tolil()
    silent = 3
    x[:, silent] = 0
    model = ControlOnlyPredictor(ControlOnlyConfig(trans_beta=0.5)).fit(x.tocsr(), bundle.genes)
    lfc = model.predict_lfc([str(bundle.genes[silent])])
    assert np.count_nonzero(lfc) == 0


def test_emission_preserves_library_size_and_stays_sparse():
    rng = np.random.default_rng(0)
    base = sp.csr_matrix(rng.poisson(0.5, size=(50, N_GENES)).astype(np.float32))
    mean_p = context_mean_proportions(base)
    out = emit_counts(base, mean_p, np.zeros(N_GENES), pseudocount=1.0, rng=rng)
    got = np.asarray(out.sum(axis=1)).ravel()
    want = np.asarray(base.sum(axis=1)).ravel()
    assert abs(got.mean() - want.mean()) / want.mean() < 0.05
    assert out.nnz < base.shape[0] * N_GENES


def test_emission_applies_the_requested_fold_change():
    rng = np.random.default_rng(0)
    base = sp.csr_matrix(rng.poisson(4.0, size=(400, N_GENES)).astype(np.float32))
    mean_p = context_mean_proportions(base)
    lfc = np.zeros(N_GENES)
    lfc[7] = np.log(0.25)
    out = emit_counts(base, mean_p, lfc, pseudocount=1.0, rng=rng)

    def share(m, j):
        col = np.asarray(m[:, j].todense()).ravel().sum()
        return col / m.sum()

    ratio = share(out, 7) / share(base, 7)
    assert 0.15 < ratio < 0.40, ratio
    untouched = share(out, 8) / share(base, 8)
    assert 0.9 < untouched < 1.1


def test_response_scales_have_mean_one():
    rng = np.random.default_rng(0)
    cfg = SubmissionConfig(non_responder_fraction=0.25, response_shape=2.0)
    r = response_scales(5000, cfg, rng)
    assert abs(r.mean() - 1.0) < 1e-9
    assert (r == 0).mean() > 0.15
    flat = response_scales(100, SubmissionConfig(), rng)
    assert np.all(flat == 1.0), "the default must be a deterministic full effect"


def test_pseudocount_calibration_matches_real_dispersion():
    """The emission must be a statistical no-op when nothing is perturbed."""
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(1200, 5))
    loading = rng.normal(size=(5, N_GENES))
    rate = np.exp(1.2 + 0.4 * (latent @ loading))
    rate = 4000 * rate / rate.sum(axis=1, keepdims=True)
    controls = sp.csr_matrix(rng.poisson(rate).astype(np.float32))
    best, rows = calibrate_pseudocount(controls, n_probe=500)
    chosen = next(r for r in rows if r["pseudocount"] == best)
    assert 0.7 < chosen["var_ratio"] < 1.4, chosen

    # The mechanism the calibration relies on: the control cell already carries
    # one round of measurement noise and the Poisson draw adds a second, so
    # shrinking toward the context mean is what removes the excess dispersion.
    ratios = [r["var_ratio"] for r in sorted(rows, key=lambda r: r["pseudocount"])]
    assert ratios[-1] < ratios[0], ratios
