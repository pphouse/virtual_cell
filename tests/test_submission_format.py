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
from vcc2026.counts import context_mean_proportions, emit_counts
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


def test_a_null_prediction_emits_the_cells_unchanged():
    """The property the first submission lacked, and it cost 0.34 on one metric.

    Resampling every gene made the emission look right on per-gene variance and
    detection rate while still handing the scorer ~93 spuriously significant
    genes per perturbation, 89% of them in the same direction. Now the emission
    is the identity wherever the model predicts nothing, so every DE call the
    scorer makes traces back to a deliberate prediction.
    """
    rng = np.random.default_rng(0)
    base = sp.csr_matrix(rng.poisson(0.5, size=(50, N_GENES)).astype(np.float32))
    mean_p = context_mean_proportions(base)
    out = emit_counts(base, mean_p, np.zeros(N_GENES), pseudocount=1.0, rng=rng)
    assert (out != base).nnz == 0

    # ... and a per-cell fold change of zero is the same null
    out2 = emit_counts(base, mean_p, np.zeros((50, N_GENES)), pseudocount=1.0, rng=rng)
    assert (out2 != base).nnz == 0


def test_emission_touches_only_the_predicted_genes():
    rng = np.random.default_rng(0)
    base = sp.csr_matrix(rng.poisson(2.0, size=(80, N_GENES)).astype(np.float32))
    mean_p = context_mean_proportions(base)
    lfc = np.zeros(N_GENES)
    lfc[[3, 11]] = np.log(0.2)
    out = emit_counts(base, mean_p, lfc, pseudocount=1.0, rng=rng)

    untouched = np.ones(N_GENES, dtype=bool)
    untouched[[3, 11]] = False
    assert (out[:, untouched] != base[:, untouched]).nnz == 0
    for j in (3, 11):
        assert out[:, j].sum() < base[:, j].sum()


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
    assert ratio == pytest.approx(0.25, rel=0.15), ratio
    assert share(out, 8) == pytest.approx(share(base, 8), rel=0.05)


def test_the_pseudocount_does_not_dilute_a_knockdown():
    """The prior opens the way up; it must not soften the way down.

    Added symmetrically it inflates the mass a fold change multiplies, so an
    intended knockdown to 0.15 of baseline arrives at 0.30. The first
    submission shipped that way.
    """
    rng = np.random.default_rng(1)
    base = sp.csr_matrix(rng.poisson(6.0, size=(400, N_GENES)).astype(np.float32))
    mean_p = context_mean_proportions(base)
    lfc = np.zeros(N_GENES)
    lfc[5] = np.log(0.15)
    for a in (0.0, 1.0, 4.0):
        out = emit_counts(base, mean_p, lfc, pseudocount=a, rng=np.random.default_rng(2))
        delivered = out[:, 5].sum() / base[:, 5].sum()
        assert delivered == pytest.approx(0.15, rel=0.20), (a, delivered)


def test_the_pseudocount_enables_upregulation_of_a_silent_gene():
    rng = np.random.default_rng(3)
    base = sp.csr_matrix(rng.poisson(3.0, size=(400, N_GENES)).astype(np.float32))
    silent = base.tolil()
    silent[:, 9] = 0
    silent = silent.tocsr()
    mean_p = context_mean_proportions(base)  # the context still expresses it
    lfc = np.zeros(N_GENES)
    lfc[9] = np.log(3.0)

    without = emit_counts(silent, mean_p, lfc, pseudocount=0.0, rng=np.random.default_rng(4))
    assert without[:, 9].sum() == 0, "with no prior a silent gene can never come up"

    with_prior = emit_counts(silent, mean_p, lfc, pseudocount=1.0, rng=np.random.default_rng(4))
    assert with_prior[:, 9].sum() > 0


def test_response_scales_have_mean_one():
    rng = np.random.default_rng(0)
    cfg = SubmissionConfig(non_responder_fraction=0.25, response_shape=2.0)
    r = response_scales(5000, cfg, rng)
    assert abs(r.mean() - 1.0) < 1e-9
    assert (r == 0).mean() > 0.15
    flat = response_scales(100, SubmissionConfig(), rng)
    assert np.all(flat == 1.0), "the default must be a deterministic full effect"
