import numpy as np
import pytest

from vcc import (
    CellContext,
    CellSampler,
    ContextTransferModel,
    ModelConfig,
    SamplerConfig,
    build_signature_library,
)
from vcc.calibrate import evaluate_holdout, sweep
from vcc.localeval import baseline_pair, evaluate, score_against_baseline
from vcc.normalize import normlog
from vcc.submit import validate

from .synthetic import SimConfig, simulate

HELD_OUT = "line3"


@pytest.fixture(scope="module")
def sim():
    genes, targets, adatas = simulate(SimConfig())
    lib = build_signature_library({k: v for k, v in adatas.items()}, genes=genes)
    ctx = CellContext.from_anndata(adatas[HELD_OUT], name=HELD_OUT, genes=genes)
    return genes, targets, adatas, lib, ctx


def test_library_shape(sim):
    genes, targets, _, lib, _ = sim
    assert set(lib.lines) == {f"line{i}" for i in range(4)}
    for line in lib.lines:
        assert len(lib.targets(line)) == len(targets)
        assert lib.baseline[line].shape == (genes.size,)


def test_context_only_sees_controls(sim):
    genes, _, adatas, _, ctx = sim
    n_ctrl = int((adatas[HELD_OUT].obs["target_gene"] == "non-targeting").sum())
    assert ctx.n_control == n_ctrl
    assert ctx.mu.shape == (genes.size,)


def test_zero_shot_beats_control_baseline(sim):
    _, _, _, lib, ctx = sim
    metrics, scores, pair = evaluate_holdout(lib, HELD_OUT, ctx, config=ModelConfig())
    base = evaluate(baseline_pair(pair))

    assert metrics["pearson_delta"] > 0.3, metrics
    assert metrics["discrimination_score_l1"] > 0.6, metrics
    assert metrics["overlap_at_N_proxy"] > base["overlap_at_N_proxy"], (metrics, base)
    assert scores["avg_score"] > 0.05, scores


def test_unseen_targets_still_predicted(sim):
    """The realistic case: the target gene was never perturbed in any source line."""
    _, targets, _, lib, ctx = sim
    hidden = set(targets[:15])
    trimmed = type(lib)(genes=lib.genes)
    for line in lib.lines:
        if line == HELD_OUT:
            continue
        trimmed.baseline[line] = lib.baseline[line]
        trimmed.deltas[line] = {t: d for t, d in lib.deltas[line].items() if t not in hidden}
        trimmed.n_cells[line] = dict(lib.n_cells[line])

    model = ContextTransferModel(ModelConfig()).fit(trimmed)
    pred = model.predict(sorted(hidden), ctx)
    assert not pred.observed.any(), "these targets must be unobserved"
    assert (pred.magnitude > 0).all()

    gene_pos = {g: i for i, g in enumerate(ctx.genes)}
    real = np.vstack([lib.deltas[HELD_OUT][t] for t in sorted(hidden)])
    r = [np.corrcoef(pred.delta[i], real[i])[0, 1] for i in range(len(hidden))]
    assert np.mean(r) > 0.2, f"kNN extrapolation failed: mean r={np.mean(r):.3f}"
    # the on-target gene must be driven down even with no signature for it --
    # except where the gene is already silent here, which cannot go any lower
    expressed = 0
    for i, t in enumerate(sorted(hidden)):
        j = gene_pos[t]
        assert pred.delta[i][j] <= 0
        if ctx.mu[j] > 0.05:
            assert pred.delta[i][j] < 0
            expressed += 1
    assert expressed >= 10


def test_sampler_preserves_pseudobulk_mean_exactly(sim):
    """Heterogeneity must be free: the emitted pseudobulk delta is the predicted one.

    Verified on a degenerate context whose control cells are all identical, so
    the only thing that can move the mean is the response-scale draw.
    """
    _, _, _, lib, ctx = sim
    model = ContextTransferModel(ModelConfig()).fit(lib)
    targets = lib.targets("line0")[:8]
    pred = model.predict(targets, ctx)

    flat = CellContext(
        name=ctx.name,
        genes=ctx.genes,
        control=np.repeat(ctx.mu[None, :], 400, axis=0),
    )
    for hetero in [(0.0, 8.0), (0.3, 2.0), (0.6, 1.0)]:
        cfg = SamplerConfig(non_responder_fraction=hetero[0], response_shape=hetero[1])
        adata = CellSampler(cfg).sample(flat, pred, {t: 150 for t in targets})
        x = adata.X.toarray()
        labels = adata.obs["target_gene"].astype(str).to_numpy()
        base = x[labels == "non-targeting"].mean(axis=0)
        assert np.abs(base - np.clip(flat.mu, 0, None)).max() < 1e-3
        for i, t in enumerate(targets):
            emitted = x[labels == t].mean(axis=0) - base
            assert np.abs(emitted - pred.delta[i]).max() < 1e-3, hetero


def test_sampler_anchors_means_on_the_full_control_pool(sim):
    """Resampling noise must not reach the pseudobulk means the metrics read."""
    _, _, _, lib, ctx = sim
    model = ContextTransferModel(ModelConfig()).fit(lib)
    targets = lib.targets("line0")[:5]
    pred = model.predict(targets, ctx)
    adata = CellSampler(SamplerConfig()).sample(ctx, pred, {t: 40 for t in targets})
    x = adata.X.toarray()
    labels = adata.obs["target_gene"].astype(str).to_numpy()
    ctrl = x[labels == "non-targeting"].mean(axis=0)
    assert np.abs(ctrl - np.clip(ctx.mu, 0, None)).max() < 1e-3
    for i, t in enumerate(targets):
        want = np.clip(ctx.mu + pred.delta[i], 0, None)
        assert np.abs(x[labels == t].mean(axis=0) - want).max() < 1e-3


def test_sampler_matches_requested_cell_counts(sim):
    _, _, _, lib, ctx = sim
    model = ContextTransferModel(ModelConfig()).fit(lib)
    targets = lib.targets("line0")[:6]
    pred = model.predict(targets, ctx)
    want = {t: 10 + 7 * i for i, t in enumerate(targets)}
    adata = CellSampler(SamplerConfig()).sample(ctx, pred, want)
    counts = adata.obs["target_gene"].astype(str).value_counts().to_dict()
    for t, n in want.items():
        assert counts[t] == n
    assert counts["non-targeting"] > 0


def test_sampler_heterogeneity_is_real(sim):
    _, _, _, lib, ctx = sim
    model = ContextTransferModel(ModelConfig()).fit(lib)
    targets = lib.targets("line0")[:3]
    pred = model.predict(targets, ctx)
    scales = CellSampler(SamplerConfig(non_responder_fraction=0.2)).response_scales(
        5000, np.random.default_rng(0)
    )
    assert abs(scales.mean() - 1.0) < 1e-9  # mean is exact by construction
    assert (scales == 0).mean() > 0.1  # non-responders present
    assert scales.std() > 0.3
    assert pred.delta.shape == (3, ctx.genes.size)


def test_alpha_sweep_runs_and_moves_the_score(sim):
    _, _, _, lib, ctx = sim
    rows = sweep(lib, HELD_OUT, ctx, "alpha", [0.25, 0.75, 1.25], config=ModelConfig())
    assert len(rows) == 3
    scores = [r["avg_score"] for r in rows]
    assert max(scores) > min(scores), "alpha should matter"


def test_scoring_clips_losses_at_zero():
    user = {"mae": 2.0, "pearson_delta": 0.1}
    base = {"mae": 1.0, "pearson_delta": 0.5}
    s = score_against_baseline(user, base)
    assert s["mae"] == 0.0 and s["pearson_delta"] == 0.0
    assert s["avg_score"] == 0.0


def test_submission_validation(sim):
    genes, _, _, lib, ctx = sim
    model = ContextTransferModel(ModelConfig()).fit(lib)
    targets = lib.targets("line0")[:5]
    pred = model.predict(targets, ctx)
    adata = CellSampler(SamplerConfig()).sample(ctx, pred, {t: 50 for t in targets})
    validate(adata, genes=genes)

    with pytest.raises(ValueError):
        validate(adata, genes=np.array(list(genes) + ["EXTRA"]))
    with pytest.raises(ValueError):
        validate(adata, genes=genes, max_cells=10)


def test_normlog_defaults_to_median_library_size():
    """cell-eval calls scanpy normalize_total with no target_sum -- median, not 1e4."""
    counts = np.random.default_rng(0).poisson(3, size=(20, 30)).astype(float)
    x, used = normlog(counts)
    assert used == np.median(counts.sum(axis=1))
    assert np.allclose(np.expm1(x).sum(axis=1), used)
    x10k, used10k = normlog(counts, target_sum=1e4)
    assert used10k == 1e4
    assert not np.allclose(x, x10k)


def test_context_scale_matches_cell_evals_own_conversion(sim):
    """Lock in the scale finding against the real implementation.

    cell-eval normalises reference counts with `scanpy.pp.normalize_total` and
    no target_sum, i.e. to the median library size.  A submission written on
    the conventional 1e4 scale lands on a different axis and MAE -- the only
    metric in absolute expression units -- collapses.  This asserts our context
    reproduces cell-eval's conversion, and that the 1e4 convention would not.
    """
    ce = pytest.importorskip("cell_eval._evaluator")
    _, _, adatas, _, ctx = sim

    held = adatas[HELD_OUT]
    ctrl = held[held.obs["target_gene"].astype(str) == "non-targeting"].copy()
    reference = ctrl.copy()
    ce._convert_to_normlog(reference)
    ref_mean = np.asarray(reference.X).mean(axis=0)

    assert np.abs(ctx.mu - ref_mean).max() < 1e-4  # float32 storage in cell-eval

    wrong = CellContext.from_anndata(ctrl, name=HELD_OUT, target_sum=1e4)
    assert np.abs(wrong.mu - ref_mean).mean() > 0.1, (
        "simulator depth happens to equal 1e4; the test cannot show the gap"
    )
