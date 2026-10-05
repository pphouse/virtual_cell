"""The Context-Conditioned Delta Transfer model, on a multi-cell-line simulator.

These exercise the *transfer* half of the system -- rebasing a measured
signature into an unseen context, extrapolating to targets nobody perturbed,
and holding out a whole cell line to score it.  The submission format is tested
separately in `test_submission_format.py`, against the real 2026 rules.

The simulator is not challenge data.  A number here is evidence the wiring is
right, never a prediction of leaderboard performance.
"""

import numpy as np
import pytest

from vcc2026 import (
    CellContext,
    ContextTransferModel,
    ModelConfig,
    build_signature_library,
)
from vcc2026.calibrate import evaluate_holdout, sweep
from vcc2026.localeval import baseline_pair, evaluate, score_against_baseline
from vcc2026.normalize import normlog

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


def test_alpha_sweep_runs_and_moves_the_score(sim):
    _, _, _, lib, ctx = sim
    rows = sweep(lib, HELD_OUT, ctx, "alpha", [0.25, 0.75, 1.25], config=ModelConfig())
    assert len(rows) == 3
    scores = [r["avg_score"] for r in rows]
    assert max(scores) > min(scores), "alpha should matter"


def test_a_confidently_wrong_prediction_scores_below_baseline():
    """2026 has no floor at zero, so losing a metric is a real cost.

    The 2025 scorer clipped every metric at the baseline, which made an
    aggressive prediction free to be wrong. That is not how the 2026 aggregate
    works -- only expression accuracy stops at 0 -- so the local proxy must not
    reproduce the old clip or it will recommend over-confident settings.
    """
    user = {"mae": 2.0, "mse": 2.0, "pearson_delta": 0.1}
    base = {"mae": 1.0, "mse": 1.0, "pearson_delta": 0.5}
    s = score_against_baseline(user, base)
    assert s["mse"] == 0.0, "expression accuracy is the one metric with a floor"
    assert s["mae"] < 0, "everything else can go negative"
    assert s["pearson_delta"] < 0
    assert s["avg_score"] < 0


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


def _library_without(lib, hidden, drop_line):
    trimmed = type(lib)(genes=lib.genes)
    for line in lib.lines:
        if line == drop_line:
            continue
        trimmed.baseline[line] = lib.baseline[line]
        trimmed.deltas[line] = {t: d for t, d in lib.deltas[line].items() if t not in hidden}
        trimmed.n_cells[line] = dict(lib.n_cells[line])
    return trimmed


def test_a_target_measured_elsewhere_ignores_the_similarity(sim):
    """Why the panel understated the co-essentiality blend by twelve times.

    A prediction is `b * d_direct + (1 - b) * d_knn`, and only `d_knn` reads the
    gene similarity.  `b` is set by whether the *target itself* carries a
    signature anywhere in the source library, so a target measured in another
    line answers from its own rebased measurement and barely moves when the
    similarity changes underneath it.

    Every one of the offline panel's K562 targets, and 83% of RPE1's, is
    measured in the lines left in the library, while none of the real panel's
    300 are.  So the offline panel was scoring a regime the competition does not
    have: it reported +0.0005 for a change that scored +0.0062 on two seeds
    (`docs/08` 22, 23).  Holding each target's own signature out restores the
    regime, and this test fails if that stops being true.
    """
    from vcc2026.features import GeneFeatures

    _, targets, _, lib, ctx = sim
    scored = sorted(targets)[:15]

    rng = np.random.default_rng(0)
    real = GeneFeatures(genes=lib.genes, matrix=rng.normal(size=(lib.genes.size, 8)))
    real.matrix /= np.linalg.norm(real.matrix, axis=1, keepdims=True)
    shuffled = GeneFeatures(genes=lib.genes, matrix=real.matrix[rng.permutation(lib.genes.size)])

    def spread(library):
        a, b = (
            ContextTransferModel(ModelConfig()).fit(library, features=f).predict(scored, ctx)
            for f in (real, shuffled)
        )
        # how far the two similarities drive the predictions apart, per target
        num = np.linalg.norm(a.delta - b.delta, axis=1)
        den = np.linalg.norm(a.delta, axis=1) + np.linalg.norm(b.delta, axis=1)
        return float(np.mean(num / np.maximum(den, 1e-12)))

    measured_elsewhere = _library_without(lib, hidden=set(), drop_line=HELD_OUT)
    held_out = _library_without(lib, hidden=set(scored), drop_line=HELD_OUT)

    diluted = spread(measured_elsewhere)
    faithful = spread(held_out)
    assert faithful > 2 * diluted, (
        "holding the targets' own signatures out must expose the similarity: "
        f"moved {faithful:.3f} held out against {diluted:.3f} left in"
    )
