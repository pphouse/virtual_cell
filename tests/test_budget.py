"""The call set a budgeted prediction asserts, and the four forms it comes in.

Three of the four differential-expression members read the set rather than the
fold changes -- the Jaccard reads which genes are in it, fidelity reads their
direction, reach reads the order they are ranked in -- so the offline validator
scores `BudgetedPredictor.predict` directly.  These check that what it returns
is the same prediction the emitter sends, expressed four ways.
"""

import numpy as np
import scipy.sparse as sp

from vcc2026.budget import LN2, BudgetConfig, BudgetedPredictor


def _context(seed: int = 0, n_cells: int = 200, n_genes: int = 60):
    rng = np.random.default_rng(seed)
    depth = rng.integers(30, 400, size=n_genes)
    counts = rng.poisson(depth[None, :].astype(float), size=(n_cells, n_genes))
    genes = np.array([f"G{i}" for i in range(n_genes)])
    return sp.csr_matrix(counts.astype(np.float32)), genes


def _fit(n_calls: int, seed: int = 0, **over):
    counts, genes = _context(seed)
    rng = np.random.default_rng(seed + 1)
    generic = rng.normal(size=genes.size)
    cfg = BudgetConfig(n_calls=n_calls, **over)
    return BudgetedPredictor(cfg, generic).fit(counts, genes), genes


def test_budget_is_the_number_of_calls():
    model, genes = _fit(n_calls=10)
    pred = model.predict(["G3", "G7"])
    assert pred.call.sum(1).tolist() == [10, 10]
    assert (np.count_nonzero(pred.lfc, axis=1) == 10).all()


def test_lfc_is_non_zero_exactly_on_the_call_set():
    model, _ = _fit(n_calls=12)
    pred = model.predict(["G1", "G2", "G3"])
    assert ((pred.lfc != 0.0) == pred.call).all()


def test_predict_lfc_agrees_with_predict():
    model, _ = _fit(n_calls=15)
    targets = ["G0", "G5"]
    assert np.array_equal(model.predict_lfc(targets), model.predict(targets).lfc)


def test_log2_undoes_the_natural_log():
    model, _ = _fit(n_calls=8)
    pred = model.predict(["G4"])
    assert np.allclose(pred.log2() * LN2, pred.lfc)


def test_the_target_never_spends_a_call_on_itself():
    """Every scored member drops the perturbation's own gene, so a call there
    buys nothing and costs one of the budget."""
    model, genes = _fit(n_calls=50)
    own = int(np.flatnonzero(genes == "G9")[0])
    pred = model.predict(["G9"])
    assert not pred.call[0, own]
    assert pred.lfc[0, own] == 0.0


def test_direction_matches_the_sign_of_the_emitted_fold_change():
    model, _ = _fit(n_calls=20)
    pred = model.predict(["G2", "G6"])
    called = pred.call
    assert (np.sign(pred.lfc[called]) == pred.sign[called]).all()


def test_confidence_orders_the_magnitudes_within_a_call_set():
    """Reach walks the reference's genes in the submission's own confidence
    order, so a higher ranking score has to carry a larger fold change."""
    model, _ = _fit(n_calls=25, top_margin=2.0, margin=1.0)
    pred = model.predict(["G11"])
    take = np.flatnonzero(pred.call[0])
    order = np.argsort(-pred.key[0][take])
    ramp = np.abs(pred.lfc[0][take][order]) / model.magnitude[take][order]
    assert np.all(np.diff(ramp) <= 1e-6)
