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


class _Constant:
    """A target-specific signal that ignores the target, for scale tests."""

    def __init__(self, row):
        self._row = np.asarray(row, dtype=float)

    def predict_log2fc(self, targets):
        return np.repeat(self._row[None, :], len(targets), axis=0)


def _mixed(generic, specific, scale_generic=1.0, scale_specific=1.0):
    counts, genes = _context()
    cfg = BudgetConfig(n_calls=15, specific_weight=1.0, generic_weight=0.4)
    model = BudgetedPredictor(
        cfg, generic * scale_generic, specific=_Constant(specific * scale_specific)
    ).fit(counts, genes)
    return model.predict(["G3"])


def test_a_mixing_weight_is_a_ratio_not_a_unit():
    """The same response arrives as a log2 fold change in one place and as a
    library delta six times smaller in another; a weight fitted in one has to
    mean the same thing in the other."""
    rng = np.random.default_rng(3)
    generic, specific = rng.normal(size=60), rng.normal(size=60)
    base = _mixed(generic, specific)
    for sg, ss in ((5.9, 1.0), (1.0, 0.17), (100.0, 0.01)):
        other = _mixed(generic, specific, sg, ss)
        assert np.array_equal(other.call, base.call)
        assert np.array_equal(other.sign, base.sign)


def test_mixing_actually_mixes():
    """Guard the test above from passing because the generic side is ignored."""
    rng = np.random.default_rng(4)
    generic, specific = rng.normal(size=60), rng.normal(size=60)
    counts, genes = _context()
    only = BudgetedPredictor(
        BudgetConfig(n_calls=15, specific_weight=1.0, generic_weight=0.0),
        generic,
        specific=_Constant(specific),
    ).fit(counts, genes)
    assert not np.array_equal(only.predict(["G3"]).call, _mixed(generic, specific).call)


def test_magnitude_gamma_puts_the_target_into_the_sizes():
    """`pds` ranks a predicted profile against every real one, and a size that
    depends only on expression is the same coordinate for every target."""
    counts, genes = _context()
    rng = np.random.default_rng(7)
    generic = rng.normal(size=genes.size)
    flat = BudgetedPredictor(
        BudgetConfig(n_calls=20, generic_weight=1.0), generic
    ).fit(counts, genes)
    graded = BudgetedPredictor(
        BudgetConfig(n_calls=20, generic_weight=1.0, magnitude_gamma=1.0), generic
    ).fit(counts, genes)
    a, b = flat.predict(["G3"]), graded.predict(["G3"])
    # Same genes, same directions -- only the sizes move.
    assert np.array_equal(a.call, b.call)
    assert np.array_equal(a.sign, b.sign)
    assert not np.allclose(a.lfc, b.lfc)


def test_a_graded_call_never_drops_below_its_threshold():
    """Below the significance threshold a call simply does not happen, so the
    grading is only ever allowed to make a call larger."""
    counts, genes = _context()
    rng = np.random.default_rng(8)
    generic = rng.normal(size=genes.size)
    cfg = dict(n_calls=25, generic_weight=1.0, top_margin=1.15, margin=1.15)
    flat = BudgetedPredictor(BudgetConfig(**cfg), generic).fit(counts, genes)
    graded = BudgetedPredictor(
        BudgetConfig(**cfg, magnitude_gamma=1.5), generic
    ).fit(counts, genes)
    a, b = flat.predict(["G3"]), graded.predict(["G3"])
    called = a.call[0]
    assert (np.abs(b.lfc[0][called]) >= np.abs(a.lfc[0][called]) - 1e-9).all()
    assert np.abs(b.lfc[0][called]).max() <= np.abs(a.lfc[0][called]).max() * 3.0 + 1e-9
