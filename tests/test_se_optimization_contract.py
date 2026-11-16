"""Public SE contracts and numerical checks for the uncached optimization."""
import inspect
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from model.meas_array_model import build_meas_ppc_from_e_file
from secore.ac_se import ACStateEstimator
from secore.dc_se import DCStateEstimator
from secore.hybrid_se import HybridStateEstimator
import secore.hybrid_se as hybrid_module


ROOT = Path(__file__).resolve().parents[1]
CASES = [
    (ACStateEstimator, 'ac', 'ac_net_30'),
    (DCStateEstimator, 'dc', 'dc_net_30'),
    (HybridStateEstimator, 'hybrid', 'hybrid_net_40'),
    (HybridStateEstimator, 'ac', 'ac_net_30'),
    (HybridStateEstimator, 'dc', 'dc_net_30'),
]


def make_estimator(case, **kwargs):
    cls, side, name = case
    return cls(e_file=ROOT / f'data/model/{side}/{name}.e',
               meas_file=ROOT / f'data/meas/{side}/{name}.meas',
               flat_start=True, **kwargs)


def solve(estimator, **kwargs):
    estimator.run(verbose=False, skip_bad_data=True, final_diagnostics=False,
                  result_mode=kwargs.pop('result_mode', 'array'), **kwargs)
    return estimator.estimate_result


@pytest.mark.parametrize('method', ['prepare', 'finalize_prepare', 'run', 'estimate', 'evaluate', 'jacobian_sparse', 'observability_analysis'])
def test_public_method_parameters_match(method):
    def parameters(cls):
        return [(p.name, p.kind, p.default) for p in
                inspect.signature(getattr(cls, method)).parameters.values()]
    assert parameters(ACStateEstimator) == parameters(DCStateEstimator)
    assert parameters(ACStateEstimator) == parameters(HybridStateEstimator)


@pytest.mark.parametrize('case', CASES)
def test_staged_prepare_matches_eager(case):
    eager = make_estimator(case)
    staged = make_estimator(case, defer_prepare_finalize=True)
    assert not staged._prepared
    assert staged.finalize_prepare() is staged
    assert staged._prepared
    np.testing.assert_allclose(staged.active_z, eager.active_z, rtol=0, atol=0)
    np.testing.assert_allclose(solve(staged).x, solve(eager).x, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize('case', CASES)
def test_inactive_prepare_can_be_finalized(case):
    eager = make_estimator(case)
    staged = make_estimator(case, prepare_active_measurements=False)
    assert staged._prepared
    assert staged.n_state == eager.n_state
    assert len(staged.active_measurements) == 0
    assert staged.active_z.size == staged.active_weight.size == 0
    staged.finalize_prepare()
    np.testing.assert_allclose(staged.active_z, eager.active_z, rtol=0, atol=0)
    np.testing.assert_allclose(solve(staged).x, solve(eager).x, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize('case', CASES)
def test_evaluate_measurements_keyword_and_out(case):
    estimator = make_estimator(case)
    result = solve(estimator)
    expected = estimator.evaluate(result.x, estimator.active_measurements)
    out = np.full(expected.shape, np.nan)
    actual = estimator.evaluate(result.x, measurements=estimator.active_measurements, out=out)
    assert actual is out
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('case', CASES)
def test_evaluate_preserves_old_keyword_and_rejects_ambiguous_input(case):
    estimator = make_estimator(case)
    x = estimator.initial_state()
    np.testing.assert_array_equal(
        estimator.evaluate(x, measurement_plan_tables=estimator.active_measurements),
        estimator.evaluate(x, measurements=estimator.active_measurements),
    )
    with pytest.raises(TypeError, match='either measurements'):
        estimator.evaluate(x, estimator.active_measurements, measurements=estimator.active_measurements)


@pytest.mark.parametrize('case', CASES)
def test_explicit_inputs_and_replacement(case):
    estimator = make_estimator(case)
    original = build_meas_ppc_from_e_file(estimator.meas_file)
    supplied = deepcopy(original)
    # Changing the measurement values must not retain the old active plan.
    supplied['value_array'] = supplied['value_array'].copy() * 1.0001
    fresh = make_estimator(case, network=deepcopy(estimator.network), measurements=deepcopy(supplied))
    estimator.prepare(network=deepcopy(estimator.network), measurements=deepcopy(supplied))
    np.testing.assert_array_equal(estimator.active_z, fresh.active_z)
    np.testing.assert_allclose(solve(estimator).x, solve(fresh).x, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize('case', CASES[:3])
def test_output_modes_preserve_numerical_result(case):
    expected = solve(make_estimator(case))
    for mode in ('full', 'none', 'summary', 'array'):
        actual = solve(make_estimator(case), result_mode=mode)
        assert actual.converged == expected.converged
        assert actual.iterations == expected.iterations
        np.testing.assert_array_equal(actual.x, expected.x)
        np.testing.assert_array_equal(actual.residual, expected.residual)


@pytest.mark.parametrize('case', CASES[2:])
def test_hybrid_ppc_input_supports_both_and_single_sides(case):
    original = make_estimator(case)
    supplied = make_estimator(case, network=original.network.ppc)
    np.testing.assert_allclose(solve(supplied).x, solve(original).x, rtol=1e-11, atol=1e-12)


def test_compatibility_device_columns_match_scalar_reference():
    ppc = {'branch': np.array([[1, 2, 3, 1, .25], [4, 3, 5, 0, .7]]),
           'branch_name': np.array(['a', 'b'], dtype=object)}
    cols = {'idx': 0, 'i_node': 1, 'j_node': 2, 'run_stat': 3, 'r': 4}
    attrs = dict(idx='idx', i_node='i_node', j_node='j_node', run_stat='run_stat', r='r', x='x')
    actual = HybridStateEstimator._compat_devices(ppc, 'branch', 'branch_name', cols, 'branch', attrs)
    for device, row, name in zip(actual, ppc['branch'], ppc['branch_name']):
        assert device.name == name
        for attr, column in attrs.items():
            expected = float(row[cols[column]]) if column in cols else 0.0
            if attr in ('idx', 'i_node', 'j_node', 'run_stat'):
                expected = int(expected)
            assert getattr(device, attr) == expected
            assert type(getattr(device, attr)) is type(expected)
        assert device.is_alive == (int(row[3]) == 1)


@pytest.mark.parametrize('max_iter', [1, 2])
def test_rejected_line_search_recomputes_fallback_result(monkeypatch, max_iter):
    # One-dimensional h(x)=x, z=0. Force an uphill step so every trial is
    # rejected, including when the fallback is the last permitted iteration.
    estimator = HybridStateEstimator.__new__(HybridStateEstimator)
    estimator._prepared = True
    estimator.fluid_estimators = {}
    estimator.profile_enabled = False
    estimator.n_state = 1
    estimator.max_iter = max_iter
    estimator.tol = 1e-10
    estimator.voltage_cols = np.array([], dtype=np.int32)
    estimator.active_measurements = [object()]
    estimator.active_uniform_weight = 1.0
    estimator.active_weights_are_uniform = True
    estimator._active_normal_pattern = None
    estimator._array_only_estimate_result = True
    estimator._delegate = lambda: None
    estimator._normalize_measurements = lambda _: estimator.active_measurements
    estimator._observability_matrix_cache_for = lambda *args: None
    estimator._measurement_vectors = lambda _: (np.zeros(1), np.ones(1))
    estimator.jacobian_sparse = lambda *args: np.ones((1, 1))
    estimator._measurement_residual = lambda z, h, m, out: np.subtract(z, h, out=out)
    evaluated = []

    def evaluate(x, measurements, out):
        evaluated.append(x.copy())
        out[:] = x
        return out

    estimator.evaluate = evaluate

    class UphillSolver:
        def __init__(self, **kwargs):
            self.steps = 0

        def solve(self, *args, **kwargs):
            self.steps += 1
            return np.array([1.0 if self.steps == 1 else 0.0]), None

    monkeypatch.setattr(hybrid_module, 'NormalEquationSolver', UphillSolver)
    result = estimator.estimate(x0=np.ones(1), observability=object(), final_diagnostics=False)
    np.testing.assert_array_equal(result.x, [2.0])
    np.testing.assert_array_equal(result.z_est, [2.0])
    np.testing.assert_array_equal(result.residual, [-2.0])
    assert result.objective == 2.0  # one half of the weighted squared residual
    assert len(evaluated) == 14  # initial + twelve rejected trials + fallback


@pytest.mark.parametrize('side,stem', [('heat', 'heat_net_3'), ('gas', 'gas_net_3'),
                                      ('hydro', 'hydro_net_3'), ('steam', 'steam_net_5')])
def test_fluid_deferred_prepare(side, stem):
    case = (HybridStateEstimator, side, stem)
    eager = make_estimator(case)
    staged = make_estimator(case, defer_prepare_finalize=True)
    assert not staged._prepared
    assert not staged.fluid_estimators
    staged.finalize_prepare()
    assert set(staged.fluid_estimators) == {side}
    np.testing.assert_allclose(solve(staged).x, solve(eager).x, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize('case', CASES[2:])
def test_hybrid_normalized_measurement_input_is_not_scaled_twice(case):
    original = make_estimator(case)
    source = make_estimator(case, prepare_active_measurements=False)
    supplied = make_estimator(case, measurements=deepcopy(source.measurements))
    np.testing.assert_allclose(supplied.active_z, original.active_z, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(solve(supplied).x, solve(original).x, rtol=1e-10, atol=1e-11)


def test_hybrid_replacement_publishes_new_topology_and_preserves_old_on_error():
    estimator = make_estimator(CASES[2])
    replacement = make_estimator(CASES[3])
    original_network = estimator.network
    with pytest.raises(TypeError):
        estimator.prepare(network={})
    assert estimator.network is original_network
    estimator.prepare(network=replacement.network, measurements=build_meas_ppc_from_e_file(replacement.meas_file))
    assert estimator.n_state == replacement.n_state
    assert list(estimator.state_labels) == list(replacement.state_labels)
    np.testing.assert_allclose(solve(estimator).x, solve(replacement).x, rtol=1e-11, atol=1e-12)


def test_se_instances_read_network_files_independently(monkeypatch):
    calls = []
    original_read = hybrid_module._read_efile_rows

    def read(path):
        calls.append(path)
        return original_read(path)

    monkeypatch.setattr(hybrid_module, '_read_efile_rows', read)
    first = make_estimator(CASES[2])
    second = make_estimator(CASES[2])
    assert len(calls) == 2
    assert first.network is not second.network
    assert not np.shares_memory(first.network.ppc['ac']['bus'], second.network.ppc['ac']['bus'])


@pytest.mark.parametrize('case', CASES)
def test_auto_prepare_false_never_calls_prepare(case, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError('constructor called prepare')
    monkeypatch.setattr(case[0], 'prepare', unexpected)
    estimator = make_estimator(case, auto_prepare=False)
    assert not estimator._prepared
    assert not hasattr(estimator, 'network')


def test_lazy_metadata_preserves_small_case_labels_and_sequence_bounds():
    estimator = make_estimator(CASES[2])
    assert estimator._lazy_state_layout
    expected = [meta.legacy_label for meta in estimator.state_meta]
    assert list(estimator.state_labels) == expected
    assert list(estimator.ac_state_labels) == expected[:estimator.ac_n_state]
    for view in (estimator.state_meta, estimator.state_sides, estimator.state_labels):
        assert view[-1] == view[len(view) - 1]
        with pytest.raises(IndexError):
            view[-len(view) - 1]
        with pytest.raises(IndexError):
            view[len(view)]


@pytest.mark.parametrize('side', ['ac', 'dc'])
def test_voltage_defaults_match_scalar_without_scanning_per_reference(side, monkeypatch):
    from types import SimpleNamespace
    estimator = HybridStateEstimator.__new__(HybridStateEstimator)
    ids = np.arange(100, 500, 2, dtype=np.int64)
    voltage = np.linspace(0.9, 1.1, ids.size)
    voltage[3] = 0.0
    refs = {i: 1.2 + i / 1000 for i in reversed(range(0, ids.size, 4))}
    refs[8] = 0.0  # reference values, including zero, take precedence
    child = SimpleNamespace(ref_voltages=refs, file_voltage=voltage,
                            _node_voltage_by_pos=voltage,
                            _ac_node_id_lookup_ids=ids, _ac_node_id_lookup_pos=np.arange(ids.size),
                            _node_idx_lookup_ids=ids, _node_idx_lookup_pos=np.arange(ids.size))
    setattr(estimator, '_' + side + '_sub_estimator', child)
    setattr(estimator, 'network', SimpleNamespace(**{
        side: SimpleNamespace(node_dict={999: SimpleNamespace(voltage=1.7)})}))
    nodes = np.r_[ids[::-1], ids[::3], 999, -10]
    scalar = getattr(estimator, '_' + side + '_voltage_default_for_node')
    expected = np.array([scalar(node) for node in nodes])
    calls = []
    original = np.flatnonzero
    def count_flatnonzero(values):
        calls.append(len(values))
        return original(values)
    monkeypatch.setattr(hybrid_module.np, 'flatnonzero', count_flatnonzero)
    actual = getattr(estimator, '_' + side + '_voltage_defaults_for_nodes')(nodes)
    np.testing.assert_array_equal(actual, expected)
    assert len(calls) <= 5


@pytest.mark.parametrize('case', CASES[:3])
def test_external_jacobian_plan_updates_values_and_rebuilds_on_row_change(case):
    from secore.se_array_plan import take_measurement_view
    estimator = make_estimator(case)
    reference = make_estimator(case)
    rows = np.arange(0, len(estimator.active_measurements), 2)
    subset = take_measurement_view(estimator.active_measurements, rows)
    x = estimator.initial_state()
    initial = estimator.jacobian_sparse(x, subset).toarray().copy()
    changed_x = x + 0.001 * np.sin(np.arange(x.size))
    # Reusing the same plan must update values while preserving its row layout.
    actual = estimator.jacobian_sparse(changed_x, subset).toarray().copy()
    fresh_subset = take_measurement_view(reference.active_measurements, rows)
    expected = reference.jacobian_sparse(changed_x, fresh_subset).toarray()
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(estimator.jacobian_sparse(x, subset).toarray(), initial)
    # A different plan with the same row count must not inherit the old mapping.
    reversed_subset = take_measurement_view(estimator.active_measurements, rows[::-1])
    actual = estimator.jacobian_sparse(changed_x, reversed_subset).toarray()
    np.testing.assert_array_equal(actual, expected[::-1])
