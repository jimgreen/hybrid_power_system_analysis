"""Numerical contracts for LF optimizations; no timing assertions."""
import numpy as np
import pytest


def test_grouped_balancing_matches_independent_allocations():
    from lfcore.common import allocate_grouped_residual, allocate_limited_residual

    rng = np.random.default_rng(4162)
    for count in (0, 1, 8, 150):
        positions = rng.integers(0, 40, size=count)
        baseline = rng.normal(size=count)
        target = rng.normal(size=40) * 12
        lower = baseline - rng.uniform(0, 3, count)
        upper = baseline + rng.uniform(0, 3, count)
        alpha = rng.choice([0., 1., 3., np.nan], size=count)
        lower[::3] = np.nan
        upper[::4] = np.inf
        expected = baseline.copy()
        for node in np.unique(positions):
            mask = positions == node
            expected[mask] = allocate_limited_residual(
                baseline[mask], target[node], lower=lower[mask],
                upper=upper[mask], alpha=alpha[mask],
            )
        actual = allocate_grouped_residual(
            positions, baseline, target, lower=lower, upper=upper, alpha=alpha,
        )
        np.testing.assert_array_equal(actual, expected)


def test_grouped_balancing_validates_optional_vector_lengths():
    from lfcore.common import allocate_grouped_residual

    with pytest.raises(ValueError):
        allocate_grouped_residual([0], [1.], [2.], lower=[0., 1.])


def test_result_voltage_lookup_preserves_dense_and_sparse_id_behavior():
    from lfcore.common import build_result_voltage_lookup

    dense_rows = np.array([[1., 0., 1.01], [2., 0., 1.02], [3., 0., 1.03]])
    dense_lookup = build_result_voltage_lookup(dense_rows, 0, 2)
    np.testing.assert_array_equal(
        dense_lookup(np.array([1., 3., 9.])),
        np.array([1.01, 1.03, 0.0]),
    )

    sparse_rows = np.array([[10., 0., 1.10], [3., 0., 1.03], [7., 0., 1.07]])
    sparse_lookup = build_result_voltage_lookup(sparse_rows, 0, 2)
    np.testing.assert_array_equal(
        sparse_lookup(np.array([3., 7., 10., 11.])),
        np.array([1.03, 1.07, 1.10, 0.0]),
    )


def test_public_call_parameters_remain_parallel():
    import inspect
    from lfcore.ac_lf import ACPowerFlowCalc
    from lfcore.dc_lf import DCPowerFlowCalc
    from lfcore.hybrid_lf import HybridPowerFlowCalc

    classes = (ACPowerFlowCalc, DCPowerFlowCalc, HybridPowerFlowCalc)
    for name in ('__init__', 'from_file_fast', 'prepare', 'run', 'get_f', 'get_jacobi'):
        names = [list(inspect.signature(getattr(cls, name)).parameters) for cls in classes]
        assert names[0] == names[1] == names[2]


@pytest.mark.parametrize('multi_energy', [False, True])
def test_hybrid_residual_evaluation_does_not_fill_electric_jacobians(tmp_path, monkeypatch, multi_energy):
    from pathlib import Path
    from lfcore.hybrid_lf import HybridPowerFlowCalc
    root = Path(__file__).resolve().parents[1]
    if multi_energy:
        from test_hybrid_multi_energy import _build_multi_energy_case
        path, _ = _build_multi_energy_case(tmp_path)
    else:
        path = root / 'data/model/hybrid/qinling.e'
    calc = HybridPowerFlowCalc.from_file_fast(path, linear_solver='scipy', result_mode='array')
    calc.prepare()
    candidate = calc.x.copy()
    expected, _ = calc._build_newton_system(candidate)
    expected = expected.copy()
    def reject(*args, **kwargs):
        raise AssertionError('residual evaluation must not assemble electric Jacobian values')
    for sub in (calc.ac_calc, calc.dc_calc):
        monkeypatch.setattr(sub, '_get_jacobi_from_precomputed_pattern', reject)
    actual = calc.get_f(candidate)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('kind,case', [('ac', 'ieee300'), ('dc', 'dc_net_1000'), ('hybrid', 'qinling')])
def test_result_modes_preserve_state_and_previous_result_snapshots(kind, case):
    from pathlib import Path
    from lfcore.ac_lf import ACPowerFlowCalc
    from lfcore.dc_lf import DCPowerFlowCalc
    from lfcore.hybrid_lf import HybridPowerFlowCalc

    cls = dict(ac=ACPowerFlowCalc, dc=DCPowerFlowCalc, hybrid=HybridPowerFlowCalc)[kind]
    root = Path(__file__).resolve().parents[1]
    path = root / 'data/model' / kind / (case + '.e')
    options = dict(tol=1e-8, max_iter=50, linear_solver='scipy', verbose=False)
    reference = None
    for mode in ('full', 'array', 'summary', 'none'):
        calc = cls.from_file_fast(path, result_mode=mode, **options)
        assert calc.run() == 0
        state = calc.x.copy()
        if reference is None:
            reference = state
        np.testing.assert_allclose(state, reference, atol=1e-12, rtol=1e-12)
        assert calc.converged
        assert (calc.lf_result is None) == (mode != 'full')
        if mode == 'none':
            assert calc.result == {}
        # Full -> array must not retain a stale object result.
        if mode == 'full':
            saved = calc.result
            def arrays(obj):
                if isinstance(obj, dict):
                    return {key: arrays(value) for key, value in obj.items()}
                return obj.copy() if isinstance(obj, np.ndarray) else obj
            saved_copy = arrays(saved)
            assert calc.run(result_mode='array') == 0
            assert calc.lf_result is None
            def check(a, b):
                if isinstance(a, dict):
                    for key in a:
                        check(a[key], b[key])
                elif isinstance(a, np.ndarray):
                    np.testing.assert_array_equal(a, b)
                else:
                    assert a == b
            check(saved, saved_copy)
