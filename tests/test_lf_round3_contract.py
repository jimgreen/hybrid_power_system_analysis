"""Boundary and numerical contracts for the third LF optimization pass."""
import numpy as np
import pytest


@pytest.mark.parametrize('ac', [False, True])
def test_bulk_current_bases_match_scalar_units(ac):
    from model.array_common import _current_scale_by_node
    from unit_system import ac_current_base_ka, dc_current_base_ka

    scalar = ac_current_base_ka if ac else dc_current_base_ka
    voltages = np.array([0., 1e-12, -1e-12, 1.01e-12, -.38, 10., np.inf, np.nan, .72])
    # Reordered/sparse IDs and duplicates must keep dict's last-row-wins rule.
    ids = np.array([10., 2., 900., 4., 9., 6., 7., 8., 10.])
    expected = {int(k): 1000. * scalar(1234., float(v)) for k, v in zip(ids, voltages)}
    actual = _current_scale_by_node(ids, voltages, 1234., 1000., ac=ac)
    assert list(actual) == list(expected)
    np.testing.assert_array_equal(list(actual.values()), list(expected.values()))
    assert _current_scale_by_node([], [], 1., 1., ac=ac) == {}


def test_result_names_convert_mixed_sequences_to_strings():
    from lfcore.common import result_name_list

    assert result_name_list(['bus', 2, None], 3) == ['bus', '2', 'None']


@pytest.mark.parametrize('terminal', ['ac_node', 'dc_node'])
@pytest.mark.parametrize('value,error', [(np.nan, ValueError), (np.inf, OverflowError)])
def test_dcac_invalid_terminal_ids_preserve_scalar_errors(terminal, value, error):
    from types import SimpleNamespace
    from lfcore.hybrid_lf import HybridPowerFlowCalc, DCAC_COLS

    table = np.zeros((1, max(DCAC_COLS.values()) + 1))
    table[0, DCAC_COLS['run_stat']] = 1
    table[0, DCAC_COLS[terminal]] = value
    calc = object.__new__(HybridPowerFlowCalc)
    calc.network = SimpleNamespace(ppc={'dcac': table})
    calc.ac_calc = SimpleNamespace(_active_node_solver_lookup=np.array([0]), node_pos={0: 0})
    calc.dc_calc = SimpleNamespace(_alive_node_lookup=np.array([0]), alive_node_dict={0: 0})
    with pytest.raises(error):
        calc._prepare_dcac_converters_from_ppc()


@pytest.mark.parametrize('mode', ['full', 'array'])
def test_zero_branch_jacobian_matches_finite_difference(mode):
    from pathlib import Path
    from lfcore.hybrid_lf import HybridPowerFlowCalc

    path = Path(__file__).resolve().parents[1] / 'data/model/hybrid/hybrid_net_40.e'
    calc = HybridPowerFlowCalc.from_file_fast(path, result_mode=mode, linear_solver='scipy', verbose=False)
    calc.prepare()
    ac = calc.ac_calc
    assert ac.N_phi > 0
    x = ac.x.copy()
    x += np.random.default_rng(17).normal(0., .002, x.size)
    analytic = ac.get_jacobi(x).toarray().copy()
    numeric = np.empty_like(analytic)
    step = 1e-6
    for col in range(x.size):
        delta = np.zeros_like(x)
        delta[col] = step
        plus = ac.get_f(x + delta).copy()
        minus = ac.get_f(x - delta).copy()
        numeric[:, col] = (plus - minus) / (2 * step)
    np.testing.assert_allclose(analytic, numeric, atol=2e-5, rtol=2e-6)
