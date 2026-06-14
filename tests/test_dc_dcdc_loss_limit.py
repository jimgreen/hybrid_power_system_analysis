import warnings

import numpy as np
import pytest

from dc_lf import DCPowerFlowCalc


def _minimal_calc(r1=0.075, r2=0.075):
    calc = DCPowerFlowCalc.__new__(DCPowerFlowCalc)
    calc.dcdc_r1 = np.asarray([r1], dtype=np.float64)
    calc.dcdc_r2 = np.asarray([r2], dtype=np.float64)
    calc._dcdc_loss_warning_tags = set()
    return calc


def test_dcdc_loss_infeasible_discriminant_warns_and_uses_limit_model():
    calc = _minimal_calc()
    pi = np.asarray([-6.433597469498245], dtype=np.float64)
    vi = np.asarray([1.2], dtype=np.float64)
    vj = np.asarray([1.477599], dtype=np.float64)

    with pytest.warns(RuntimeWarning, match="DCDC.*infeasible"):
        pj, dpj_dpi, dpj_dvi, dpj_dvj = calc._dcdc_j_power_from_loss(pi, vi, vj)

    expected_limit = vj[0] * vj[0] / (2.0 * calc.dcdc_r2[0])
    assert np.isclose(pj[0], expected_limit)
    assert dpj_dpi[0] == 0.0
    assert dpj_dvi[0] == 0.0
    assert np.isclose(dpj_dvj[0], vj[0] / calc.dcdc_r2[0])


def test_dcdc_loss_near_singular_denominator_warns_and_uses_limit_derivative():
    calc = _minimal_calc(r1=0.1, r2=0.1)
    pi = np.asarray([-2.0710678118654755], dtype=np.float64)
    vi = np.asarray([1.0], dtype=np.float64)
    vj = np.asarray([1.0], dtype=np.float64)

    with pytest.warns(RuntimeWarning, match="DCDC.*limit"):
        pj, dpj_dpi, dpj_dvi, dpj_dvj = calc._dcdc_j_power_from_loss(pi, vi, vj)

    assert np.isclose(pj[0], 5.0)
    assert dpj_dpi[0] == 0.0
    assert dpj_dvi[0] == 0.0
    assert np.isclose(dpj_dvj[0], 10.0)


def test_dcdc_loss_warning_is_not_repeated_for_same_calc_and_condition():
    calc = _minimal_calc()
    pi = np.asarray([-6.433597469498245], dtype=np.float64)
    vi = np.asarray([1.2], dtype=np.float64)
    vj = np.asarray([1.477599], dtype=np.float64)

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        calc._dcdc_j_power_from_loss(pi, vi, vj)
        calc._dcdc_j_power_from_loss(pi, vi, vj)

    matching = [item for item in records if "DCDC" in str(item.message)]
    assert len(matching) == 1
