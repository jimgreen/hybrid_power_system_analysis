"""Numerical and output contracts for the sixth uncached SE optimization."""
from dataclasses import asdict
import inspect
from pathlib import Path

import numpy as np
import pytest

from secore.ac_se import ACStateEstimator
from secore.dc_se import DCStateEstimator
from secore.hybrid_se import HybridStateEstimator
from model.meas_model import (MeasurementTable, ObservabilityResult, EstimateResult,
                              BadDataItem, Measurement, MEAS_STATUS_PSEUDO)
from secore.se_result import SEResult, build_seresult_full_from_table
import secore.se_result as result_module

ROOT = Path(__file__).resolve().parents[1]
CASES = [(ACStateEstimator, 'ac', 'ac_net_30'), (DCStateEstimator, 'dc', 'dc_net_30'),
         (HybridStateEstimator, 'hybrid', 'hybrid_net_40'),
         (HybridStateEstimator, 'ac', 'ac_net_30'), (HybridStateEstimator, 'dc', 'dc_net_30')]

def make(case, **kwargs):
    cls, side, name = case
    return cls(e_file=ROOT/f'data/model/{side}/{name}.e',
               meas_file=ROOT/f'data/meas/{side}/{name}.meas', flat_start=True, **kwargs)

def test_constructor_common_keywords_preserve_defaults():
    sigs = [inspect.signature(cls) for cls in (ACStateEstimator, DCStateEstimator, HybridStateEstimator)]
    expected = set(sigs[0].parameters)
    for signature in sigs[1:]:
        assert set(signature.parameters) == expected
        for name in expected - {'e_file', 'meas_file'}:
            assert signature.parameters[name].default == sigs[0].parameters[name].default
    # Existing positional callers must not be reinterpreted.
    assert list(sigs[0].parameters)[9] == 'network'
    assert list(sigs[1].parameters)[9] == 'network'
    assert list(sigs[2].parameters)[9] == 'auto_prepare'

@pytest.mark.parametrize('case', CASES)
def test_initial_state_explicit_flat_preserves_default_and_returns_copy(case):
    est = make(case)
    original = est.initial_state()
    for flat in (False, True):
        actual = est.initial_state(flat=flat)
        delegate = est._delegate() if isinstance(est, HybridStateEstimator) else est
        reference = delegate if delegate is not None else est
        old = reference.flat_start
        try:
            reference.flat_start = flat
            expected = reference.initial_state()
        finally:
            reference.flat_start = old
        np.testing.assert_array_equal(actual, expected)
        actual[:] = 123
        np.testing.assert_array_equal(est.initial_state(flat=flat), expected)
    assert est.flat_start is True
    np.testing.assert_array_equal(est.initial_state(), original)

@pytest.mark.parametrize('case', CASES)
def test_matrix_dump_is_optional_and_does_not_change_solution(case, tmp_path):
    plain, dumped = make(case), make(case, matrix_dump_dir=tmp_path)
    for est in (plain, dumped):
        est.run(result_mode='array', skip_bad_data=True, final_diagnostics=False)
    np.testing.assert_array_equal(dumped.estimate_result.x, plain.estimate_result.x)
    np.testing.assert_array_equal(dumped.estimate_result.residual, plain.estimate_result.residual)
    files = list(tmp_path.rglob('j1.txt'))
    assert files
    for file in files:
        assert (file.parent/'d1.txt').is_file()
        assert (file.parent/'h1.txt').is_file()
        assert file.read_text().strip()

def sample_table(n, strings):
    count = {'empty':0, 'partial':min(3,n), 'full':n}[strings]
    text = lambda prefix: np.array([f'{prefix}_{i}' for i in range(count)], dtype=object)
    return MeasurementTable(
        idx=np.arange(n,dtype=np.int64)+2**53, name=text('测量'), device_type=text('device'),
        device_name=text('节点'), meas_type=text('V'), weight=np.ones(n), valid=np.ones(n,dtype=bool),
        value=np.full(n,-0.0), device_type_code=np.full(n,1,dtype=np.int16), angle_mask=np.zeros(n,dtype=bool),
        status_code=np.zeros(n,dtype=np.int16), meas_type_code=np.full(n,1,dtype=np.int16),
        device_pos=np.arange(n,dtype=np.int32)-1)

@pytest.mark.parametrize('n', [0,1,8191,8192,8193])
@pytest.mark.parametrize('strings', ['empty','partial','full'])
def test_batched_rows_match_scalar_contract(n, strings):
    table = sample_table(n,strings)
    positions = np.arange(n,dtype=np.intp)
    z = np.full(max(0,n-2),-0.0)
    residual = np.arange(n,dtype=np.float64)
    actual = list(result_module._measurement_rows_from_table(table,positions,estimated=z,
                  residual=residual,normalized=None,source='normal'))
    expected = [SEResult.MeasurementRow.from_table_row(table,i,
        estimated_value=SEResult._array_value(z,i),residual=float(residual[i]),source='normal') for i in positions]
    assert [asdict(row) for row in actual] == [asdict(row) for row in expected]
    for row in actual:
        assert type(row.idx) is int and type(row.valid) is bool
        assert type(row.value) is float and np.signbit(row.value)
        if row.estimated_value is not None:
            assert np.signbit(row.estimated_value)

def test_full_result_preserves_classification_order_bad_rows_and_statistics():
    table = sample_table(7,'partial')
    table.status_code[[1,4]] = MEAS_STATUS_PSEUDO
    all_table = sample_table(9,'full')
    all_table.valid[7] = False
    all_table.weight[8] = 0
    obs = ObservabilityResult(True,2,2,7,0,np.ones(2),[])
    estimate = EstimateResult(True,2,.2,.001,.1,np.ones(2),np.arange(7.),np.arange(7.)/10,
        None,None,[],obs,measurement_table=table)
    bad = [BadDataItem(None,.3,4.,1.1,1.4,3), BadDataItem(None,.4,5.,1.2,1.6,4),
           BadDataItem(Measurement(20,'bad','DCNode','n1','V',1.,True,2.),1.,6.,1.,2.,-1)]
    normalized = np.arange(4.)
    result = build_seresult_full_from_table(estimate,bad_items=bad,normalized_residual=normalized,
        all_measurement_table=all_table,active_source_rows=np.arange(7))
    assert [r.idx for r in result.normal_measurements] == table.idx[[0,2,5,6]].tolist()
    assert [r.idx for r in result.pseudo_measurements] == table.idx[[1,4]].tolist()
    assert [r.idx for r in result.bad_data] == [int(table.idx[3]),int(table.idx[4]),20]
    assert [r.reason for r in result.prefiltered_measurements] == ['invalid','zero weight']
    assert result.statistics.prefiltered_measurement_count == 2
    assert result.statistics.pseudo_measurement_count == 2
    assert result.statistics.bad_data_count == 3
    assert result.statistics.normal_measurement_count == 2  # legacy statistics contract
    for pos,row in zip([0,2,5,6],result.normal_measurements):
        expected = SEResult.MeasurementRow.from_table_row(table,pos,estimated_value=estimate.z_est[pos],
            residual=estimate.residual[pos],normalized_residual=SEResult._array_value(normalized,pos),source='normal')
        assert asdict(row) == asdict(expected)
    assert result.normal_measurements[2].normalized_residual is None


def test_hybrid_fixed_jacobian_refresh_matches_fresh_and_finite_difference():
    est = make(CASES[2])
    x = est.initial_state()
    est.jacobian_sparse(x)
    x = x + .001 * np.sin(np.arange(x.size))
    actual = est.jacobian_sparse(x).toarray().copy()
    reference = make(CASES[2])
    np.testing.assert_array_equal(actual, reference.jacobian_sparse(x).toarray())
    for column in np.unique(np.linspace(0,x.size-1,10,dtype=int)):
        step = 1e-6
        upper, lower = x.copy(), x.copy()
        upper[column] += step
        lower[column] -= step
        numeric = (est.evaluate(upper)-est.evaluate(lower))/(2*step)
        np.testing.assert_allclose(actual[:,column],numeric,rtol=2e-5,atol=2e-7)


def test_multi_energy_matrix_dump_contains_joint_system(tmp_path):
    from test_hybrid_multi_energy import _build_multi_energy_case
    case, measurements = _build_multi_energy_case(tmp_path)
    est = HybridStateEstimator(e_file=case,meas_file=measurements,flat_start=True,
                               max_iter=2,matrix_dump_dir=tmp_path/'matrices')
    assert est.fluid_estimators
    est.run(result_mode='array',skip_bad_data=True,final_diagnostics=False)
    assert (tmp_path/'matrices/j1.txt').is_file()
    assert (tmp_path/'matrices/d1.txt').is_file()
    assert (tmp_path/'matrices/h1.txt').is_file()
