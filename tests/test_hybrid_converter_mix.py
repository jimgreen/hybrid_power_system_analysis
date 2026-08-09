import sys
import unittest
import warnings
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "hybrid_power_system_analysis"))

from lfcore.hybrid_lf import HybridPowerFlowCalc, _read_lf_network_from_file
from ac_array_model import SHUNT_COLS
from model.meas_type import (
    DEVICE_TYPE_ACACConverter,
    DEVICE_TYPE_DCACConverter,
    DEVICE_TYPE_DCDCConverter,
)
from scripts.update_meas_from_lf import (
    _reconstruct_ac_ideal_edge_flows,
    _reconstruct_dc_ideal_edge_flows,
)
from secore.hybrid_se import HybridStateEstimator


CASE = ROOT / "data" / "model" / "hybrid" / "hybrid_converter_mix.e"
MEASUREMENTS = ROOT / "data" / "meas" / "hybrid" / "hybrid_converter_mix.meas"


class HybridConverterMixTest(unittest.TestCase):
    def test_hybrid_lf_owns_only_dcac_coupling_block(self):
        network = _read_lf_network_from_file(CASE)
        calc = HybridPowerFlowCalc(network, result_mode="full", verbose=False)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rc = calc.run()

        self.assertEqual(0, rc)
        self.assertTrue(calc.converged)
        self.assertEqual(1, calc.ac_calc.N_acac)
        self.assertEqual(9, calc.dc_calc.N_dcdc)
        self.assertEqual(3, calc.N_dcac)
        self.assertEqual(calc.total_vars, calc.total_eq)
        self.assertLess(calc.normF, 1e-8)
        self.assertFalse(caught)

        residual = calc.get_f(calc.x)
        self.assertLess(np.max(np.abs(residual[: calc.ac_eq])), 1e-8)
        self.assertLess(
            np.max(np.abs(residual[calc.ac_eq : calc.dcac_eq_start])),
            1e-8,
        )
        self.assertLess(np.max(np.abs(residual[calc.dcac_eq_start :])), 1e-8)
        self.assertFalse(np.any(calc.dc_calc.last_dcdc_loss_infeasible_mask))

        shunts = {device.name: device for device in network.ac.shunt_compensators}
        self.assertTrue(all(device.run_stat == 1 for device in shunts.values()))
        self.assertAlmostEqual(0.2, shunts["shunt_3"].q, places=12)
        self.assertGreater(shunts["shunt_4"].q, 0.0)
        self.assertGreater(abs(shunts["shunt_5"].q), 1.0)
        self.assertGreater(shunts["shunt_5"].current, 1.0)
        ac_explicit_devices = list(network.ac.zero_branches) + list(network.ac.breakers)
        ac_explicit_before = np.asarray(
            [(device.p, device.q, device.current) for device in ac_explicit_devices],
            dtype=np.float64,
        )
        dc_explicit_devices = list(network.dc.zero_branches) + list(network.dc.breakers)
        dc_explicit_before = np.asarray(
            [(device.p, device.current) for device in dc_explicit_devices],
            dtype=np.float64,
        )
        self.assertLess(
            _reconstruct_ac_ideal_edge_flows(
                network.ac,
                network.dcac_converters,
                network.acac_converters,
            ),
            1e-8,
        )
        self.assertLess(
            _reconstruct_dc_ideal_edge_flows(network.dc, network.dcac_converters),
            1e-8,
        )
        np.testing.assert_allclose(
            [(device.p, device.q, device.current) for device in ac_explicit_devices],
            ac_explicit_before,
            rtol=0.0,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            [(device.p, device.current) for device in dc_explicit_devices],
            dc_explicit_before,
            rtol=0.0,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            [(device.p, device.current) for device in network.dc.switches],
            0.0,
            rtol=0.0,
            atol=1e-12,
        )

    def test_hybrid_se_recovers_lf_measurements_from_flat_start(self):
        estimator = HybridStateEstimator(
            CASE,
            MEASUREMENTS,
            flat_start=True,
            max_iter=50,
        )
        estimator.run(result_mode="summary", verbose=False)

        observability = estimator.observability_result
        result = estimator.estimate_result
        self.assertTrue(observability.observable)
        self.assertEqual(observability.rank, observability.state_count)
        self.assertTrue(result.converged)
        self.assertLess(result.objective, 1e-12)
        self.assertLess(result.residual_inf, 1e-8)
        self.assertFalse(estimator.bad_items)

        ac_estimator = estimator._ac_sub_estimator
        ac_estimator.apply_state(estimator._ac_sub_state_from_hybrid(result.x))
        shunt = ac_estimator._ac_ppc_dict()["shunt"]
        self.assertAlmostEqual(0.2, shunt[0, SHUNT_COLS["q"]], places=12)
        self.assertGreater(shunt[1, SHUNT_COLS["q"]], 0.0)
        self.assertGreater(abs(shunt[2, SHUNT_COLS["q"]]), 1.0)
        self.assertGreater(shunt[2, SHUNT_COLS["current"]], 1.0)

        table = result.measurement_table
        device_type_code = np.asarray(table.device_type_code, dtype=np.int16)
        for code in (
            DEVICE_TYPE_ACACConverter,
            DEVICE_TYPE_DCACConverter,
            DEVICE_TYPE_DCDCConverter,
        ):
            rows = device_type_code == int(code)
            self.assertTrue(np.any(rows))
            self.assertLess(np.max(np.abs(result.residual[rows])), 1e-8)


if __name__ == "__main__":
    unittest.main()
