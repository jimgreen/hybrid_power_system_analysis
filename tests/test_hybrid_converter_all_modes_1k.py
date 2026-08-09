import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "hybrid_power_system_analysis"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from efile_read import EBook
from lfcore.hybrid_lf import HybridPowerFlowCalc, _read_lf_network_from_file
from model.meas_type import (
    DEVICE_TYPE_ACACConverter,
    DEVICE_TYPE_DCACConverter,
    DEVICE_TYPE_DCDCConverter,
)
from scripts.check_hybrid_converter_all_modes_1k import (
    EXPECTED_ACAC_MODES,
    EXPECTED_DCAC_MODES,
    EXPECTED_DCDC_MODES,
    _decode_control_modes,
    _network_statistics,
)
from secore.hybrid_se import HybridStateEstimator


CASE = ROOT / "data" / "model" / "hybrid" / "hybrid_converter_all_modes_1k.e"
MEASUREMENTS = ROOT / "data" / "meas" / "hybrid" / "hybrid_converter_all_modes_1k.meas"


class HybridConverterAllModes1KTest(unittest.TestCase):
    def test_case_covers_every_solver_supported_converter_mode(self):
        network = _read_lf_network_from_file(CASE)
        calc = HybridPowerFlowCalc(network, result_mode="array", verbose=False)
        calc.prepare()

        self.assertGreater(
            calc.ac_calc.ppc["bus"].shape[0] + calc.dc_calc.ppc["bus"].shape[0],
            1000,
        )
        modes = {
            key: frozenset(tuple(mode) for mode in values)
            for key, values in _decode_control_modes(calc).items()
        }
        self.assertEqual(EXPECTED_ACAC_MODES, modes["acac"])
        self.assertEqual(EXPECTED_DCDC_MODES, modes["dcdc"])
        self.assertEqual(EXPECTED_DCAC_MODES, modes["dcac"])
        self.assertEqual(
            {"DCACConverter": 3, "ACDCConverter": 1},
            _network_statistics(calc)["dcac_device_types"],
        )

        measurement_rows = EBook(MEASUREMENTS).data["Measurement"].data
        measured_device_types = {str(row["dev_type"]) for row in measurement_rows}
        self.assertNotIn("ACSwitch", measured_device_types)
        self.assertNotIn("DCSwitch", measured_device_types)

    def test_hybrid_lf_converges_from_flat_start(self):
        network = _read_lf_network_from_file(CASE)
        calc = HybridPowerFlowCalc(network, result_mode="array", verbose=False)

        self.assertEqual(0, calc.run())
        self.assertTrue(calc.converged)
        self.assertLess(np.max(np.abs(calc.get_f(calc.x))), 1e-9)
        self.assertFalse(np.any(calc.dc_calc.last_dcdc_loss_infeasible_mask))

    def test_hybrid_se_is_observable_and_recovers_converter_measurements(self):
        estimator = HybridStateEstimator(
            CASE,
            MEASUREMENTS,
            flat_start=True,
            max_iter=50,
        )
        observability = estimator.observability_analysis()
        result = estimator.estimate(
            observability=observability,
            final_diagnostics=False,
        )

        self.assertTrue(observability.observable)
        self.assertEqual(observability.rank, observability.state_count)
        self.assertTrue(result.converged)
        self.assertLess(result.objective, 1e-12)
        self.assertLess(result.residual_inf, 2e-8)

        device_type = np.asarray(result.measurement_table.device_type_code, dtype=np.int16)
        for code in (
            DEVICE_TYPE_ACACConverter,
            DEVICE_TYPE_DCDCConverter,
            DEVICE_TYPE_DCACConverter,
        ):
            rows = device_type == code
            self.assertTrue(np.any(rows))
            self.assertLess(np.max(np.abs(result.residual[rows])), 2e-8)


if __name__ == "__main__":
    unittest.main()
