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
    DEVICE_TYPE_ACBreak,
    DEVICE_TYPE_ACThreeWindingTransformer,
    DEVICE_TYPE_ACZeroBranch,
    DEVICE_TYPE_DCACConverter,
    DEVICE_TYPE_DCBreak,
    DEVICE_TYPE_DCDCConverter,
    DEVICE_TYPE_DCZeroBranch,
)
from scripts.check_hybrid_converter_all_modes_1k import (
    EXPECTED_ACAC_MODES,
    EXPECTED_DCAC_MODES,
    EXPECTED_DCDC_MODES,
    _decode_control_modes,
    _network_statistics,
)
from secore.hybrid_se import HybridStateEstimator


CASE = ROOT / "data" / "model" / "hybrid" / "hybrid_all_elements_5k.e"
MEASUREMENTS = ROOT / "data" / "meas" / "hybrid" / "hybrid_all_elements_5k.meas"


class HybridAllElements5KTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case_book = EBook(CASE)
        cls.measurement_rows = EBook(MEASUREMENTS).data["Measurement"].data

    def test_case_contains_all_supported_element_families(self):
        expected_nonempty = (
            "ACNode",
            "ACBranch",
            "ACZeroBranch",
            "ACSwitch",
            "ACBreak",
            "ACShuntCompensator",
            "ACTransformer",
            "ACThreeWindingTransformer",
            "ACGenerator",
            "ACLoad",
            "DCNode",
            "DCBranch",
            "DCZeroBranch",
            "DCSwitch",
            "DCBreak",
            "DCGenerator",
            "DCLoad",
            "ACACConverter",
            "DCACConverter",
            "DCDCConverter",
        )
        for block_name in expected_nonempty:
            with self.subTest(block=block_name):
                self.assertGreater(len(self.case_book.data[block_name].data), 0)

        self.assertEqual(2500, len(self.case_book.data["ACNode"].data))
        self.assertEqual(2500, len(self.case_book.data["DCNode"].data))

    def test_controls_and_switching_states_cover_the_requested_variants(self):
        rows = self.case_book.data
        self.assertEqual(
            {"PQ", "PV", "V", "P", "PH"},
            {row["control_type"] for row in rows["ACGenerator"].data},
        )
        self.assertEqual(
            {"P", "V", "I"},
            {row["control_type"] for row in rows["DCGenerator"].data},
        )
        self.assertEqual(
            {"Q", "V", "Z", "B"},
            {row["control_type"] for row in rows["ACShuntCompensator"].data},
        )
        for block_name in ("ACSwitch", "DCSwitch", "ACBreak", "DCBreak"):
            states = {
                int(row["status"])
                for row in rows[block_name].data
                if int(row["run_stat"]) == 1
            }
            with self.subTest(block=block_name):
                self.assertEqual({0, 1}, states)

        network = _read_lf_network_from_file(CASE)
        calc = HybridPowerFlowCalc(network, result_mode="array", verbose=False)
        calc.prepare()
        modes = {
            key: frozenset(tuple(mode) for mode in values)
            for key, values in _decode_control_modes(calc).items()
        }
        self.assertEqual(EXPECTED_ACAC_MODES, modes["acac"])
        self.assertEqual(EXPECTED_DCDC_MODES, modes["dcdc"])
        self.assertEqual(EXPECTED_DCAC_MODES, modes["dcac"])

        stats = _network_statistics(calc)
        for key in ("ac_switches", "dc_switches", "ac_breakers", "dc_breakers"):
            self.assertGreater(stats[key]["closed"], 0)
            self.assertGreater(stats[key]["open"], 0)

    def test_measurements_cover_results_but_exclude_contracted_switches(self):
        measured_types = {str(row["dev_type"]) for row in self.measurement_rows}
        self.assertNotIn("ACSwitch", measured_types)
        self.assertNotIn("DCSwitch", measured_types)
        for device_type in (
            "ACZeroBranch",
            "ACBreak",
            "DCZeroBranch",
            "DCBreak",
            "ACThreeWindingTransformer",
            "ACACConverter",
            "DCACConverter",
            "DCDCConverter",
        ):
            with self.subTest(device_type=device_type):
                self.assertIn(device_type, measured_types)

        third_types = {
            row["meas_type"]
            for row in self.measurement_rows
            if row["dev_type"] == "ACThreeWindingTransformer"
        }
        self.assertTrue({"P_THIRD", "Q_THIRD", "V_THIRD", "I_THIRD"} <= third_types)

    def test_hybrid_lf_converges_and_writes_all_device_results(self):
        network = _read_lf_network_from_file(CASE)
        calc = HybridPowerFlowCalc(network, result_mode="array", verbose=False)

        self.assertEqual(0, calc.run())
        self.assertTrue(calc.converged)
        self.assertLess(np.max(np.abs(calc.get_f(calc.x))), 1e-9)
        self.assertFalse(np.any(calc.dc_calc.last_dcdc_loss_infeasible_mask))

        for result, keys in (
            (
                calc.ac_calc.result,
                (
                    "branch",
                    "transformer",
                    "three_winding_transformer",
                    "shunt",
                    "zero_branch",
                    "break",
                    "acac",
                ),
            ),
            (calc.dc_calc.result, ("branch", "zero_branch", "break", "dcdc")),
        ):
            for key in keys:
                with self.subTest(result=key):
                    values = np.asarray(result[key], dtype=np.float64)
                    self.assertGreater(values.shape[0], 0)
                    self.assertTrue(np.all(np.isfinite(values)))

    def test_hybrid_se_is_observable_and_recovers_all_special_devices(self):
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

        device_codes = np.asarray(result.measurement_table.device_type_code, dtype=np.int16)
        for code in (
            DEVICE_TYPE_ACZeroBranch,
            DEVICE_TYPE_ACBreak,
            DEVICE_TYPE_DCZeroBranch,
            DEVICE_TYPE_DCBreak,
            DEVICE_TYPE_ACThreeWindingTransformer,
            DEVICE_TYPE_ACACConverter,
            DEVICE_TYPE_DCDCConverter,
            DEVICE_TYPE_DCACConverter,
        ):
            rows = device_codes == code
            with self.subTest(device_code=code):
                self.assertTrue(np.any(rows))
                self.assertLess(np.max(np.abs(result.residual[rows])), 2e-8)


if __name__ == "__main__":
    unittest.main()
