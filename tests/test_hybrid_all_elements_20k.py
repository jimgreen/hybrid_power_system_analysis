import sys
import unittest
from collections import Counter
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


CASE = ROOT / "data" / "model" / "hybrid" / "hybrid_all_elements_20k.e"
MEASUREMENTS = ROOT / "data" / "meas" / "hybrid" / "hybrid_all_elements_20k.meas"


class HybridAllElements20KTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case_book = EBook(CASE)
        cls.measurement_rows = EBook(MEASUREMENTS).data["Measurement"].data

    def test_every_required_element_family_has_at_least_forty_devices(self):
        minimum_forty = (
            "ACBranch",
            "DCBranch",
            "ACZeroBranch",
            "DCZeroBranch",
            "ACSwitch",
            "DCSwitch",
            "ACBreak",
            "DCBreak",
            "ACShuntCompensator",
            "ACTransformer",
            "ACThreeWindingTransformer",
            "ACGenerator",
            "DCGenerator",
            "ACLoad",
            "DCLoad",
            "ACACConverter",
            "DCACConverter",
            "DCDCConverter",
        )
        for block_name in minimum_forty:
            with self.subTest(block=block_name):
                self.assertGreaterEqual(len(self.case_book.data[block_name].data), 40)
        self.assertEqual(10000, len(self.case_book.data["ACNode"].data))
        self.assertEqual(10000, len(self.case_book.data["DCNode"].data))

        labels = Counter(row["dev_type"] for row in self.case_book.data["DCACConverter"].data)
        self.assertEqual(Counter({"ACDCConverter": 40, "DCACConverter": 40}), labels)

    def test_spoke_ring_topology_and_control_modes_are_complete(self):
        blocks = self.case_book.data
        ac_branch_names = {row["name"] for row in blocks["ACBranch"].data}
        dc_branch_names = {row["name"] for row in blocks["DCBranch"].data}
        self.assertTrue(any(name.startswith("ac20k_spoke_") for name in ac_branch_names))
        self.assertTrue(any(name.startswith("ac20k_ring_") for name in ac_branch_names))
        self.assertTrue(any(name.startswith("dc20k_spoke_") for name in dc_branch_names))
        self.assertTrue(any(name.startswith("dc20k_ring_") for name in dc_branch_names))

        self.assertEqual(
            {"PQ", "PV", "V", "P", "PH"},
            {row["control_type"] for row in blocks["ACGenerator"].data},
        )
        self.assertEqual(
            {"P", "V", "I"},
            {row["control_type"] for row in blocks["DCGenerator"].data},
        )
        self.assertEqual(
            {"Q", "V", "Z", "B"},
            {row["control_type"] for row in blocks["ACShuntCompensator"].data},
        )
        for block_name in ("ACSwitch", "DCSwitch", "ACBreak", "DCBreak"):
            states = {
                int(row["status"])
                for row in blocks[block_name].data
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

    def test_measurement_contract_excludes_switches_and_includes_special_devices(self):
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

    def test_hybrid_lf_converges_from_flat_start(self):
        network = _read_lf_network_from_file(CASE)
        calc = HybridPowerFlowCalc(network, result_mode="array", verbose=False)

        calc.prepare()
        self.assertEqual(calc.ac_size + calc.dc_size + 3 * calc.N_dcac, calc.total_vars)
        self.assertEqual(calc.total_vars, calc.total_eq)
        jacobian = calc.get_jacobi(calc.x).tocsr()
        self.assertEqual((calc.total_eq, calc.total_vars), jacobian.shape)
        self.assertGreater(jacobian[: calc.ac_eq, calc.dcac_start :].nnz, 0)
        self.assertGreater(jacobian[calc.ac_eq : calc.dcac_eq_start, calc.dcac_start :].nnz, 0)
        self.assertGreater(jacobian[calc.dcac_eq_start :, : calc.ac_size].nnz, 0)
        self.assertGreater(
            jacobian[
                calc.dcac_eq_start :,
                calc.ac_size : calc.ac_size + calc.dc_size,
            ].nnz,
            0,
        )

        def reject_independent_subsolve(*_args, **_kwargs):
            raise AssertionError("hybrid LF must solve one global Newton system")

        calc.ac_calc._run_newton_raphson = reject_independent_subsolve
        calc.dc_calc._run_newton_raphson = reject_independent_subsolve

        self.assertEqual(0, calc.run())
        self.assertTrue(calc.converged)
        self.assertLessEqual(calc.iterations, 15)
        self.assertLess(np.max(np.abs(calc.get_f(calc.x))), 2e-9)
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
                    self.assertGreaterEqual(values.shape[0], 40)
                    self.assertTrue(np.all(np.isfinite(values)))

    def test_hybrid_se_is_observable_and_recovers_special_device_measurements(self):
        estimator = HybridStateEstimator(
            CASE,
            MEASUREMENTS,
            flat_start=True,
            max_iter=50,
        )
        self.assertIsNone(estimator._delegate())
        self.assertEqual(
            estimator.ac_n_state + estimator.dc_n_state + estimator.hybrid_n_state,
            estimator.n_state,
        )

        def reject_independent_subestimate(*_args, **_kwargs):
            raise AssertionError("hybrid SE must solve one global WLS problem")

        estimator._ac_sub_estimator.estimate = reject_independent_subestimate
        estimator._dc_sub_estimator.estimate = reject_independent_subestimate
        observability = estimator.observability_analysis()
        result = estimator.estimate(
            observability=observability,
            final_diagnostics=False,
        )

        self.assertTrue(observability.observable)
        self.assertEqual(observability.rank, observability.state_count)
        self.assertEqual(estimator.n_state, result.x.size)
        self.assertTrue(result.converged)
        self.assertLessEqual(result.iterations, 15)
        self.assertLess(result.objective, 1e-12)
        self.assertLess(result.residual_inf, 1e-8)
        self.assertEqual(0, len(estimator.bad_items))

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
                self.assertLess(np.max(np.abs(result.residual[rows])), 1e-8)


if __name__ == "__main__":
    unittest.main()
