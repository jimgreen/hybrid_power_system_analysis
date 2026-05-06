import sys
import math
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "model"))
sys.path.insert(0, str(ROOT_DIR / "lfcore"))


MEAS_FILES = sorted((ROOT_DIR / "data").glob("**/*.meas"))


class NamedUnitFileTest(unittest.TestCase):
    def test_qinling_e_file_uses_named_values_but_network_uses_pu(self):
        from efile_read import EBook
        from hybrid_flow import HybridPowerNetwork

        case_path = ROOT_DIR / "data" / "hybrid" / "qinling.e"
        raw = EBook(case_path).to_dict()

        self.assertIn("PowerBase", raw)
        power_base = raw["PowerBase"]["data"][0]
        self.assertAlmostEqual(float(power_base["p_base"]), 100.0)
        self.assertAlmostEqual(float(power_base["u_scale"]), 1000.0)
        self.assertAlmostEqual(float(power_base["p_scale"]), 1.0)
        self.assertAlmostEqual(float(power_base["i_scale"]), 1000.0)

        ac_node = next(row for row in raw["ACNode"]["data"] if row["name"] == "wt01_src")
        dcac = next(row for row in raw["DCACConverter"]["data"] if row["name"] == "wt01_rect")
        self.assertAlmostEqual(float(ac_node["voltage"]), 300.0)
        self.assertAlmostEqual(float(dcac["p_ac_set"]), 10.0)

        network = HybridPowerNetwork.read_from_file(case_path)
        wt01_node = next(node for node in network.ac.nodes if node.name == "wt01_src")
        wt01_conv = next(conv for conv in network.dcac_converters if conv.name == "wt01_rect")
        self.assertAlmostEqual(wt01_node.voltage, 1.0)
        self.assertAlmostEqual(wt01_conv.p_ac_set, 0.1)

    def test_state_estimation_measurements_are_named_in_file_and_pu_in_estimator(self):
        from efile_read import EBook
        from secore.hybrid_se import HybridStateEstimator

        meas_path = ROOT_DIR / "data" / "hybrid" / "qinling.meas"
        raw = EBook(meas_path).to_dict()
        self.assertNotIn("PowerBase", raw)
        node_meas = next(row for row in raw["Measurement"]["data"] if row["name"] == "vm_wt01_src")
        conv_meas = next(row for row in raw["Measurement"]["data"] if row["name"] == "pac_wt01_rect")

        self.assertAlmostEqual(float(node_meas["value"]), 300.0)
        self.assertAlmostEqual(float(conv_meas["value"]), 10.0, places=6)

        estimator = HybridStateEstimator(
            e_file=ROOT_DIR / "data" / "hybrid" / "qinling.e",
            meas_file=meas_path,
        )
        node_est = next(meas for meas in estimator.active_measurements if meas.name == "vm_wt01_src")
        conv_est = next(meas for meas in estimator.active_measurements if meas.name == "pac_wt01_rect")

        self.assertAlmostEqual(node_est.value, 1.0)
        self.assertAlmostEqual(conv_est.value, 0.1)

    def test_measurement_files_use_split_device_columns_and_aligned_spaces(self):
        from efile_read import EBook

        self.assertTrue(MEAS_FILES)
        for meas_path in MEAS_FILES:
            text = meas_path.read_text(encoding="utf-8")
            self.assertNotIn("\t", text, meas_path)
            raw = EBook(meas_path).to_dict()
            header = raw["Measurement"]["header_list"]
            self.assertEqual(["idx", "name", "dev_type", "dev_name", "meas_type", "weight", "valid", "value"], header)
            for row in raw["Measurement"]["data"]:
                self.assertNotIn("device", row)
                self.assertTrue(row["dev_type"])
                self.assertTrue(row["dev_name"])
                weight = float(row["weight"])
                self.assertGreaterEqual(weight, 0.1)
                self.assertLessEqual(weight, 10.0)

    def test_ac_node_angle_is_degrees_in_file_and_radians_in_network(self):
        from efile_read import EBook
        from ac_model import ACPowerNetwork

        case_path = ROOT_DIR / "data" / "ac" / "ieee39.e"
        raw = EBook(case_path).to_dict()
        raw_node = next(row for row in raw["ACNode"]["data"] if row["name"] == "bus_1")
        raw_angle_deg = float(raw_node["angle"])

        self.assertGreater(abs(raw_angle_deg), 1.0)

        network = ACPowerNetwork()
        network.read_from_file(case_path)
        node = next(item for item in network.nodes if item.name == "bus_1")

        self.assertAlmostEqual(node.angle, math.radians(raw_angle_deg), places=10)

    def test_ac_state_estimator_angle_measurement_is_degrees_in_file(self):
        from efile_read import EBook
        from secore.ac_se import ACStateEstimator

        case_path = ROOT_DIR / "data" / "ac" / "ieee39.e"
        raw = EBook(case_path).to_dict()
        raw_node = next(row for row in raw["ACNode"]["data"] if row["name"] == "bus_1")
        raw_angle_deg = float(raw_node["angle"])

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_path = Path(tmp_dir) / "angle.meas"
            meas_path.write_text(
                "\n".join(
                    [
                        "<Measurement>",
                        "@ idx name dev_type dev_name meas_type weight valid value",
                        f"# 1 va_bus_1 ACNode bus_1 ANGLE 1.0 1 {raw_angle_deg}",
                        "</Measurement>",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            estimator = ACStateEstimator(e_file=case_path, meas_file=meas_path)
            measurement = next(meas for meas in estimator.measurements if meas.name == "va_bus_1")

            self.assertFalse(measurement.valid)
            self.assertAlmostEqual(measurement.value, raw_angle_deg, places=10)
            self.assertFalse(
                any(meas.meas_type in ("ANGLE", "THETA") for meas in estimator.active_measurements)
            )

    def test_hybrid_state_estimator_angle_measurement_is_degrees_in_file(self):
        from secore.hybrid_se import HybridStateEstimator

        case_path = ROOT_DIR / "data" / "hybrid" / "qinling.e"
        raw_angle_deg = 30.0

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_path = Path(tmp_dir) / "hybrid_angle.meas"
            meas_path.write_text(
                "\n".join(
                    [
                        "<Measurement>",
                        "@ idx name dev_type dev_name meas_type weight valid value",
                        f"# 1 va_ac_bus ACNode ac_bus ANGLE 1.0 1 {raw_angle_deg}",
                        "</Measurement>",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            estimator = HybridStateEstimator(e_file=case_path, meas_file=meas_path)
            measurement = next(meas for meas in estimator.measurements if meas.name == "va_ac_bus")

            self.assertFalse(measurement.valid)
            self.assertAlmostEqual(measurement.value, raw_angle_deg, places=10)
            self.assertFalse(
                any(meas.meas_type in ("ANGLE", "THETA") for meas in estimator.active_measurements)
            )

    def test_power_base_scales_file_units_before_pu_normalization(self):
        from unit_system import normalize_model_named_units

        model = SimpleNamespace(
            PowerBase=[SimpleNamespace(p_base=100000.0, u_scale=1000.0, p_scale=1000.0, i_scale=1000.0)],
            ACNode=[
                SimpleNamespace(idx=1, vbase=300.0, voltage=300.0),
                SimpleNamespace(idx=2, vbase=300.0, voltage=297.0),
            ],
            ACBranch=[
                SimpleNamespace(
                    i_node=1,
                    j_node=2,
                    i_p=10000.0,
                    i_q=5000.0,
                    j_p=-9900.0,
                    j_q=-4900.0,
                    i_c=192.4500897,
                    j_c=192.4500897,
                )
            ],
            DCACConverter=[
                SimpleNamespace(ac_node=1, dc_node=0, p_ac_set=10000.0, q_ac_set=0.0, dc_p=0.0, ac_p=0.0, ac_q=0.0)
            ],
        )

        normalize_model_named_units(model)

        self.assertAlmostEqual(model.ACNode[0].vbase, 0.3)
        self.assertAlmostEqual(model.ACNode[0].voltage, 1.0)
        self.assertAlmostEqual(model.ACNode[1].voltage, 0.99)
        self.assertAlmostEqual(model.ACBranch[0].i_p, 0.1)
        self.assertAlmostEqual(model.ACBranch[0].i_q, 0.05)
        self.assertAlmostEqual(model.ACBranch[0].i_c, 1.0, places=7)
        self.assertAlmostEqual(model.DCACConverter[0].p_ac_set, 0.1)


if __name__ == "__main__":
    unittest.main()
