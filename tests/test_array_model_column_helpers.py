import unittest
from unittest.mock import patch

import numpy as np


class ArrayModelColumnHelperTest(unittest.TestCase):
    def test_ac_rows_for_returns_reusable_column_view(self):
        import model.ac_array_model as ac_array_model

        data = {
            "ACNode": {
                "header_list": ["idx", "name", "vbase", "voltage", "run_stat"],
                "rows": [
                    ["1", "n1", "110", "110", "1"],
                    ["2", "", "220", "", ""],
                ],
            }
        }

        columns, rows = ac_array_model._rows_for(data, "ACNode")

        self.assertTrue(hasattr(rows, "raw_column"))
        np.testing.assert_allclose(ac_array_model._int_column(rows, columns, "idx"), np.array([1.0, 2.0]))
        np.testing.assert_allclose(ac_array_model._float_column(rows, columns, "voltage", 1.0), np.array([110.0, 1.0]))
        np.testing.assert_array_equal(
            ac_array_model._names_from_rows(rows, columns, "bus", np.array([1.0, 2.0])),
            np.array(["n1", "bus_2"], dtype=object),
        )

    def test_ac_current_scale_provider_is_lazy(self):
        import model.ac_array_model as ac_array_model

        rows = [["10.0"], ["20.0"]]
        node_values = np.array([1.0, 2.0])
        out = np.zeros((2, 1), dtype=np.float64)
        calls = []

        def scale_provider():
            calls.append("called")
            return {1: 10.0, 2: 20.0}

        ac_array_model._assign_current_if_present(
            out,
            0,
            rows,
            {},
            "current",
            node_values,
            scale_provider,
        )
        self.assertEqual([], calls)
        np.testing.assert_allclose(out[:, 0], np.array([0.0, 0.0]))

        ac_array_model._assign_current_if_present(
            out,
            0,
            rows,
            {"current": 0},
            "current",
            node_values,
            scale_provider,
        )
        self.assertEqual(["called"], calls)
        np.testing.assert_allclose(out[:, 0], np.array([1.0, 1.0]))

    def test_ppc_column_helpers_do_not_use_numpy_fromiter(self):
        import model.ac_array_model as ac_array_model
        import model.dc_array_model as dc_array_model
        import model.hybrid_array_model as hybrid_array_model

        rows = [
            ["1", "n1", "V", ""],
            ["2", "n2", "P", "7.5"],
        ]
        columns = {"idx": 0, "name": 1, "control_type": 2, "value": 3}

        for module in (ac_array_model, dc_array_model, hybrid_array_model):
            with self.subTest(module=module.__name__):
                with patch.object(np, "fromiter", side_effect=AssertionError("fromiter path should be avoided")):
                    np.testing.assert_allclose(module._float_column(rows, columns, "value", 1.25), np.array([1.25, 7.5]))
                    np.testing.assert_allclose(module._int_column(rows, columns, "idx"), np.array([1.0, 2.0]))
                    np.testing.assert_allclose(
                        module._code_column(rows, columns, "control_type", {"V": 3, "P": 5}, "P"),
                        np.array([3.0, 5.0]),
                    )

    def test_ac_dc_ppc_base_contract_is_named_and_dc_hides_node_pos(self):
        import model.ac_array_model as ac_array_model
        import model.dc_array_model as dc_array_model

        base_table = {
            "header_list": ["p_base", "u_unit", "p_unit", "i_unit"],
            "rows": [["100000", "V", "W", "kA"]],
        }
        ac_ppc = ac_array_model.build_ac_ppc_from_efile_rows(
            "inline_ac.e",
            {
                "PowerBase": base_table,
                "ACNode": {
                    "header_list": ["idx", "name", "vbase", "voltage", "angle", "run_stat"],
                    "rows": [["1", "ac1", "110", "110", "0", "1"]],
                },
            },
        )
        dc_ppc = dc_array_model.build_dc_ppc_from_efile_rows(
            "inline_dc.e",
            {
                "PowerBase": base_table,
                "DCNode": {
                    "header_list": ["idx", "name", "vbase", "voltage", "run_stat"],
                    "rows": [["1", "dc1", "500", "500", "1"]],
                },
            },
        )

        expected_base = {
            "p_base": 100000.0,
            "u_scale": 1000.0,
            "p_scale": 1000.0,
            "i_scale": 1.0,
            "p_base_kW": 100.0,
        }
        self.assertEqual(expected_base, ac_ppc["base"])
        self.assertEqual(expected_base, dc_ppc["base"])
        self.assertNotIn("node_pos", dc_ppc)


if __name__ == "__main__":
    unittest.main()
