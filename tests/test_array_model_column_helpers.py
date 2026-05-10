import unittest
from unittest.mock import patch

import numpy as np


class ArrayModelColumnHelperTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
