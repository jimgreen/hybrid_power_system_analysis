import contextlib
import io
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


class StateEstimatorIterationOutputTest(unittest.TestCase):
    def test_ac_estimator_prints_iteration_process_when_verbose(self):
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
            max_iter=5,
        )

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            result = estimator.estimate(verbose=True)

        text = out.getvalue()
        self.assertIn("Iteration process:", text)
        self.assertIn("objective", text)
        self.assertIn("max_dx", text)
        self.assertIn("norm_res", text)
        self.assertRegex(text, r"\n\s*1\s+")
        self.assertTrue(result.converged)

    def test_dc_estimator_prints_iteration_process_when_verbose(self):
        from secore.dc_se import DCStateEstimator

        estimator = DCStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas",
            max_iter=5,
        )

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            result = estimator.estimate(verbose=True)

        text = out.getvalue()
        self.assertIn("Iteration process:", text)
        self.assertIn("objective", text)
        self.assertIn("max_dx", text)
        self.assertIn("norm_res", text)
        self.assertRegex(text, r"\n\s*1\s+")
        self.assertTrue(result.converged)


if __name__ == "__main__":
    unittest.main()
