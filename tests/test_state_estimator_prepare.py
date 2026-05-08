import unittest
from pathlib import Path


class StateEstimatorPrepareTest(unittest.TestCase):
    def test_ac_constructor_delegates_preparation(self):
        from secore.ac_se import ACStateEstimator

        calls = []
        original_prepare = getattr(ACStateEstimator, "prepare", None)
        network = object()
        measurements = object()

        def prepare(self, *, network=None, measurements=None, prepare_active_measurements=True):
            calls.append((network, measurements, prepare_active_measurements))
            self._prepared = True
            return self

        ACStateEstimator.prepare = prepare
        try:
            estimator = ACStateEstimator(
                e_file=Path("case.e"),
                meas_file=Path("case.meas"),
                network=network,
                measurements=measurements,
                prepare_active_measurements=False,
            )
        finally:
            if original_prepare is None:
                delattr(ACStateEstimator, "prepare")
            else:
                ACStateEstimator.prepare = original_prepare

        self.assertEqual([(network, measurements, False)], calls)
        self.assertEqual(Path("case.e"), estimator.e_file)
        self.assertEqual(Path("case.meas"), estimator.meas_file)
        self.assertTrue(estimator._prepared)

    def test_dc_constructor_delegates_preparation(self):
        from secore.dc_se import DCStateEstimator

        calls = []
        original_prepare = getattr(DCStateEstimator, "prepare", None)
        network = object()
        measurements = object()

        def prepare(self, *, network=None, measurements=None, prepare_active_measurements=True):
            calls.append((network, measurements, prepare_active_measurements))
            self._prepared = True
            return self

        DCStateEstimator.prepare = prepare
        try:
            estimator = DCStateEstimator(
                e_file=Path("case.e"),
                meas_file=Path("case.meas"),
                network=network,
                measurements=measurements,
                prepare_active_measurements=False,
            )
        finally:
            if original_prepare is None:
                delattr(DCStateEstimator, "prepare")
            else:
                DCStateEstimator.prepare = original_prepare

        self.assertEqual([(network, measurements, False)], calls)
        self.assertEqual(Path("case.e"), estimator.e_file)
        self.assertEqual(Path("case.meas"), estimator.meas_file)
        self.assertTrue(estimator._prepared)

    def test_hybrid_constructor_delegates_preparation(self):
        from secore.hybrid_se import HybridStateEstimator

        calls = []
        original_prepare = getattr(HybridStateEstimator, "prepare", None)

        def prepare(self):
            calls.append(True)
            self._prepared = True
            return self

        HybridStateEstimator.prepare = prepare
        try:
            estimator = HybridStateEstimator(e_file=Path("case.e"), meas_file=Path("case.meas"))
        finally:
            if original_prepare is None:
                delattr(HybridStateEstimator, "prepare")
            else:
                HybridStateEstimator.prepare = original_prepare

        self.assertEqual([True], calls)
        self.assertEqual(Path("case.e"), estimator.e_file)
        self.assertEqual(Path("case.meas"), estimator.meas_file)
        self.assertTrue(estimator._prepared)


if __name__ == "__main__":
    unittest.main()
