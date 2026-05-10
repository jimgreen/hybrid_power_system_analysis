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

    def test_ac_load_network_uses_ppc_topology_array_applier(self):
        import secore.ac_se as ac_se
        from secore.ac_se import ACStateEstimator

        calls = []
        original_apply_topology = ac_se.network_topology.apply_ac_topology_arrays
        original_prepare_topology = ac_se.network_topology.prepare_ac_topology

        def apply_topology(network, topology, **kwargs):
            calls.append((network, topology, kwargs))
            return original_apply_topology(network, topology, **kwargs)

        def reject_object_topology(*_args, **_kwargs):
            raise AssertionError("AC SE load should not rerun object topology")

        ac_se.network_topology.apply_ac_topology_arrays = apply_topology
        ac_se.network_topology.prepare_ac_topology = reject_object_topology
        try:
            loader = ACStateEstimator.__new__(ACStateEstimator)
            network = loader._load_network(
                Path(__file__).resolve().parents[1] / "data" / "model" / "ac" / "ieee39.e"
            )
        finally:
            ac_se.network_topology.apply_ac_topology_arrays = original_apply_topology
            ac_se.network_topology.prepare_ac_topology = original_prepare_topology

        self.assertEqual(1, len(calls))
        self.assertIs(network, calls[0][0])
        self.assertIs(network._topology_arrays, calls[0][1])
        self.assertTrue(calls[0][2].get("compact"))

    def test_dc_load_network_uses_ppc_topology_array_applier(self):
        import secore.dc_se as dc_se
        from secore.dc_se import DCStateEstimator

        calls = []
        original_apply_topology = dc_se.network_topology.apply_dc_topology_arrays
        original_prepare_topology = dc_se.network_topology.prepare_dc_topology

        def apply_topology(network, topology, **kwargs):
            calls.append((network, topology, kwargs))
            return original_apply_topology(network, topology, **kwargs)

        def reject_object_topology(*_args, **_kwargs):
            raise AssertionError("DC SE load should not rerun object topology")

        dc_se.network_topology.apply_dc_topology_arrays = apply_topology
        dc_se.network_topology.prepare_dc_topology = reject_object_topology
        try:
            network = DCStateEstimator._load_network(
                Path(__file__).resolve().parents[1] / "data" / "model" / "dc" / "dc_net_30.e"
            )
        finally:
            dc_se.network_topology.apply_dc_topology_arrays = original_apply_topology
            dc_se.network_topology.prepare_dc_topology = original_prepare_topology

        self.assertEqual(1, len(calls))
        self.assertIs(network, calls[0][0])
        self.assertIs(network._topology_arrays, calls[0][1])
        self.assertTrue(calls[0][2].get("compact"))

    def test_hybrid_load_network_uses_in_memory_efile_rows(self):
        import secore.hybrid_se as hybrid_se
        from secore.hybrid_se import HybridStateEstimator

        calls = []

        class FakeSide:
            pass

        class FakeModel:
            pass

        class FakeNetwork:
            def __init__(self):
                self.ac = FakeSide()
                self.dc = FakeSide()

            def _build_hybrid_topo(self):
                calls.append(("build_hybrid_topo",))

        fake_network = FakeNetwork()
        fake_ppc = {
            "ac": {"kind": "ac"},
            "dc": {"kind": "dc"},
            "base": [100.0, 2.0, 3.0, 4.0, 100000.0],
        }
        fake_rows = [["Base"], ["ACNode"]]
        e_file = Path("case.e")
        sentinel = object()
        original_read_rows = getattr(hybrid_se, "_read_efile_rows", sentinel)
        original_build = getattr(hybrid_se, "build_hybrid_ppc_from_efile_rows", sentinel)
        original_read_from_file = hybrid_se.HybridPowerNetwork.read_from_file

        def read_rows(path):
            calls.append(("read_rows", path))
            return fake_rows

        def build_from_rows(path, rows):
            calls.append(("build_from_rows", path, rows))
            return fake_network, fake_ppc

        def read_from_file_forbidden(*args, **kwargs):
            raise AssertionError("Hybrid load should reuse in-memory E rows")

        hybrid_se._read_efile_rows = read_rows
        hybrid_se.build_hybrid_ppc_from_efile_rows = build_from_rows
        hybrid_se.HybridPowerNetwork.read_from_file = read_from_file_forbidden
        try:
            network = HybridStateEstimator._load_network(e_file)
        finally:
            if original_read_rows is sentinel:
                delattr(hybrid_se, "_read_efile_rows")
            else:
                hybrid_se._read_efile_rows = original_read_rows
            if original_build is sentinel:
                delattr(hybrid_se, "build_hybrid_ppc_from_efile_rows")
            else:
                hybrid_se.build_hybrid_ppc_from_efile_rows = original_build
            hybrid_se.HybridPowerNetwork.read_from_file = original_read_from_file

        self.assertIs(network, fake_network)
        self.assertEqual(
            [("read_rows", e_file), ("build_from_rows", e_file, fake_rows), ("build_hybrid_topo",)],
            calls,
        )
        self.assertIs(network.ppc, fake_ppc)
        self.assertIs(network.ac.ppc, fake_ppc["ac"])
        self.assertIs(network.dc.ppc, fake_ppc["dc"])
        self.assertEqual(100.0, network.p_base)
        self.assertEqual(2.0, network.u_scale)
        self.assertEqual(3.0, network.p_scale)
        self.assertEqual(4.0, network.i_scale)
        self.assertEqual(100000.0, network.p_base_kW)

    def test_hybrid_ppc_model_build_normalizes_named_units_once(self):
        import model.hybrid_array_model as hybrid_array_model

        calls = []
        sentinel = object()
        original_normalize = getattr(hybrid_array_model, "normalize_model_named_units", sentinel)
        original_build_ac = hybrid_array_model._build_ac_ppc_from_model
        original_build_dc = hybrid_array_model._build_dc_ppc_from_model
        original_build_model = hybrid_array_model.build_hybrid_model_from_ppc

        class FakeSide:
            pass

        class FakeModel:
            pass

        fake_ac_network = FakeSide()
        fake_dc_network = FakeSide()
        fake_hybrid_network = FakeSide()
        fake_hybrid_network.ac = fake_ac_network
        fake_hybrid_network.dc = fake_dc_network

        def normalize(model):
            calls.append(model)
            model.p_base = 100.0
            model.p_base_kW = 100000.0
            model.u_scale = 1.0
            model.p_scale = 0.001
            model.i_scale = 1.0
            model._named_units_normalized = True
            return 100.0

        def build_ac(model, **kwargs):
            self.assertTrue(getattr(model, "_named_units_normalized", False))
            self.assertTrue(kwargs.get("units_already_normalized"))
            return fake_ac_network, {"base": [100.0, 1.0, 0.001, 1.0, 100000.0]}

        def build_dc(model, **kwargs):
            self.assertTrue(getattr(model, "_named_units_normalized", False))
            self.assertTrue(kwargs.get("units_already_normalized"))
            return fake_dc_network, {"base": {"p_base": 100.0}}

        def build_model(ppc):
            return fake_hybrid_network

        hybrid_array_model.normalize_model_named_units = normalize
        hybrid_array_model._build_ac_ppc_from_model = build_ac
        hybrid_array_model._build_dc_ppc_from_model = build_dc
        hybrid_array_model.build_hybrid_model_from_ppc = build_model
        try:
            network, _ppc = hybrid_array_model.build_hybrid_ppc_from_model(Path("case.e"), FakeModel())
        finally:
            if original_normalize is sentinel:
                delattr(hybrid_array_model, "normalize_model_named_units")
            else:
                hybrid_array_model.normalize_model_named_units = original_normalize
            hybrid_array_model._build_ac_ppc_from_model = original_build_ac
            hybrid_array_model._build_dc_ppc_from_model = original_build_dc
            hybrid_array_model.build_hybrid_model_from_ppc = original_build_model

        self.assertIs(network, fake_hybrid_network)
        self.assertEqual(1, len(calls))

    def test_hybrid_model_from_ppc_reuses_converter_objects_when_available(self):
        import numpy as np
        from model.hybrid_array_model import ACAC_COLS, DCAC_COLS, build_hybrid_model_from_ppc

        class FakeSide:
            pass

        class FakeConverter:
            pass

        ac = FakeSide()
        dc = FakeSide()
        for side in (ac, dc):
            side.nodes = []
            side.branches = []
            side.loads = []
            side.generators = []
            side.zero_branches = []
            side.switches = []
            side.breakers = []
        ac.transformers = []
        ac.shunt_compensators = []
        dc.dcdc_converters = []
        dcac = FakeConverter()
        acac = FakeConverter()
        ppc = {
            "base": [100.0, 1.0, 0.001, 1.0, 100000.0],
            "ac": {},
            "dc": {},
            "ac_network": ac,
            "dc_network": dc,
            "dcac": np.zeros((1, len(DCAC_COLS)), dtype=np.float64),
            "acac": np.zeros((1, len(ACAC_COLS)), dtype=np.float64),
            "dcac_name": np.array(["dcac_1"], dtype=object),
            "acac_name": np.array(["acac_1"], dtype=object),
            "dcac_objects": [dcac],
            "acac_objects": [acac],
        }

        network = build_hybrid_model_from_ppc(ppc)

        self.assertIs(network.dcac_converters[0], dcac)
        self.assertIs(network.acac_converters[0], acac)


if __name__ == "__main__":
    unittest.main()
