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

    def test_ac_load_network_returns_ppc_namespace_without_object_topology(self):
        from model import topology as network_topology
        from secore.ac_se import ACStateEstimator

        original_apply_topology = network_topology.apply_ac_topology_arrays
        original_prepare_topology = network_topology.prepare_ac_topology

        def reject_array_object_topology(*_args, **_kwargs):
            raise AssertionError("AC SE PPC namespace load should not materialize object topology")

        def reject_object_topology(*_args, **_kwargs):
            raise AssertionError("AC SE load should not rerun object topology")

        network_topology.apply_ac_topology_arrays = reject_array_object_topology
        network_topology.prepare_ac_topology = reject_object_topology
        try:
            loader = ACStateEstimator.__new__(ACStateEstimator)
            network = loader._load_network(
                Path(__file__).resolve().parents[1] / "data" / "model" / "ac" / "ieee39.e"
            )
        finally:
            network_topology.apply_ac_topology_arrays = original_apply_topology
            network_topology.prepare_ac_topology = original_prepare_topology

        self.assertEqual("ac_ppc_v1", network.ppc["format"])
        self.assertIs(network.topology, network.ppc["_topology_arrays"])
        self.assertIs(network.base, network.ppc["base"])
        self.assertFalse(hasattr(network, "nodes"))
        self.assertFalse(hasattr(network, "generators"))

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

        fake_network = FakeNetwork()
        fake_ppc = {
            "ac": {"kind": "ac"},
            "dc": {"kind": "dc"},
            "base": {"p_base": 100.0, "u_scale": 2.0, "p_scale": 3.0, "i_scale": 4.0, "p_base_kW": 100000.0},
        }
        fake_rows = [["Base"], ["ACNode"]]
        e_file = Path("case.e")
        sentinel = object()
        original_read_rows = getattr(hybrid_se, "_read_efile_rows", sentinel)
        original_build = getattr(hybrid_se, "build_hybrid_ppc_with_topology_from_efile_rows", sentinel)
        original_build_network = getattr(hybrid_se, "_build_hybrid_se_network_from_ppc", sentinel)
        original_read_from_file = hybrid_se.HybridPowerNetwork.read_from_file

        def read_rows(path):
            calls.append(("read_rows", path))
            return fake_rows

        def build_from_rows(path, rows):
            calls.append(("build_from_rows", path, rows))
            return fake_ppc

        def build_network(ppc):
            calls.append(("build_network", ppc))
            fake_network.ppc = ppc
            fake_network.ac.ppc = ppc["ac"]
            fake_network.dc.ppc = ppc["dc"]
            base = ppc["base"]
            fake_network.p_base = float(base["p_base"])
            fake_network.u_scale = float(base["u_scale"])
            fake_network.p_scale = float(base["p_scale"])
            fake_network.i_scale = float(base["i_scale"])
            fake_network.p_base_kW = float(base["p_base_kW"])
            return fake_network

        def read_from_file_forbidden(*args, **kwargs):
            raise AssertionError("Hybrid load should reuse in-memory E rows")

        hybrid_se._read_efile_rows = read_rows
        hybrid_se.build_hybrid_ppc_with_topology_from_efile_rows = build_from_rows
        hybrid_se._build_hybrid_se_network_from_ppc = build_network
        hybrid_se.HybridPowerNetwork.read_from_file = read_from_file_forbidden
        try:
            network = HybridStateEstimator._load_network(e_file)
        finally:
            if original_read_rows is sentinel:
                delattr(hybrid_se, "_read_efile_rows")
            else:
                hybrid_se._read_efile_rows = original_read_rows
            if original_build is sentinel:
                delattr(hybrid_se, "build_hybrid_ppc_with_topology_from_efile_rows")
            else:
                hybrid_se.build_hybrid_ppc_with_topology_from_efile_rows = original_build
            if original_build_network is sentinel:
                delattr(hybrid_se, "_build_hybrid_se_network_from_ppc")
            else:
                hybrid_se._build_hybrid_se_network_from_ppc = original_build_network
            hybrid_se.HybridPowerNetwork.read_from_file = original_read_from_file

        self.assertIs(network, fake_network)
        self.assertEqual(
            [("read_rows", e_file), ("build_from_rows", e_file, fake_rows), ("build_network", fake_ppc)],
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

    def test_hybrid_load_network_uses_ppc_only_loader_without_full_hybrid_topology(self):
        import numpy as np
        import secore.hybrid_se as hybrid_se
        from model.ac_array_model import (
            BRANCH_COLS as AC_BRANCH_COLS,
            BUS_COLS as AC_BUS_COLS,
            GEN_COLS as AC_GEN_COLS,
            LOAD_COLS as AC_LOAD_COLS,
            SHUNT_COLS as AC_SHUNT_COLS,
            SWITCH_COLS as AC_SWITCH_COLS,
            TRANSFORMER_COLS as AC_TRANSFORMER_COLS,
            ZERO_BRANCH_COLS as AC_ZERO_BRANCH_COLS,
        )
        from model.dc_array_model import (
            BRANCH_COLS as DC_BRANCH_COLS,
            BUS_COLS as DC_BUS_COLS,
            DCDC_COLS,
            GEN_COLS as DC_GEN_COLS,
            LOAD_COLS as DC_LOAD_COLS,
            SWITCH_COLS as DC_SWITCH_COLS,
            ZERO_BRANCH_COLS as DC_ZERO_BRANCH_COLS,
        )
        from model.hybrid_array_model import ACAC_COLS, DCAC_COLS
        from secore.hybrid_se import HybridStateEstimator

        calls = []
        fake_rows = {"ACNode": {"rows": []}}
        e_file = Path("case.e")

        def empty(width):
            return np.zeros((0, width), dtype=np.float64)

        ac_bus = np.zeros((1, len(AC_BUS_COLS)), dtype=np.float64)
        ac_bus[0, AC_BUS_COLS["idx"]] = 1
        ac_bus[0, AC_BUS_COLS["vbase"]] = 110.0
        ac_bus[0, AC_BUS_COLS["voltage"]] = 1.0
        ac_bus[0, AC_BUS_COLS["run_stat"]] = 1
        dc_bus = np.zeros((1, len(DC_BUS_COLS)), dtype=np.float64)
        dc_bus[0, DC_BUS_COLS["idx"]] = 1
        dc_bus[0, DC_BUS_COLS["vbase"]] = 500.0
        dc_bus[0, DC_BUS_COLS["voltage"]] = 1.0
        dc_bus[0, DC_BUS_COLS["run_stat"]] = 1
        fake_ppc = {
            "base": {"p_base": 100.0, "u_scale": 2.0, "p_scale": 3.0, "i_scale": 4.0, "p_base_kW": 100000.0},
            "ac": {
                "base": {"p_base": 100.0, "u_scale": 2.0, "p_scale": 3.0, "i_scale": 4.0, "p_base_kW": 100000.0},
                "bus": ac_bus,
                "branch": empty(len(AC_BRANCH_COLS)),
                "transformer": empty(len(AC_TRANSFORMER_COLS)),
                "gen": empty(len(AC_GEN_COLS)),
                "load": empty(len(AC_LOAD_COLS)),
                "shunt": empty(len(AC_SHUNT_COLS)),
                "zero_branch": empty(len(AC_ZERO_BRANCH_COLS)),
                "switch": empty(len(AC_SWITCH_COLS)),
                "break": empty(len(AC_SWITCH_COLS)),
                "bus_name": np.asarray(["ac_1"], dtype=object),
                "branch_name": np.asarray([], dtype=object),
                "transformer_name": np.asarray([], dtype=object),
                "gen_name": np.asarray([], dtype=object),
                "load_name": np.asarray([], dtype=object),
                "shunt_name": np.asarray([], dtype=object),
                "zero_branch_name": np.asarray([], dtype=object),
                "switch_name": np.asarray([], dtype=object),
                "break_name": np.asarray([], dtype=object),
            },
            "dc": {
                "base": {"p_base": 100.0, "u_scale": 2.0, "p_scale": 3.0, "i_scale": 4.0, "p_base_kW": 100000.0},
                "bus": dc_bus,
                "branch": empty(len(DC_BRANCH_COLS)),
                "load": empty(len(DC_LOAD_COLS)),
                "gen": empty(len(DC_GEN_COLS)),
                "zero_branch": empty(len(DC_ZERO_BRANCH_COLS)),
                "switch": empty(len(DC_SWITCH_COLS)),
                "break": empty(len(DC_SWITCH_COLS)),
                "dcdc": empty(len(DCDC_COLS)),
                "bus_name": np.asarray(["dc_1"], dtype=object),
                "branch_name": np.asarray([], dtype=object),
                "load_name": np.asarray([], dtype=object),
                "gen_name": np.asarray([], dtype=object),
                "zero_branch_name": np.asarray([], dtype=object),
                "switch_name": np.asarray([], dtype=object),
                "break_name": np.asarray([], dtype=object),
                "dcdc_name": np.asarray([], dtype=object),
            },
            "dcac": empty(len(DCAC_COLS)),
            "acac": empty(len(ACAC_COLS)),
            "dcac_name": np.asarray([], dtype=object),
            "acac_name": np.asarray([], dtype=object),
        }

        original_read_rows = hybrid_se._read_efile_rows
        original_ppc_loader = getattr(hybrid_se, "build_hybrid_ppc_with_topology_from_efile_rows", None)
        original_full = getattr(hybrid_se, "build_hybrid_ppc_from_efile_rows", None)

        def read_rows(path):
            calls.append(("read_rows", path))
            return fake_rows

        def build_ppc_only(path, rows):
            calls.append(("ppc_only", path, rows))
            return fake_ppc

        def reject_full_loader(*_args, **_kwargs):
            raise AssertionError("Hybrid SE array path should not build the full HybridPowerNetwork")

        hybrid_se._read_efile_rows = read_rows
        hybrid_se.build_hybrid_ppc_with_topology_from_efile_rows = build_ppc_only
        hybrid_se.build_hybrid_ppc_from_efile_rows = reject_full_loader
        try:
            network = HybridStateEstimator._load_network(e_file)
        finally:
            hybrid_se._read_efile_rows = original_read_rows
            if original_ppc_loader is None:
                delattr(hybrid_se, "build_hybrid_ppc_with_topology_from_efile_rows")
            else:
                hybrid_se.build_hybrid_ppc_with_topology_from_efile_rows = original_ppc_loader
            if original_full is None:
                delattr(hybrid_se, "build_hybrid_ppc_from_efile_rows")
            else:
                hybrid_se.build_hybrid_ppc_from_efile_rows = original_full

        self.assertEqual([("read_rows", e_file), ("ppc_only", e_file, fake_rows)], calls)
        self.assertIs(network.ppc, fake_ppc)
        self.assertIs(network.ac.ppc, fake_ppc["ac"])
        self.assertIs(network.dc.ppc, fake_ppc["dc"])
        self.assertTrue(getattr(network, "_se_lightweight", False))
        self.assertTrue(getattr(network.ac, "_se_lightweight", False))
        self.assertTrue(getattr(network.dc, "_se_lightweight", False))

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
            return fake_ac_network, {"base": {"p_base": 100.0, "u_scale": 1.0, "p_scale": 0.001, "i_scale": 1.0, "p_base_kW": 100000.0}}

        def build_dc(model, **kwargs):
            self.assertTrue(getattr(model, "_named_units_normalized", False))
            self.assertTrue(kwargs.get("units_already_normalized"))
            return fake_dc_network, {"base": {"p_base": 100.0, "u_scale": 1.0, "p_scale": 0.001, "i_scale": 1.0, "p_base_kW": 100000.0}}

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
            "base": {"p_base": 100.0, "u_scale": 1.0, "p_scale": 0.001, "i_scale": 1.0, "p_base_kW": 100000.0},
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
