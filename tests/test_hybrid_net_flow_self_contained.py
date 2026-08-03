import sys
import tempfile
import unittest
import contextlib
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "hybrid_power_system_analysis"))
sys.path.insert(0, str(ROOT / "src" / "hybrid_power_system_analysis" / "model"))
sys.path.insert(0, str(ROOT / "src" / "hybrid_power_system_analysis" / "lfcore"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT))


def _ensure_block_column(text, block_name, column_name, default_value):
    start = text.index(f"<{block_name}>")
    end = text.index(f"</{block_name}>", start)
    block = text[start:end]
    lines = block.splitlines()
    header_idx = next(i for i, line in enumerate(lines) if line.startswith("@"))
    header = lines[header_idx].split()
    if column_name in header[1:]:
        return text
    lines[header_idx] = lines[header_idx] + f" {column_name}"
    for i, line in enumerate(lines):
        if line.startswith("#"):
            lines[i] = line + f" {default_value}"
    return text[:start] + "\n".join(lines) + "\n" + text[end:]


def _set_block_value(text, block_name, row_idx, column_name, value):
    start = text.index(f"<{block_name}>")
    end = text.index(f"</{block_name}>", start)
    block = text[start:end]
    lines = block.splitlines()
    header = next(line for line in lines if line.startswith("@")).split()[1:]
    column_idx = header.index(column_name)
    for i, line in enumerate(lines):
        if not line.startswith("#"):
            continue
        parts = line.split()
        if parts[1] == str(row_idx):
            parts[column_idx + 1] = str(value)
            lines[i] = " ".join(parts)
            break
    else:
        raise AssertionError(f"{block_name}[{row_idx}] not found")
    return text[:start] + "\n".join(lines) + "\n" + text[end:]


def _run_hybrid_lf(file_name, **kwargs):
    import lfcore.hybrid_lf as hybrid_lf

    network = hybrid_lf._read_lf_network_from_file(file_name)
    calc = hybrid_lf.HybridPowerFlowCalc(network, **kwargs)
    calc.run()
    if calc.result_mode in {"array", "summary", "none"}:
        return calc.result
    return calc.lf_result


def _table(header, rows):
    return {"header_list": header.split(), "rows": rows}


def _independent_hybrid_rows(*, ac_run=1, dc_run=1, stale=0.0, include_converter=False):
    rows = {
        "Model": _table(
            "path name p_base u_unit p_unit i_unit",
            [["test", "independent", 100, "V", "kW", "A"]],
        ),
        "ACNode": _table(
            "idx name vbase voltage angle run_stat",
            [[1, "ac-bus", 380, stale, stale, ac_run]],
        ),
        "ACGenerator": _table(
            "idx name node control_type p_set q_set v_set alpha run_stat p q current",
            [[1, "ac-source", 1, "PH", 0, 0, 380, 1, 1, stale, stale, stale]],
        ),
        "ACLoad": _table(
            "idx name node pbase pv0 pv1 pv2 qbase qv0 qv1 qv2 run_stat p q current",
            [[1, "ac-load", 1, 10, 1, 0, 0, 2, 1, 0, 0, 1, stale, stale, stale]],
        ),
        "DCNode": _table(
            "idx name vbase voltage run_stat",
            [[1, "dc-bus", 750, stale, dc_run]],
        ),
        "DCGenerator": _table(
            "idx name node control_type v_set p_set i_set run_stat p current",
            [[1, "dc-source", 1, "V", 750, 0, 0, 1, stale, stale]],
        ),
        "DCLoad": _table(
            "idx name node pbase pv0 pv1 pv2 run_stat p current",
            [[1, "dc-load", 1, 10, 1, 0, 0, 1, stale, stale]],
        ),
    }
    if include_converter:
        rows["DCACConverter"] = _table(
            "idx name ac_node dc_node r1 r2 ac_control_type dc_control_type "
            "p_ac_set q_ac_set v_ac_set v_dc_set run_stat dc_p ac_p ac_q dc_i ac_i",
            [
                [
                    1,
                    "inactive-converter",
                    1,
                    1,
                    0,
                    0,
                    "PQ",
                    "NONE",
                    0,
                    0,
                    380,
                    750,
                    1,
                    stale,
                    stale,
                    stale,
                    stale,
                    stale,
                ]
            ],
        )
    return rows


def _retire_dc_ppc_with_stale_outputs(ppc, stale=99.0):
    from dc_array_model import (
        BRANCH_COLS,
        BREAK_COLS,
        BUS_COLS,
        DCDC_COLS,
        GEN_COLS,
        LOAD_COLS,
        SWITCH_COLS,
        ZERO_BRANCH_COLS,
    )

    dynamic_columns = {
        "bus": (BUS_COLS["voltage"],),
        "branch": tuple(BRANCH_COLS[name] for name in ("i_p", "j_p", "current")),
        "load": tuple(LOAD_COLS[name] for name in ("p", "current")),
        "gen": tuple(GEN_COLS[name] for name in ("p", "current")),
        "zero_branch": tuple(ZERO_BRANCH_COLS[name] for name in ("p", "current")),
        "switch": tuple(SWITCH_COLS[name] for name in ("p", "current")),
        "break": tuple(BREAK_COLS[name] for name in ("p", "current")),
        "dcdc": tuple(DCDC_COLS[name] for name in ("i_p", "j_p", "i_c", "j_c")),
    }
    ppc["bus"][:, BUS_COLS["run_stat"]] = 0.0
    for key, columns in dynamic_columns.items():
        table = ppc.get(key)
        if table is not None and table.size:
            table[:, list(columns)] = stale
    return {key: ppc[key].copy() for key in dynamic_columns}, dynamic_columns


def _name(device, value):
    device.name = value
    return device


def _live_ac_object_network():
    from ac_model import ACPowerNetwork

    network = ACPowerNetwork()
    _name(network.add_node(10, 1.0, voltage=1.0, angle=0.0), "ac-live-source")
    _name(network.add_node(20, 1.0, voltage=1.0, angle=0.0), "ac-live-load")
    _name(network.add_branch(501, 10, 20, 0.01, 0.05, 0.0), "ac-live-line")
    _name(network.add_generator(101, 10, "PH", 0.0, 0.0, 1.0, alpha=1.0), "ac-live-gen")
    _name(
        network.add_load(301, 20, 0.1, 1.0, 0.0, 0.0, 0.02, 1.0, 0.0, 0.0),
        "ac-live-load-device",
    )
    return network


def _live_dc_object_network():
    from dc_model import DCPowerNetwork

    network = DCPowerNetwork()
    _name(network.add_node(110, 1.0, voltage=1.0), "dc-live-source")
    _name(network.add_node(220, 1.0, voltage=1.0), "dc-live-load-bus")
    _name(network.add_branch(901, 110, 220, 0.05), "dc-live-branch")
    _name(network.add_generator(701, 110, "V", 0.0, 1.0, 0.0), "dc-live-gen")
    _name(network.add_load(801, 220, 0.1, 1.0, 0.0, 0.0), "dc-live-load")
    return network


def _dead_ac_object_network(stale=99.0):
    from ac_model import ACPowerNetwork

    network = ACPowerNetwork()
    for idx in (30, 40):
        node = _name(
            network.add_node(idx, 1.0, voltage=stale, angle=stale, run_stat=0),
            f"ac-dead-node-{idx}",
        )
        node.is_alive = True
    generator = _name(
        network.add_generator(1101, 30, "PH", 0.0, 0.0, 1.0, alpha=1.0),
        "ac-dead-gen",
    )
    generator.p = generator.q = generator.current = stale
    switch = _name(network.add_switch(1901, 30, 40, status=0), "ac-dead-switch")
    switch.p = switch.q = switch.current = stale
    return network


def _dead_dc_object_network(stale=99.0):
    from dc_model import DCPowerNetwork

    network = DCPowerNetwork()
    for idx in (130, 140):
        node = _name(
            network.add_node(idx, 1.0, voltage=stale, run_stat=0),
            f"dc-dead-node-{idx}",
        )
        node.is_alive = True
    generator = _name(network.add_generator(1701, 130, "V", 0.0, 1.0, 0.0), "dc-dead-gen")
    generator.p = generator.current = stale
    switch = _name(network.add_switch(2901, 130, 140, status=0), "dc-dead-switch")
    switch.p = switch.current = stale
    return network


def _mixed_alive_dead_ac_object_network(stale=99.0):
    from ac_model import ACShuntCompensator

    network = _live_ac_object_network()
    for idx in (30, 40, 50):
        _name(
            network.add_node(idx, 1.0, voltage=stale, angle=stale),
            f"ac-dead-island-node-{idx}",
        )

    generator = _name(
        network.add_generator(1101, 30, "PQ", 0.0, 0.0, 1.0, alpha=1.0),
        "ac-dead-island-gen",
    )
    load = _name(
        network.add_load(1201, 40, 0.1, 1.0, 0.0, 0.0, 0.02, 1.0, 0.0, 0.0),
        "ac-dead-island-load",
    )
    shunt = _name(ACShuntCompensator(1301, 30, run_stat=1), "ac-dead-island-shunt")
    network.shunt_compensators.append(shunt)
    branch = _name(network.add_branch(1501, 30, 40, 0.01, 0.05, 0.0), "ac-dead-island-branch")
    transformer = _name(
        network.add_transformer(1601, 30, 40, 0.01, 0.05, 1.0, 0.0),
        "ac-dead-island-transformer",
    )
    three_winding = _name(
        network.add_three_winding_transformer(
            1701,
            30,
            40,
            50,
            0.01,
            0.05,
            0.01,
            0.05,
            0.01,
            0.05,
        ),
        "ac-dead-island-three-winding",
    )
    zero_branch = _name(network.add_zero_branch(1801, 30, 40), "ac-dead-island-zero")
    switch = _name(network.add_switch(1901, 30, 40, status=1), "ac-dead-island-switch")
    breaker = _name(network.add_break(2001, 30, 40, status=1), "ac-dead-island-breaker")

    for device in (generator, load, shunt):
        device.p = device.q = device.current = stale
    for device in (branch, transformer):
        for attr in ("i_p", "i_q", "i_c", "j_p", "j_q", "j_c"):
            setattr(device, attr, stale)
    for attr in ("i_p", "i_q", "i_c", "j_p", "j_q", "j_c", "k_p", "k_q", "k_c"):
        setattr(three_winding, attr, stale)
    for device in (zero_branch, switch, breaker):
        device.p = device.q = device.current = stale
    return network


def _pollute_dc_object_results(network, stale=99.0):
    for node in network.nodes:
        node.voltage = stale
        node.is_alive = False
    for device in (*network.generators, *network.loads):
        device.p = stale
        device.current = stale
        device.is_alive = False
    for branch in network.branches:
        branch.i_p = stale
        branch.j_p = stale
        branch.current = stale
        branch.is_alive = False


def _pollute_ac_object_results(network, stale=99.0):
    for node in network.nodes:
        node.voltage = stale
        node.angle = stale
        node.is_alive = False
    for device in (*network.generators, *network.loads):
        device.p = stale
        device.q = stale
        device.current = stale
        device.is_alive = False
    for branch in network.branches:
        branch.i_p = stale
        branch.i_q = stale
        branch.i_c = stale
        branch.j_p = stale
        branch.j_q = stale
        branch.j_c = stale
        branch.is_alive = False


def _assert_ac_objects_match_array_result(test_case, network, result):
    from ac_array_model import BRANCH_COLS, BUS_COLS, GEN_COLS, LOAD_COLS

    def rows_by_idx(key, columns):
        return {int(row[columns["idx"]]): row for row in result[key]}

    bus_by_idx = rows_by_idx("bus", BUS_COLS)
    for node in network.nodes:
        row = bus_by_idx[int(node.idx)]
        test_case.assertAlmostEqual(float(row[BUS_COLS["voltage"]]), node.voltage)
        test_case.assertAlmostEqual(float(row[BUS_COLS["angle"]]), node.angle)
        test_case.assertTrue(node.is_alive)

    for collection, key, columns, attrs in (
        (network.generators, "gen", GEN_COLS, ("p", "q", "current")),
        (network.loads, "load", LOAD_COLS, ("p", "q", "current")),
        (network.branches, "branch", BRANCH_COLS, ("i_p", "i_q", "i_c", "j_p", "j_q", "j_c")),
    ):
        row_by_idx = rows_by_idx(key, columns)
        for device in collection:
            row = row_by_idx[int(device.idx)]
            for attr in attrs:
                test_case.assertAlmostEqual(float(row[columns[attr]]), getattr(device, attr))
            test_case.assertTrue(device.is_alive)
            test_case.assertEqual(1, device.run_stat)


def _assert_dc_objects_match_array_result(test_case, network, result):
    from dc_array_model import BRANCH_COLS, BUS_COLS, GEN_COLS, LOAD_COLS

    def rows_by_idx(key, columns):
        return {int(row[columns["idx"]]): row for row in result[key]}

    bus_by_idx = rows_by_idx("bus", BUS_COLS)
    for node in network.nodes:
        row = bus_by_idx[int(node.idx)]
        test_case.assertAlmostEqual(float(row[BUS_COLS["voltage"]]), node.voltage)
        test_case.assertTrue(node.is_alive)

    for collection, key, columns, attrs in (
        (network.generators, "gen", GEN_COLS, ("p", "current")),
        (network.loads, "load", LOAD_COLS, ("p", "current")),
        (network.branches, "branch", BRANCH_COLS, ("i_p", "j_p", "current")),
    ):
        row_by_idx = rows_by_idx(key, columns)
        for device in collection:
            row = row_by_idx[int(device.idx)]
            for attr in attrs:
                test_case.assertAlmostEqual(float(row[columns[attr]]), getattr(device, attr))
            test_case.assertTrue(device.is_alive)
            test_case.assertEqual(1, device.run_stat)


class HybridNetFlowSelfContainedTest(unittest.TestCase):
    def test_ppc_skipped_snapshots_are_built_only_for_dead_sides(self):
        from unittest import mock

        import lfcore.hybrid_lf as hybrid_lf

        cases = (
            (1, 1, 0, False, False),
            (0, 1, 1, True, False),
            (1, 0, 1, False, True),
            (0, 0, 2, True, True),
        )
        for ac_run, dc_run, expected_calls, expect_ac, expect_dc in cases:
            with self.subTest(ac_run=ac_run, dc_run=dc_run):
                network = hybrid_lf._build_lf_network_from_hybrid_rows(
                    Path("skipped_snapshot_counts.e"),
                    _independent_hybrid_rows(ac_run=ac_run, dc_run=dc_run, stale=99.0),
                )
                with mock.patch.object(
                    hybrid_lf,
                    "_zero_subgrid_result",
                    wraps=hybrid_lf._zero_subgrid_result,
                ) as zero_result:
                    calc = hybrid_lf.HybridPowerFlowCalc(
                        network,
                        result_mode="array",
                        linear_solver="scipy",
                        verbose=False,
                    )

                self.assertEqual(expected_calls, zero_result.call_count)
                self.assertEqual(expect_ac, calc._skipped_ac_result is not None)
                self.assertEqual(expect_dc, calc._skipped_dc_result is not None)

    def test_single_live_full_object_side_writeback_respects_result_mode(self):
        from hybrid_model import HybridPowerNetwork
        from lfcore.hybrid_lf import HybridPowerFlowCalc

        cases = (
            (
                "ac",
                lambda: HybridPowerNetwork(_live_ac_object_network(), _dead_dc_object_network(), [], []),
                _pollute_ac_object_results,
                _assert_ac_objects_match_array_result,
                "_write_ac_ppc_result_to_network",
            ),
            (
                "dc",
                lambda: HybridPowerNetwork(_dead_ac_object_network(), _live_dc_object_network(), [], []),
                _pollute_dc_object_results,
                _assert_dc_objects_match_array_result,
                "_write_dc_ppc_result_to_network",
            ),
        )
        for side, network_factory, pollute, assert_matches, writeback_name in cases:
            for mode in ("none", "summary", "array", "full"):
                with self.subTest(side=side, mode=mode):
                    network = network_factory()
                    calc = HybridPowerFlowCalc(
                        network,
                        result_mode=mode,
                        linear_solver="scipy",
                        verbose=False,
                    )
                    active_network = getattr(network, side)
                    dead_side = "dc" if side == "ac" else "ac"
                    dead_network = getattr(network, dead_side)
                    pollute(active_network)

                    if mode != "full":
                        def reject_active_object_writeback():
                            raise AssertionError(f"{side} object writeback is full-mode only")

                        setattr(calc, writeback_name, reject_active_object_writeback)

                    self.assertEqual(0, calc.run())
                    self.assertTrue(getattr(calc, f"_single_{side}_newton_block"))
                    self.assertIsNotNone(dead_network.result)
                    self.assertTrue(all(node.voltage == 0.0 for node in dead_network.nodes))
                    self.assertTrue(all(node.run_stat == 0 for node in dead_network.nodes))
                    self.assertEqual(0, dead_network.switches[0].status)

                    if mode == "full":
                        assert_matches(self, active_network, calc.result[side])
                        self.assertIsNotNone(calc.lf_result)
                    else:
                        self.assertTrue(all(node.voltage == 99.0 for node in active_network.nodes))
                        self.assertTrue(all(not node.is_alive for node in active_network.nodes))
                        self.assertIsNone(calc.lf_result)

    def test_general_full_hybrid_writes_dc_array_results_back_to_full_objects(self):
        from hybrid_model import HybridPowerNetwork
        from lfcore.hybrid_lf import HybridPowerFlowCalc

        dc_network = _live_dc_object_network()
        network = HybridPowerNetwork(_live_ac_object_network(), dc_network, [], [])
        calc = HybridPowerFlowCalc(
            network,
            result_mode="full",
            linear_solver="scipy",
            verbose=False,
        )
        _pollute_dc_object_results(dc_network)

        self.assertEqual(0, calc.run())

        self.assertFalse(calc._single_dc_newton_block)
        self.assertIsNotNone(calc.lf_result)
        _assert_dc_objects_match_array_result(self, dc_network, calc.result["dc"])

    def test_general_full_hybrid_writes_dc_objects_when_only_dc_object_owns_ppc(self):
        from dc_array_model import build_dc_ppc_from_network
        from hybrid_model import HybridPowerNetwork
        from lfcore.hybrid_lf import HybridPowerFlowCalc

        dc_network = _live_dc_object_network()
        dc_network.ppc = build_dc_ppc_from_network(dc_network)
        network = HybridPowerNetwork(_live_ac_object_network(), dc_network, [], [])
        self.assertFalse(hasattr(network, "_dc_ppc"))
        calc = HybridPowerFlowCalc(
            network,
            result_mode="full",
            linear_solver="scipy",
            verbose=False,
        )
        _pollute_dc_object_results(dc_network)

        self.assertEqual(0, calc.run())

        self.assertIsNotNone(calc.lf_result)
        _assert_dc_objects_match_array_result(self, dc_network, calc.result["dc"])

    def test_object_skipped_ppc_builders_run_only_for_dead_sides(self):
        from unittest import mock

        from hybrid_model import HybridPowerNetwork
        import lfcore.hybrid_lf as hybrid_lf

        cases = (
            (_live_ac_object_network, _live_dc_object_network, 0, 0),
            (_dead_ac_object_network, _live_dc_object_network, 1, 0),
            (_live_ac_object_network, _dead_dc_object_network, 0, 1),
            (_dead_ac_object_network, _dead_dc_object_network, 1, 1),
        )
        for ac_factory, dc_factory, expected_ac_calls, expected_dc_calls in cases:
            with self.subTest(ac_builder=expected_ac_calls, dc_builder=expected_dc_calls):
                network = HybridPowerNetwork(ac_factory(), dc_factory(), [], [])
                with (
                    mock.patch.object(
                        hybrid_lf,
                        "build_ac_ppc_from_network",
                        wraps=hybrid_lf.build_ac_ppc_from_network,
                    ) as ac_builder,
                    mock.patch.object(
                        hybrid_lf,
                        "build_dc_ppc_from_network",
                        wraps=hybrid_lf.build_dc_ppc_from_network,
                    ) as dc_builder,
                ):
                    hybrid_lf.HybridPowerFlowCalc(
                        network,
                        result_mode="array",
                        linear_solver="scipy",
                        verbose=False,
                    )

                self.assertEqual(expected_ac_calls, ac_builder.call_count)
                self.assertEqual(expected_dc_calls, dc_builder.call_count)

    def test_no_ppc_full_object_skips_each_dead_side_and_solves_the_other(self):
        from ac_array_model import BUS_COLS as AC_BUS_COLS
        from dc_array_model import BUS_COLS as DC_BUS_COLS
        from hybrid_model import HybridPowerNetwork
        from lfcore.hybrid_lf import HybridPowerFlowCalc

        cases = (
            ("ac", _dead_ac_object_network, _live_dc_object_network),
            ("dc", _dead_dc_object_network, _live_ac_object_network),
        )
        for dead_side, dead_factory, live_factory in cases:
            with self.subTest(dead_side=dead_side):
                if dead_side == "ac":
                    network = HybridPowerNetwork(dead_factory(), live_factory(), [], [])
                else:
                    network = HybridPowerNetwork(live_factory(), dead_factory(), [], [])
                self.assertFalse(hasattr(network, "ppc"))

                calc = HybridPowerFlowCalc(network, result_mode="array", linear_solver="scipy", verbose=False)
                rc = calc.run()

                self.assertEqual(0, rc)
                self.assertTrue(calc.converged)
                self.assertIsNone(getattr(calc, f"{dead_side}_calc"))
                live_side = "dc" if dead_side == "ac" else "ac"
                self.assertIsNotNone(getattr(calc, f"{live_side}_calc"))
                skipped = calc.result[dead_side]
                self.assertIsNotNone(skipped)
                bus_cols = AC_BUS_COLS if dead_side == "ac" else DC_BUS_COLS
                dynamic_columns = [bus_cols["voltage"]]
                if dead_side == "ac":
                    dynamic_columns.append(bus_cols["angle"])
                self.assertTrue((skipped["bus"][:, dynamic_columns] == 0.0).all())
                self.assertTrue((skipped["bus"][:, bus_cols["run_stat"]] == 0.0).all())
                dead_network = getattr(network, dead_side)
                self.assertTrue(all(node.run_stat == 0 and not node.is_alive for node in dead_network.nodes))
                self.assertTrue(all(node.voltage == 0.0 for node in dead_network.nodes))
                self.assertEqual(0, dead_network.switches[0].status)

    def test_no_ppc_full_object_returns_empty_success_when_both_sides_are_dead(self):
        from ac_array_model import BUS_COLS as AC_BUS_COLS, SWITCH_COLS as AC_SWITCH_COLS
        from dc_array_model import BUS_COLS as DC_BUS_COLS, SWITCH_COLS as DC_SWITCH_COLS
        from hybrid_model import HybridPowerNetwork
        from lfcore.hybrid_lf import HybridPowerFlowCalc

        network = HybridPowerNetwork(_dead_ac_object_network(), _dead_dc_object_network(), [], [])
        self.assertFalse(hasattr(network, "ppc"))
        calc = HybridPowerFlowCalc(network, result_mode="array", linear_solver="scipy", verbose=False)

        rc = calc.run()

        self.assertEqual(0, rc)
        self.assertTrue(calc.converged)
        self.assertEqual(0, calc.iterations)
        self.assertEqual(0.0, calc.normF)
        self.assertEqual((0, 0), calc.last_jacobian_shape)
        self.assertIsNone(calc.ac_calc)
        self.assertIsNone(calc.dc_calc)
        self.assertTrue((calc.result["ac"]["bus"][:, AC_BUS_COLS["voltage"]] == 0.0).all())
        self.assertTrue((calc.result["dc"]["bus"][:, DC_BUS_COLS["voltage"]] == 0.0).all())
        self.assertEqual(0, int(calc.result["ac"]["switch"][0, AC_SWITCH_COLS["status"]]))
        self.assertEqual(0, int(calc.result["dc"]["switch"][0, DC_SWITCH_COLS["status"]]))

    def test_ac_full_object_writeback_uses_topology_alive_masks_by_sparse_idx(self):
        from ac_array_model import (
            BRANCH_COLS,
            BREAK_COLS,
            BUS_COLS,
            GEN_COLS,
            LOAD_COLS,
            SHUNT_COLS,
            SWITCH_COLS,
            THREE_WINDING_TRANSFORMER_COLS,
            TRANSFORMER_COLS,
            ZERO_BRANCH_COLS,
        )
        from dc_model import DCPowerNetwork
        from hybrid_model import HybridPowerNetwork
        from lfcore.hybrid_lf import HybridPowerFlowCalc

        mappings = (
            ("nodes", "bus", None, BUS_COLS),
            ("generators", "gen", "gen", GEN_COLS),
            ("loads", "load", "load", LOAD_COLS),
            ("shunt_compensators", "shunt", "shunt", SHUNT_COLS),
            ("branches", "branch", "branch", BRANCH_COLS),
            ("transformers", "transformer", "transformer", TRANSFORMER_COLS),
            (
                "three_winding_transformers",
                "three_winding_transformer",
                "three_winding_transformer",
                THREE_WINDING_TRANSFORMER_COLS,
            ),
            ("zero_branches", "zero_branch", "zero_branch", ZERO_BRANCH_COLS),
            ("switches", "switch", "switch", SWITCH_COLS),
            ("breakers", "break", "break", BREAK_COLS),
        )

        for mode in ("full",):
            with self.subTest(mode=mode):
                ac_network = _mixed_alive_dead_ac_object_network()
                network = HybridPowerNetwork(ac_network, DCPowerNetwork(), [], [])
                calc = HybridPowerFlowCalc(
                    network,
                    result_mode=mode,
                    linear_solver="scipy",
                    verbose=False,
                )
                for collection_name, _table_name, _topology_key, _columns in mappings:
                    for device in getattr(ac_network, collection_name):
                        device.is_alive = True

                self.assertEqual(0, calc.run())
                topology = calc.ac_calc._ppc_topology
                for collection_name, table_name, topology_key, columns in mappings:
                    rows = calc.ac_calc.ppc[table_name]
                    alive_mask = (
                        topology.node_alive_mask
                        if topology_key is None
                        else topology.devices[topology_key].alive_mask
                    )
                    expected_by_idx = {
                        int(row[columns["idx"]]): bool(alive)
                        for row, alive in zip(rows, alive_mask)
                    }
                    actual_by_idx = {
                        int(device.idx): bool(device.is_alive)
                        for device in getattr(ac_network, collection_name)
                    }
                    self.assertEqual(expected_by_idx, actual_by_idx, (mode, collection_name))

                self.assertTrue(ac_network.branch_dict[501].is_alive)
                self.assertFalse(ac_network.branch_dict[1501].is_alive)
                self.assertEqual(1, ac_network.branch_dict[1501].run_stat)
                self.assertEqual(1, ac_network.switch_dict[1901].status)
                self.assertEqual(1, ac_network.break_dict[2001].status)
                self.assertEqual(0.0, ac_network.branch_dict[1501].i_p)
                self.assertEqual(0.0, ac_network.branch_dict[1501].j_p)

    def test_hybrid_solver_skips_dead_dc_and_solves_live_ac(self):
        import numpy as np
        from ac_array_model import BUS_COLS as AC_BUS_COLS
        from dc_array_model import BUS_COLS as DC_BUS_COLS, GEN_COLS as DC_GEN_COLS
        from lfcore.hybrid_lf import HybridPowerFlowCalc, _build_lf_network_from_hybrid_rows

        network = _build_lf_network_from_hybrid_rows(
            Path("dead_dc.e"),
            _independent_hybrid_rows(ac_run=1, dc_run=0, stale=99.0),
        )
        calc = HybridPowerFlowCalc(network, result_mode="array", verbose=False)
        rc = calc.run()

        self.assertEqual(0, rc)
        self.assertTrue(calc.converged)
        self.assertIsNotNone(calc.ac_calc)
        self.assertIsNone(calc.dc_calc)
        self.assertEqual(1, calc.result["dc"]["bus"].shape[0])
        self.assertEqual(1, calc.result["dc"]["gen"].shape[0])
        self.assertTrue(np.all(calc.result["dc"]["bus"][:, DC_BUS_COLS["voltage"]] == 0.0))
        self.assertTrue(
            np.all(
                calc.result["dc"]["gen"]
                [:, [DC_GEN_COLS["p"], DC_GEN_COLS["current"]]]
                == 0.0
            )
        )
        self.assertEqual(0, int(calc.result["dc"]["bus"][0, DC_BUS_COLS["run_stat"]]))
        self.assertEqual(1, int(calc.result["dc"]["gen"][0, DC_GEN_COLS["run_stat"]]))
        self.assertGreater(float(calc.result["ac"]["bus"][0, AC_BUS_COLS["voltage"]]), 0.0)

    def test_hybrid_solver_skips_dead_ac_and_solves_live_dc(self):
        import numpy as np
        from ac_array_model import BUS_COLS as AC_BUS_COLS, GEN_COLS as AC_GEN_COLS
        from lfcore.hybrid_lf import HybridPowerFlowCalc, _build_lf_network_from_hybrid_rows

        network = _build_lf_network_from_hybrid_rows(
            Path("dead_ac.e"),
            _independent_hybrid_rows(ac_run=0, dc_run=1, stale=99.0),
        )
        calc = HybridPowerFlowCalc(network, result_mode="array", verbose=False)
        rc = calc.run()

        self.assertEqual(0, rc)
        self.assertTrue(calc.converged)
        self.assertIsNone(calc.ac_calc)
        self.assertIsNotNone(calc.dc_calc)
        self.assertTrue(
            np.all(
                calc.result["ac"]["bus"]
                [:, [AC_BUS_COLS["voltage"], AC_BUS_COLS["angle"]]]
                == 0.0
            )
        )
        self.assertTrue(
            np.all(
                calc.result["ac"]["gen"]
                [:, [AC_GEN_COLS["p"], AC_GEN_COLS["q"], AC_GEN_COLS["current"]]]
                == 0.0
            )
        )
        self.assertEqual(1, int(calc.result["ac"]["gen"][0, AC_GEN_COLS["run_stat"]]))

    def test_hybrid_solver_returns_successful_zero_result_when_both_sides_are_dead(self):
        import numpy as np
        from ac_array_model import BUS_COLS as AC_BUS_COLS
        from dc_array_model import BUS_COLS as DC_BUS_COLS
        from lfcore.hybrid_lf import HybridPowerFlowCalc, _build_lf_network_from_hybrid_rows

        network = _build_lf_network_from_hybrid_rows(
            Path("all_dead.e"),
            _independent_hybrid_rows(ac_run=0, dc_run=0, stale=99.0),
        )
        calc = HybridPowerFlowCalc(network, result_mode="array", verbose=False)
        rc = calc.run()

        self.assertEqual(0, rc)
        self.assertTrue(calc.converged)
        self.assertEqual(0, calc.iterations)
        self.assertEqual(0.0, calc.normF)
        self.assertEqual((0, 0), calc.last_jacobian_shape)
        self.assertEqual(0, calc.x.size)
        self.assertTrue(
            np.all(
                calc.result["ac"]["bus"]
                [:, [AC_BUS_COLS["voltage"], AC_BUS_COLS["angle"]]]
                == 0.0
            )
        )
        self.assertTrue(
            np.all(calc.result["dc"]["bus"][:, DC_BUS_COLS["voltage"]] == 0.0)
        )

    def test_hybrid_solver_does_not_convert_active_numerical_failure_into_empty_success(self):
        from lfcore.hybrid_lf import HybridPowerFlowCalc, _build_lf_network_from_hybrid_rows

        network = _build_lf_network_from_hybrid_rows(
            Path("live_ac.e"),
            _independent_hybrid_rows(ac_run=1, dc_run=0),
        )
        calc = HybridPowerFlowCalc(network, result_mode="none", verbose=False)
        calc.prepare()

        def fail_active_ac():
            calc.ac_calc.converged = False
            calc.ac_calc.iterations = 1
            calc.ac_calc.normF = 1.0
            return -1

        calc.ac_calc._run_newton_raphson = fail_active_ac
        self.assertEqual(-1, calc.run())
        self.assertFalse(calc.converged)

    def test_inactive_dcac_converter_does_not_retain_stale_outputs(self):
        from hybrid_array_model import DCAC_COLS
        from lfcore.hybrid_lf import HybridPowerFlowCalc, _build_lf_network_from_hybrid_rows

        network = _build_lf_network_from_hybrid_rows(
            Path("inactive_converter.e"),
            _independent_hybrid_rows(
                ac_run=0,
                dc_run=0,
                stale=99.0,
                include_converter=True,
            ),
        )
        calc = HybridPowerFlowCalc(network, result_mode="array", verbose=False)

        self.assertEqual(0, calc.run())
        converter = network.ppc["dcac"][0]
        self.assertEqual(1, int(converter[DCAC_COLS["run_stat"]]))
        self.assertEqual(
            [0.0] * 5,
            [
                float(converter[DCAC_COLS[name]])
                for name in ("dc_p", "ac_p", "ac_q", "dc_i", "ac_i")
            ],
        )

    def test_skipped_lightweight_sides_do_not_materialize_object_facades(self):
        import lfcore.hybrid_lf as hybrid_lf

        network = hybrid_lf._build_lf_network_from_hybrid_rows(
            Path("all_dead_array_only.e"),
            _independent_hybrid_rows(ac_run=0, dc_run=0, stale=99.0),
        )
        original_array_device = hybrid_lf._array_device

        def reject_object_materialization(*_args, **_kwargs):
            raise AssertionError("skipped lightweight sides should stay PPC-backed")

        hybrid_lf._array_device = reject_object_materialization
        try:
            calc = hybrid_lf.HybridPowerFlowCalc(network, result_mode="array", verbose=False)
            self.assertEqual(0, calc.run())
        finally:
            hybrid_lf._array_device = original_array_device

    def test_general_hybrid_writeback_installs_dead_dc_results_for_every_mode(self):
        import numpy as np
        from dc_array_model import BUS_COLS, GEN_COLS, SWITCH_COLS
        import lfcore.hybrid_lf as hybrid_lf

        for mode in ("none", "summary", "array", "full"):
            with self.subTest(mode=mode):
                network = hybrid_lf._read_lf_network_from_file(
                    ROOT / "data" / "model" / "hybrid" / "hybrid_net_40.e"
                )
                source, dynamic_columns = _retire_dc_ppc_with_stale_outputs(network._dc_ppc)
                calc = hybrid_lf.HybridPowerFlowCalc(
                    network,
                    result_mode=mode,
                    verbose=False,
                )

                self.assertEqual(0, calc.run())
                self.assertTrue(calc.converged)
                self.assertIsNotNone(calc.ac_calc)
                self.assertIsNone(calc.dc_calc)
                self.assertEqual(1, calc.N_acac)
                self.assertFalse(calc._single_ac_newton_block)
                self.assertIsNotNone(network.dc.result)

                for key, columns in dynamic_columns.items():
                    actual = network.dc.result[key]
                    expected_source = source[key]
                    self.assertEqual(expected_source.shape, actual.shape, (mode, key))
                    self.assertTrue(np.all(actual[:, list(columns)] == 0.0), (mode, key))
                    static_columns = [
                        column
                        for column in range(actual.shape[1])
                        if column not in columns
                    ]
                    np.testing.assert_array_equal(
                        actual[:, static_columns],
                        expected_source[:, static_columns],
                        err_msg=f"{mode} {key} static columns changed",
                    )

                self.assertEqual(
                    0,
                    int(network.dc.result["bus"][0, BUS_COLS["run_stat"]]),
                )
                self.assertEqual(
                    float(source["gen"][0, GEN_COLS["v_set"]]),
                    float(network.dc.result["gen"][0, GEN_COLS["v_set"]]),
                )
                self.assertEqual(
                    int(source["gen"][0, GEN_COLS["run_stat"]]),
                    int(network.dc.result["gen"][0, GEN_COLS["run_stat"]]),
                )
                self.assertEqual(
                    int(source["switch"][0, SWITCH_COLS["status"]]),
                    int(network.dc.result["switch"][0, SWITCH_COLS["status"]]),
                )

                if mode == "none":
                    self.assertEqual({}, calc.result)
                    self.assertIsNone(calc.lf_result)
                elif mode == "summary":
                    self.assertTrue(np.all(calc.result["dc"]["voltage"] == 0.0))
                    self.assertIsNone(calc.lf_result)
                else:
                    self.assertIs(calc.result["dc"], network.dc.result)
                    if mode == "array":
                        self.assertIsNone(calc.lf_result)
                    else:
                        self.assertIsNotNone(calc.lf_result)

    def test_general_hybrid_none_writeback_zeros_dead_dc_full_objects(self):
        from dc_array_model import BUS_COLS
        import lfcore.hybrid_lf as hybrid_lf

        network = hybrid_lf.HybridPowerNetwork.read_from_file(
            ROOT / "data" / "model" / "hybrid" / "hybrid_net_40.e"
        )
        _retire_dc_ppc_with_stale_outputs(network._dc_ppc)
        for node in network.dc.nodes:
            node.run_stat = 0
            node.voltage = 99.0
        for generator in network.dc.generators:
            generator.p = 99.0
            generator.current = 99.0

        calc = hybrid_lf.HybridPowerFlowCalc(network, result_mode="none", verbose=False)

        self.assertEqual(0, calc.run())
        self.assertTrue(calc.converged)
        self.assertIsNone(calc.dc_calc)
        self.assertTrue(calc._single_ac_newton_block)
        self.assertEqual({}, calc.result)
        self.assertIsNotNone(network.dc.result)
        self.assertTrue(
            all(node.voltage == 0.0 and node.run_stat == 0 for node in network.dc.nodes)
        )
        self.assertTrue(
            all(generator.p == 0.0 and generator.current == 0.0 for generator in network.dc.generators)
        )
        self.assertTrue(
            all(
                row[BUS_COLS["voltage"]] == 0.0
                and int(row[BUS_COLS["run_stat"]]) == 0
                for row in network.dc.result["bus"]
            )
        )

    def test_skipped_result_synthesizes_canonical_empty_table_widths(self):
        import numpy as np
        from ac_array_model import (
            ACAC_COLS,
            BRANCH_COLS as AC_BRANCH_COLS,
            BREAK_COLS as AC_BREAK_COLS,
            BUS_COLS as AC_BUS_COLS,
            GEN_COLS as AC_GEN_COLS,
            LOAD_COLS as AC_LOAD_COLS,
            SHUNT_COLS as AC_SHUNT_COLS,
            SWITCH_COLS as AC_SWITCH_COLS,
            THREE_WINDING_TRANSFORMER_COLS as AC_THREE_COLS,
            TRANSFORMER_COLS as AC_TRANSFORMER_COLS,
            ZERO_BRANCH_COLS as AC_ZERO_BRANCH_COLS,
        )
        from dc_array_model import (
            BRANCH_COLS as DC_BRANCH_COLS,
            BREAK_COLS as DC_BREAK_COLS,
            BUS_COLS as DC_BUS_COLS,
            DCDC_COLS,
            GEN_COLS as DC_GEN_COLS,
            LOAD_COLS as DC_LOAD_COLS,
            SWITCH_COLS as DC_SWITCH_COLS,
            ZERO_BRANCH_COLS as DC_ZERO_BRANCH_COLS,
        )
        import lfcore.hybrid_lf as hybrid_lf

        ac_columns = {
            "bus": AC_BUS_COLS,
            "gen": AC_GEN_COLS,
            "load": AC_LOAD_COLS,
            "shunt": AC_SHUNT_COLS,
            "branch": AC_BRANCH_COLS,
            "transformer": AC_TRANSFORMER_COLS,
            "three_winding_transformer": AC_THREE_COLS,
            "zero_branch": AC_ZERO_BRANCH_COLS,
            "switch": AC_SWITCH_COLS,
            "break": AC_BREAK_COLS,
            "acac": ACAC_COLS,
        }
        dc_columns = {
            "bus": DC_BUS_COLS,
            "branch": DC_BRANCH_COLS,
            "load": DC_LOAD_COLS,
            "gen": DC_GEN_COLS,
            "zero_branch": DC_ZERO_BRANCH_COLS,
            "switch": DC_SWITCH_COLS,
            "break": DC_BREAK_COLS,
            "dcdc": DCDC_COLS,
        }

        ac_result = hybrid_lf._zero_subgrid_result(
            {},
            hybrid_lf._AC_ZERO_RESULT_TABLE_SPECS,
        )
        dc_result = hybrid_lf._zero_subgrid_result(
            {},
            hybrid_lf._DC_ZERO_RESULT_TABLE_SPECS,
        )

        for key, columns in ac_columns.items():
            self.assertEqual((0, max(columns.values()) + 1), ac_result[key].shape, key)
        for key, columns in dc_columns.items():
            self.assertEqual((0, max(columns.values()) + 1), dc_result[key].shape, key)
        self.assertGreater(
            ac_result["gen"].shape[1],
            AC_GEN_COLS["p_max"],
        )

        invalid_ac = hybrid_lf._zero_subgrid_result(
            {"gen": np.arange(3, dtype=np.float64)},
            hybrid_lf._AC_ZERO_RESULT_TABLE_SPECS,
        )
        self.assertEqual((0, max(AC_GEN_COLS.values()) + 1), invalid_ac["gen"].shape)

    def test_skipped_result_zeros_only_existing_columns_in_narrow_source_table(self):
        import numpy as np
        from ac_array_model import GEN_COLS
        import lfcore.hybrid_lf as hybrid_lf

        source = np.arange(1.0, 11.0, dtype=np.float64).reshape(1, 10)
        ppc = {"gen": source.copy()}
        try:
            result = hybrid_lf._zero_table_columns(
                ppc,
                "gen",
                hybrid_lf._AC_ZERO_RESULT_TABLE_SPECS["gen"],
            )
        except IndexError as exc:
            self.fail(f"narrow source tables must not raise IndexError: {exc}")

        expected = source.copy()
        expected[:, [GEN_COLS["p"], GEN_COLS["q"]]] = 0.0
        np.testing.assert_array_equal(expected, result)
        self.assertEqual(source.shape, result.shape)
        self.assertEqual(source[0, GEN_COLS["p_set"]], result[0, GEN_COLS["p_set"]])
        self.assertEqual(source[0, GEN_COLS["run_stat"]], result[0, GEN_COLS["run_stat"]])

    def test_hybrid_net_flow_does_not_import_network_classes(self):
        import hybrid_net_flow

        source = Path(hybrid_net_flow.__file__).read_text(encoding="utf-8")
        self.assertNotIn("from hybrid_net_model import ACPowerNetwork", source)
        self.assertNotIn("DCPowerNetwork", source)
        self.assertFalse(hasattr(hybrid_net_flow, "ACPowerNetwork"))
        self.assertFalse(hasattr(hybrid_net_flow, "DCPowerNetwork"))
        self.assertFalse(hasattr(hybrid_net_flow, "HybridACGrid"))
        self.assertFalse(hasattr(hybrid_net_flow, "HybridDCGrid"))
        self.assertFalse(hasattr(hybrid_net_flow, "run_hybrid_power_flow"))
        self.assertTrue(hasattr(hybrid_net_flow, "HybridPowerNetwork"))

    def test_hybrid_net_40_runs_from_self_contained_network(self):
        from scipy.sparse import issparse
        import hybrid_net_flow

        result = _run_hybrid_lf(
            ROOT / "data" / "model" / "hybrid" / "hybrid_net_40.e",
            verbose=False,
        )

        self.assertTrue(result.converged)
        self.assertEqual(result.total_nodes, 40)
        self.assertEqual(len(result.ac_network.nodes), 10)
        self.assertEqual(len(result.dc_network.nodes), 30)
        self.assertTrue(result.has_acac)
        self.assertEqual(result.calc.N_acac, 1)
        self.assertGreater(result.global_jacobian_shape[0], 0)
        self.assertEqual(result.global_jacobian_shape[0], result.global_jacobian_shape[1])

        ac_x, dc_x, _, _ = result.calc._split_x(result.calc.x)
        with contextlib.redirect_stdout(io.StringIO()):
            ac_j = result.calc.ac_calc.get_jacobi(ac_x)
            dc_j = result.calc.dc_calc.get_jacobi(dc_x)
            hybrid_j = result.calc.get_jacobi(result.calc.x)
        self.assertTrue(issparse(ac_j))
        self.assertTrue(issparse(dc_j))
        self.assertTrue(issparse(hybrid_j))

    def test_ac_generator_writeback_includes_converter_bus_injection(self):
        from ac_array_model import GEN_COLS
        from lfcore.hybrid_lf import HybridPowerFlowCalc, _read_lf_network_from_file

        network = _read_lf_network_from_file(ROOT / "data" / "model" / "hybrid" / "hybrid_net_40.e")
        calc = HybridPowerFlowCalc(network, verbose=False, result_mode="array")
        with contextlib.redirect_stdout(io.StringIO()):
            calc.run()

        names = calc.ac_calc.ppc["gen_name"]
        row = list(names).index("gen_pq9")
        gen_result = calc.ac_calc.result["gen"][row]
        gen_source = calc.ac_calc.ppc["gen"][row]
        self.assertAlmostEqual(float(gen_source[GEN_COLS["p_set"]]), float(gen_result[GEN_COLS["p"]]), places=9)
        self.assertAlmostEqual(float(gen_source[GEN_COLS["q_set"]]), float(gen_result[GEN_COLS["q"]]), places=9)

    def test_hybrid_jacobian_builds_converter_terms_in_one_sparse_pass(self):
        from scipy.sparse import issparse
        import hybrid_net_flow

        network = hybrid_net_flow.HybridPowerNetwork.read_from_file(
            ROOT / "data" / "model" / "hybrid" / "hybrid_net_40.e"
        )
        network.prepare(verbose=False)
        calc = hybrid_net_flow.HybridPowerFlowCalc(network, verbose=False)
        with contextlib.redirect_stdout(io.StringIO()):
            calc.prepare()

        def fail_old_converter_sparse_path(*_args, **_kwargs):
            raise AssertionError("converter coupling terms should be appended to the global COO data")

        calc._get_dcac_jacobi = fail_old_converter_sparse_path
        calc._get_acac_jacobi = fail_old_converter_sparse_path

        jac = calc.get_jacobi(calc.x)

        self.assertTrue(issparse(jac))
        self.assertEqual(jac.shape, (calc.total_eq, calc.total_vars))

    def test_hybrid_jacobian_reuses_precomputed_global_csr_pattern(self):
        import numpy as np
        import lfcore.hybrid_lf as hybrid_lf

        network = hybrid_lf.HybridPowerNetwork.read_from_file(
            ROOT / "data" / "model" / "hybrid" / "hybrid_net_40.e"
        )
        calc = hybrid_lf.HybridPowerFlowCalc(network, verbose=False)
        with contextlib.redirect_stdout(io.StringIO()):
            calc.prepare()

        expected = calc.get_jacobi(calc.x).toarray()
        self.assertGreater(calc.global_jac_csr_indices.size, 0)

        original_coo_matrix = hybrid_lf.coo_matrix

        def reject_coo_matrix(*_args, **_kwargs):
            raise AssertionError("Hybrid Jacobian should refresh precomputed global CSR data")

        hybrid_lf.coo_matrix = reject_coo_matrix
        try:
            actual = calc.get_jacobi(calc.x).toarray()
        finally:
            hybrid_lf.coo_matrix = original_coo_matrix

        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_hybrid_newton_system_direct_stamps_subsolver_data(self):
        import numpy as np
        import ac_lf
        import lfcore.dc_lf as dc_lf
        import lfcore.hybrid_lf as hybrid_lf

        network = hybrid_lf.HybridPowerNetwork.read_from_file(
            ROOT / "data" / "model" / "hybrid" / "hybrid_net_40.e"
        )
        calc = hybrid_lf.HybridPowerFlowCalc(network, verbose=False)
        with contextlib.redirect_stdout(io.StringIO()):
            calc.prepare()

        _expected_f, expected_j = calc._build_newton_system(calc.x)

        original_ac_csr = ac_lf.csr_matrix
        original_dc_csr = dc_lf.csr_matrix

        def reject_submatrix_materialization(*_args, **_kwargs):
            raise AssertionError("Hybrid assembly should stamp AC/DC Jacobian data without submatrix materialization")

        ac_lf.csr_matrix = reject_submatrix_materialization
        dc_lf.csr_matrix = reject_submatrix_materialization
        try:
            _actual_f, actual_j = calc._build_newton_system(calc.x)
        finally:
            ac_lf.csr_matrix = original_ac_csr
            dc_lf.csr_matrix = original_dc_csr

        self.assertEqual("csc", actual_j.format)
        np.testing.assert_allclose(actual_j.toarray(), expected_j.toarray(), atol=1e-12)

    def test_hybrid_residual_reuses_preallocated_work_array(self):
        import numpy as np
        import lfcore.hybrid_lf as hybrid_lf

        network = hybrid_lf.HybridPowerNetwork.read_from_file(
            ROOT / "data" / "model" / "hybrid" / "hybrid_net_40.e"
        )
        calc = hybrid_lf.HybridPowerFlowCalc(network, verbose=False)
        with contextlib.redirect_stdout(io.StringIO()):
            calc.prepare()

        expected_f = calc.get_f(calc.x).copy()
        self.assertEqual(calc.total_eq, expected_f.size)

        original_concatenate = hybrid_lf.np.concatenate

        def reject_concatenate(*_args, **_kwargs):
            raise AssertionError("Hybrid residual should reuse preallocated F workspace")

        hybrid_lf.np.concatenate = reject_concatenate
        try:
            actual_f = calc.get_f(calc.x)
            combined_f, jac = calc._build_newton_system(calc.x)
        finally:
            hybrid_lf.np.concatenate = original_concatenate

        self.assertIs(actual_f, calc._residual_work)
        self.assertIs(combined_f, calc._residual_work)
        np.testing.assert_allclose(actual_f, expected_f, atol=1e-12)
        np.testing.assert_allclose(combined_f, expected_f, atol=1e-12)
        self.assertEqual(jac.shape, (calc.total_eq, calc.total_vars))

    def test_hybrid_verbose_false_avoids_stdout_redirect_wrappers(self):
        import lfcore.hybrid_lf as hybrid_lf

        network = hybrid_lf.HybridPowerNetwork.read_from_file(
            ROOT / "data" / "model" / "hybrid" / "hybrid_net_40.e"
        )
        calc = hybrid_lf.HybridPowerFlowCalc(network, verbose=False, result_mode="none")

        original_runner = getattr(hybrid_lf, "_run_with_optional_output", None)

        def reject_output_wrapper(*_args, **_kwargs):
            raise AssertionError("Hybrid quiet path should call quiet sub-solvers directly")

        if original_runner is not None:
            hybrid_lf._run_with_optional_output = reject_output_wrapper
        captured = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured):
                calc.prepare()
                rc = calc.run()
        finally:
            if original_runner is not None:
                hybrid_lf._run_with_optional_output = original_runner

        self.assertEqual(0, rc)
        self.assertEqual("", captured.getvalue())

    def test_hybrid_newton_uses_current_factored_sparse_solver(self):
        import lfcore.hybrid_lf as hybrid_lf

        network = hybrid_lf.HybridPowerNetwork.read_from_file(ROOT / "data" / "model" / "hybrid" / "hybrid_net_40.e")
        calc = hybrid_lf.HybridPowerFlowCalc(network, verbose=False, linear_solver="scipy")
        with contextlib.redirect_stdout(io.StringIO()):
            calc.prepare()

        self.assertEqual("scipy", calc.ac_calc.linear_solver)
        self.assertEqual("scipy", calc.dc_calc.linear_solver)

        original_factor = hybrid_lf._factor_jacobian
        calls = []

        def counted_factor(matrix, resolved_name, solver_fn):
            calls.append((matrix.shape, resolved_name, solver_fn))
            return original_factor(matrix, resolved_name, solver_fn)

        hybrid_lf._factor_jacobian = counted_factor
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                rc = calc.run()
        finally:
            hybrid_lf._factor_jacobian = original_factor

        self.assertEqual(0, rc)
        self.assertTrue(calls)
        self.assertTrue(all(shape == (calc.total_eq, calc.total_vars) for shape, _solver, _fn in calls))
        self.assertTrue(all(solver == "scipy" for _shape, solver, _fn in calls))

    def test_hybrid_with_dc_defaults_to_pyklu_linear_solver(self):
        import lfcore.hybrid_lf as hybrid_lf

        network = hybrid_lf.HybridPowerNetwork.read_from_file(ROOT / "data" / "model" / "hybrid" / "hybrid_net_40.e")
        calc = hybrid_lf.HybridPowerFlowCalc(network, verbose=False, result_mode="none")
        explicit_calc = hybrid_lf.HybridPowerFlowCalc(
            network,
            verbose=False,
            linear_solver="umfpack",
            result_mode="none",
        )

        self.assertEqual("pyklu", calc.linear_solver)
        self.assertIn(calc._linear_solver_resolved, {"pyklu", "scipy"})
        self.assertEqual("umfpack", explicit_calc.linear_solver)
        self.assertIn(explicit_calc._linear_solver_resolved, {"umfpack", "scipy"})

    def test_converter_terms_reuse_ac_state_cache(self):
        import hybrid_net_flow

        network = hybrid_net_flow.HybridPowerNetwork.read_from_file(
            ROOT / "data" / "model" / "hybrid" / "hybrid_net_40.e"
        )
        network.prepare(verbose=False)
        calc = hybrid_net_flow.HybridPowerFlowCalc(network, verbose=False)
        with contextlib.redirect_stdout(io.StringIO()):
            calc.prepare()

        original_extract = calc.ac_calc._extract_state_vars
        call_count = 0

        def counted_extract(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_extract(*args, **kwargs)

        calc.ac_calc._extract_state_vars = counted_extract
        calc.get_f(calc.x)
        self.assertEqual(1, call_count)

        call_count = 0
        calc.get_jacobi(calc.x)
        self.assertEqual(1, call_count)

        call_count = 0
        calc._write_back()
        self.assertEqual(1, call_count)

    def test_hybrid_network_load_uses_array_model_for_all_case_shapes(self):
        import lfcore.hybrid_lf as hybrid_lf

        ac_network = hybrid_lf.HybridPowerNetwork.read_from_file(ROOT / "data" / "model" / "ac" / "ieee300.e")
        dc_network = hybrid_lf.HybridPowerNetwork.read_from_file(ROOT / "data" / "model" / "dc" / "dc_net_30.e")
        hybrid_network = hybrid_lf.HybridPowerNetwork.read_from_file(ROOT / "data" / "model" / "hybrid" / "hybrid_net_40.e")

        self.assertEqual("hybrid_ppc_v1", ac_network.ppc["format"])
        self.assertEqual("hybrid_ppc_v1", dc_network.ppc["format"])
        self.assertEqual("hybrid_ppc_v1", hybrid_network.ppc["format"])
        self.assertTrue(hasattr(ac_network, "_ac_ppc"))
        self.assertTrue(hasattr(dc_network, "_dc_ppc"))
        self.assertTrue(hasattr(hybrid_network, "_ac_ppc"))
        self.assertTrue(hasattr(hybrid_network, "_dc_ppc"))
        self.assertTrue(ac_network.ac.nodes)
        self.assertTrue(dc_network.dc.nodes)
        self.assertTrue(hybrid_network.dcac_converters or hybrid_network.acac_converters)

    def test_hybrid_ppc_builds_sub_ppc_directly_from_loaded_rows(self):
        import model.hybrid_array_model as hybrid_array_model

        case_path = ROOT / "data" / "model" / "hybrid" / "hybrid_net_40.e"
        original_factory = hybrid_array_model.efile_factory_from_file
        original_rows_factory = hybrid_array_model.efile_factory_from_rows
        original_ac_builder = hybrid_array_model._build_ac_ppc_from_model
        original_dc_builder = hybrid_array_model._build_dc_ppc_from_model
        original_normalize = hybrid_array_model.normalize_model_named_units

        def reject_object_path(*_args, **_kwargs):
            raise AssertionError("hybrid E-file PPC load should not build dynamic model objects")

        hybrid_array_model.efile_factory_from_file = reject_object_path
        hybrid_array_model.efile_factory_from_rows = reject_object_path
        hybrid_array_model._build_ac_ppc_from_model = reject_object_path
        hybrid_array_model._build_dc_ppc_from_model = reject_object_path
        hybrid_array_model.normalize_model_named_units = reject_object_path
        try:
            _network, ppc = hybrid_array_model.build_hybrid_ppc_from_e_file(case_path)
        finally:
            hybrid_array_model.efile_factory_from_file = original_factory
            hybrid_array_model.efile_factory_from_rows = original_rows_factory
            hybrid_array_model._build_ac_ppc_from_model = original_ac_builder
            hybrid_array_model._build_dc_ppc_from_model = original_dc_builder
            hybrid_array_model.normalize_model_named_units = original_normalize

        self.assertIs(ppc["ac_network"].ppc, ppc["ac"])
        self.assertIs(ppc["dc_network"].ppc, ppc["dc"])
        self.assertEqual(10, len(ppc["ac_network"].nodes))
        self.assertEqual(30, len(ppc["dc_network"].nodes))
        self.assertEqual(10, ppc["ac"]["bus"].shape[0])
        self.assertEqual(30, ppc["dc"]["bus"].shape[0])
        self.assertGreater(ppc["dcac"].shape[0], 0)
        self.assertGreater(ppc["acac"].shape[0], 0)

    def test_lf_loader_uses_lightweight_hybrid_network_from_ppc(self):
        import lfcore.hybrid_lf as hybrid_lf

        network = hybrid_lf._read_lf_network_from_file(ROOT / "data" / "model" / "hybrid" / "hybrid_net_40.e")

        self.assertFalse(hasattr(hybrid_lf, "_build_lf_ac_facade"))
        self.assertTrue(hasattr(hybrid_lf, "build_hybrid_ppc_with_topology_from_efile_rows"))
        self.assertTrue(getattr(network.ac, "_lf_lightweight", False))
        self.assertTrue(getattr(network.dc, "_lf_lightweight", False))
        self.assertNotIn("ac_network", network.ppc)
        self.assertNotIn("dc_network", network.ppc)
        self.assertIn("_topology_arrays", network.ppc["ac"])
        self.assertIn("_topology_arrays", network.ppc["dc"])
        self.assertEqual(10, len(network.ac.nodes))
        self.assertGreater(len(network.dc.nodes), 0)
        self.assertGreater(len(network.dcac_converters), 0)
        self.assertGreater(len(network.acac_converters), 0)

    def test_lf_loader_does_not_rebuild_full_ac_dc_networks(self):
        import lfcore.hybrid_lf as hybrid_lf

        original_ppc_loader = hybrid_lf.build_hybrid_ppc_with_topology_from_efile_rows
        calls = []

        def counted_ppc_loader(path, rows):
            calls.append(Path(path).name)
            return original_ppc_loader(path, rows)

        hybrid_lf.build_hybrid_ppc_with_topology_from_efile_rows = counted_ppc_loader
        try:
            network = hybrid_lf._read_lf_network_from_file(ROOT / "data" / "model" / "hybrid" / "hybrid_net_40.e")
        finally:
            hybrid_lf.build_hybrid_ppc_with_topology_from_efile_rows = original_ppc_loader

        self.assertEqual(["hybrid_net_40.e"], calls)
        self.assertTrue(getattr(network.ac, "_lf_lightweight", False))
        self.assertTrue(getattr(network.dc, "_lf_lightweight", False))
        self.assertIs(network._ac_ppc, network.ppc["ac"])
        self.assertIs(network._dc_ppc, network.ppc["dc"])

    def test_hybrid_lf_array_path_keeps_converters_ppc_backed(self):
        import lfcore.hybrid_lf as hybrid_lf

        original_array_device = hybrid_lf._array_device

        def reject_converter_materialization(*_args, **_kwargs):
            raise AssertionError("Hybrid LF array path should not materialize converter SimpleNamespace objects")

        hybrid_lf._array_device = reject_converter_materialization
        try:
            network = hybrid_lf._read_lf_network_from_file(ROOT / "data" / "model" / "hybrid" / "hybrid_net_40.e")
            self.assertEqual(network.ppc["dcac"].shape[0], len(network.dcac_converters))
            self.assertEqual(network.ppc["acac"].shape[0], len(network.acac_converters))
            self.assertNotIsInstance(network.dcac_converters, list)
            self.assertNotIsInstance(network.acac_converters, list)

            calc = hybrid_lf.HybridPowerFlowCalc(network, verbose=False, result_mode="array")
            with contextlib.redirect_stdout(io.StringIO()):
                calc.prepare()
                rc = calc.run(result_mode="array")
        finally:
            hybrid_lf._array_device = original_array_device

        self.assertEqual(0, rc)
        self.assertTrue(calc.converged)
        self.assertEqual(network.ppc["dcac"].shape[0], calc.N_dcac)
        self.assertEqual(network.ppc["acac"].shape[0], calc.N_acac)
        self.assertEqual((calc.N_dcac, 5), calc.result["dcac"].shape)
        self.assertEqual((calc.N_acac, 6), calc.result["acac"].shape)
        self.assertFalse(calc.dcac_devices)
        self.assertFalse(calc.acac_devices)

    def test_hybrid_lf_full_result_builds_converter_entries_from_ppc(self):
        import lfcore.hybrid_lf as hybrid_lf

        original_array_device = hybrid_lf._array_device

        def reject_converter_materialization(*_args, **_kwargs):
            raise AssertionError("Hybrid LF full result should not materialize converter SimpleNamespace objects")

        hybrid_lf._array_device = reject_converter_materialization
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                result = _run_hybrid_lf(
                    ROOT / "data" / "model" / "hybrid" / "hybrid_net_40.e",
                    verbose=False,
                    result_mode="full",
                )
        finally:
            hybrid_lf._array_device = original_array_device

        self.assertTrue(result.converged, (result.ac_errors, result.dc_errors, result.calc.normF))
        self.assertTrue(result.dcac.dcac_converters)
        self.assertTrue(result.acac.acac_converters)
        self.assertFalse(result.calc.dcac_devices)
        self.assertFalse(result.calc.acac_devices)
        first_dcac = next(iter(result.dcac.dcac_converters.values()))
        first_acac = next(iter(result.acac.acac_converters.values()))
        self.assertGreater(first_dcac.i_v, 0.0)
        self.assertGreater(first_dcac.j_v, 0.0)
        self.assertGreater(first_acac.i_v, 0.0)
        self.assertGreater(first_acac.j_v, 0.0)
        self.assertIn("ac", result.arrays)
        self.assertIn("dc", result.arrays)
        self.assertIn("dcac", result.arrays)
        self.assertIn("acac", result.arrays)
        self.assertIs(result.arrays["ac"], result.calc.result["ac"])
        self.assertIs(result.arrays["dc"], result.calc.result["dc"])
        self.assertIs(result.arrays["dcac"], result.calc.result["dcac"])
        self.assertIs(result.arrays["acac"], result.calc.result["acac"])

    def test_hybrid_ppc_model_build_delegates_to_ac_dc_array_helpers(self):
        import model.hybrid_array_model as hybrid_array_model

        _network, ppc = hybrid_array_model.build_hybrid_ppc_from_e_file(ROOT / "data" / "model" / "hybrid" / "hybrid_net_40.e")

        self.assertFalse(hasattr(hybrid_array_model, "_build_ac_network"))
        self.assertFalse(hasattr(hybrid_array_model, "_build_dc_network"))
        self.assertFalse(hasattr(hybrid_array_model, "_empty_ac_ppc"))
        self.assertFalse(hasattr(hybrid_array_model, "_empty_dc_ppc"))

        self.assertFalse(hasattr(hybrid_array_model, "build_ac_network_from_ppc"))
        self.assertFalse(hasattr(hybrid_array_model, "build_dc_network_from_ppc"))
        model = hybrid_array_model.build_hybrid_model_from_ppc(ppc)

        from model.hybrid_model import ACACConverter, DCACConverter, HybridPowerNetwork

        self.assertIsInstance(model, HybridPowerNetwork)
        self.assertIs(model.ac, ppc["ac_network"])
        self.assertIs(model.dc, ppc["dc_network"])
        self.assertTrue(all(isinstance(conv, DCACConverter) for conv in model.DCACConverter))
        self.assertTrue(all(isinstance(conv, ACACConverter) for conv in model.ACACConverter))
        self.assertEqual(10, len(model.ac.nodes))
        self.assertEqual(30, len(model.dc.nodes))

    def test_hybrid_calc_run_returns_hybrid_lf_result(self):
        import lfcore.hybrid_lf as hybrid_lf

        result = _run_hybrid_lf(
            ROOT / "data" / "model" / "hybrid" / "hybrid_net_40.e",
            verbose=False,
        )

        self.assertFalse(hasattr(hybrid_lf, "HybridPowerFlowResult"))
        self.assertFalse(hasattr(hybrid_lf, "run_hybrid_power_flow"))
        self.assertIsInstance(result, hybrid_lf.HybridLFResult)
        self.assertIs(result.lf_result, result)
        self.assertTrue(result.converged)
        self.assertIs(result.network, result.calc.network)
        self.assertIsNone(result.ac)
        self.assertIsNone(result.dc)
        self.assertIsNone(getattr(result.calc.ac_calc, "lf_result", None))
        self.assertIsNone(getattr(result.calc.dc_calc, "lf_result", None))
        self.assertEqual("array", result.calc.ac_calc.result_mode)
        self.assertEqual("array", result.calc.dc_calc.result_mode)
        self.assertIn("bus", result.calc.ac_calc.result)
        self.assertIn("bus", result.calc.dc_calc.result)
        self.assertTrue(result.dcac.dcac_converters)
        self.assertTrue(result.acac.acac_converters)

    def test_hybrid_subcalculations_always_use_array_results(self):
        from lfcore.hybrid_lf import HybridPowerFlowCalc, _read_lf_network_from_file

        for mode in ("none", "summary", "array", "full"):
            network = _read_lf_network_from_file(ROOT / "data" / "model" / "hybrid" / "hybrid_net_40.e")
            calc = HybridPowerFlowCalc(network, verbose=False, result_mode=mode)
            with contextlib.redirect_stdout(io.StringIO()):
                calc.prepare()

            self.assertEqual("array", calc.ac_calc.result_mode)
            self.assertEqual("array", calc.dc_calc.result_mode)
            self.assertFalse(calc.ac_calc.keep_node_objects)
            self.assertFalse(calc.dc_calc.keep_node_objects)

            def reject_sub_lf_result(*_args, **_kwargs):
                raise AssertionError("Hybrid sub-solvers must not build object LF results")

            calc.ac_calc._build_lf_result = reject_sub_lf_result
            calc.dc_calc._build_lf_result = reject_sub_lf_result
            with contextlib.redirect_stdout(io.StringIO()):
                rc = calc.run()

            self.assertEqual(0, rc, mode)
            self.assertEqual("array", calc.ac_calc.result_mode)
            self.assertEqual("array", calc.dc_calc.result_mode)
            self.assertIsNone(getattr(calc.ac_calc, "lf_result", None))
            self.assertIsNone(getattr(calc.dc_calc, "lf_result", None))
            self.assertIn("bus", calc.ac_calc.result)
            self.assertIn("bus", calc.dc_calc.result)

    def test_converter_initial_values_and_writeback_use_cached_arrays(self):
        from lfcore.hybrid_lf import HybridPowerFlowCalc, _read_lf_network_from_file

        class NonIterableConverters:
            def __iter__(self):
                raise AssertionError("hybrid converter helpers should use cached converter arrays")

        network = _read_lf_network_from_file(ROOT / "data" / "model" / "hybrid" / "hybrid_net_40.e")
        calc = HybridPowerFlowCalc(network, verbose=False)
        with contextlib.redirect_stdout(io.StringIO()):
            calc.prepare()

        expected_dcac = calc._initial_dcac_x()
        expected_acac = calc._initial_acac_x()
        calc.dcac_converters = NonIterableConverters()
        calc.acac_converters = NonIterableConverters()

        self.assertGreater(calc.N_dcac, 0)
        self.assertGreater(calc.N_acac, 0)
        self.assertEqual(expected_dcac.tolist(), calc._initial_dcac_x().tolist())
        self.assertEqual(expected_acac.tolist(), calc._initial_acac_x().tolist())
        calc._write_back()

    def test_hybrid_power_flow_can_skip_full_lf_result_build(self):
        from lfcore.hybrid_lf import HybridPowerFlowCalc, HybridPowerNetwork

        network = HybridPowerNetwork.read_from_file(ROOT / "data" / "model" / "hybrid" / "hybrid_net_40.e")
        calc = HybridPowerFlowCalc(network, verbose=False)
        calc.skip_lf_result = True

        def reject_lf_result(*_args, **_kwargs):
            raise AssertionError("Hybrid LF should skip full result construction when requested")

        calc._build_lf_result = reject_lf_result
        with contextlib.redirect_stdout(io.StringIO()):
            rc = calc.run()

        self.assertEqual(0, rc)
        self.assertIsNone(calc.lf_result)
        self.assertIsNone(getattr(calc.ac_calc, "lf_result", None))
        self.assertIsNone(getattr(calc.dc_calc, "lf_result", None))

    def test_result_mode_skips_full_hybrid_result_backfill(self):
        from lfcore.hybrid_lf import HybridPowerFlowCalc, HybridPowerNetwork

        network = HybridPowerNetwork.read_from_file(ROOT / "data" / "model" / "hybrid" / "hybrid_net_40.e")
        calc = HybridPowerFlowCalc(network, verbose=False, result_mode="none")

        def reject_full_result_build(*_args, **_kwargs):
            raise AssertionError("result_mode should skip full hybrid LF result construction")

        calc._build_lf_result = reject_full_result_build
        with contextlib.redirect_stdout(io.StringIO()):
            rc = calc.run()

        self.assertEqual(0, rc)
        self.assertTrue(calc.converged)
        self.assertIsNone(calc.lf_result)
        self.assertTrue(hasattr(calc, "x"))

        summary_calc = HybridPowerFlowCalc(network, verbose=False, result_mode="summary")
        summary_calc._build_lf_result = reject_full_result_build
        with contextlib.redirect_stdout(io.StringIO()):
            rc = summary_calc.run()

        self.assertEqual(0, rc)
        self.assertTrue(summary_calc.converged)
        self.assertIsNone(summary_calc.lf_result)
        self.assertEqual(
            {"ac", "dc", "hybrid"},
            set(summary_calc.result),
        )
        self.assertEqual(summary_calc.iterations, summary_calc.result["hybrid"]["iterations"])

    def test_array_result_mode_keeps_arrays_without_hybrid_object_backfill(self):
        import numpy as np
        from lfcore.hybrid_lf import HybridPowerFlowCalc, _read_lf_network_from_file

        network = _read_lf_network_from_file(ROOT / "data" / "model" / "hybrid" / "hybrid_net_40.e")
        calc = HybridPowerFlowCalc(network, verbose=False, result_mode="none")

        def reject_full_result_build(*_args, **_kwargs):
            raise AssertionError("result_mode='array' should skip full hybrid LF result construction")

        calc._build_lf_result = reject_full_result_build
        with contextlib.redirect_stdout(io.StringIO()):
            rc = calc.run(result_mode="array")

        self.assertEqual(0, rc)
        self.assertTrue(calc.converged)
        self.assertIsNone(calc.lf_result)
        self.assertEqual({"ac", "dc", "dcac", "acac", "summary"}, set(calc.result))
        self.assertIsInstance(calc.result["dcac"], np.ndarray)
        self.assertIsInstance(calc.result["acac"], np.ndarray)
        self.assertIn("bus", calc.result["ac"])
        self.assertIn("bus", calc.result["dc"])
        self.assertIsNone(getattr(calc.ac_calc, "lf_result", None))
        self.assertIsNone(getattr(calc.dc_calc, "lf_result", None))

        result = _run_hybrid_lf(
            ROOT / "data" / "model" / "hybrid" / "hybrid_net_40.e",
            verbose=False,
            result_mode="array",
        )
        self.assertIsInstance(result, dict)
        self.assertEqual({"ac", "dc", "dcac", "acac", "summary"}, set(result))
        self.assertNotIn("calc", result)
        self.assertNotIn("network", result)
        self.assertTrue(result["summary"]["converged"])
        self.assertIsInstance(result["ac"]["bus"], np.ndarray)
        self.assertIsInstance(result["dc"]["bus"], np.ndarray)
        self.assertIsInstance(result["dcac"], np.ndarray)
        self.assertIsInstance(result["acac"], np.ndarray)

        ac_only_result = _run_hybrid_lf(
            ROOT / "data" / "model" / "ac" / "ieee14.e",
            verbose=False,
            result_mode="array",
        )
        self.assertIsInstance(ac_only_result, dict)
        self.assertTrue(ac_only_result["summary"]["converged"])
        self.assertIn("bus", ac_only_result["ac"])
        self.assertIsNone(ac_only_result["dc"])
        self.assertEqual((0, 5), ac_only_result["dcac"].shape)
        self.assertEqual((0, 6), ac_only_result["acac"].shape)

    def test_single_ac_hybrid_newton_uses_ac_solver_without_global_packaging(self):
        from lfcore.hybrid_lf import HybridPowerFlowCalc, _read_lf_network_from_file

        network = _read_lf_network_from_file(ROOT / "data" / "model" / "ac" / "ieee14.e")
        calc = HybridPowerFlowCalc(network, verbose=False, result_mode="none")
        with contextlib.redirect_stdout(io.StringIO()):
            calc.prepare()

        def reject_residual_packaging(*_args, **_kwargs):
            raise AssertionError("single AC block should reuse AC residuals directly")

        def reject_jacobian_packaging(*_args, **_kwargs):
            raise AssertionError("single AC block should reuse AC Jacobian directly")

        calc._fill_residual_work = reject_residual_packaging
        calc._assemble_jacobian = reject_jacobian_packaging
        with contextlib.redirect_stdout(io.StringIO()):
            rc = calc.run()

        self.assertEqual(0, rc)
        self.assertTrue(calc.converged)
        self.assertTrue(calc._single_ac_newton_block)
        self.assertEqual(0, calc.global_jac_raw_data.size)
        self.assertEqual(calc.ac_calc.iterations, calc.iterations)

    def test_single_ac_hybrid_newton_calls_ac_core_without_run_prepare_gate(self):
        import lfcore.hybrid_lf as hybrid_lf

        network = hybrid_lf._read_lf_network_from_file(ROOT / "data" / "model" / "ac" / "ieee14.e")
        calc = hybrid_lf.HybridPowerFlowCalc(network, verbose=False, result_mode="none")
        with contextlib.redirect_stdout(io.StringIO()):
            calc.prepare()
        self.assertTrue(calc._single_ac_newton_block)

        calls = []
        original_ac_core = calc.ac_calc._run_newton_raphson

        def counted_ac_core():
            calls.append("core")
            return original_ac_core()

        def reject_ac_run(*_args, **_kwargs):
            raise AssertionError("single AC hybrid run should call ACPowerFlowCalc._run_newton_raphson directly")

        def reject_ac_prepare(*_args, **_kwargs):
            raise AssertionError("prepared single AC hybrid run should not call ACPowerFlowCalc.prepare again")

        calc.ac_calc._run_newton_raphson = counted_ac_core
        calc.ac_calc.run = reject_ac_run
        calc.ac_calc.prepare = reject_ac_prepare
        with contextlib.redirect_stdout(io.StringIO()):
            rc = calc.run()

        self.assertEqual(0, rc)
        self.assertEqual(["core"], calls)
        self.assertTrue(calc.converged)

    def test_single_dc_hybrid_newton_uses_dc_solver_without_global_packaging(self):
        from lfcore.hybrid_lf import HybridPowerFlowCalc, _read_lf_network_from_file

        network = _read_lf_network_from_file(ROOT / "data" / "model" / "dc" / "dc_net_30.e")
        calc = HybridPowerFlowCalc(network, verbose=False, result_mode="none")
        with contextlib.redirect_stdout(io.StringIO()):
            calc.prepare()

        def reject_residual_packaging(*_args, **_kwargs):
            raise AssertionError("single DC block should reuse DC residuals directly")

        def reject_jacobian_packaging(*_args, **_kwargs):
            raise AssertionError("single DC block should reuse DC Jacobian directly")

        calc._fill_residual_work = reject_residual_packaging
        calc._assemble_jacobian = reject_jacobian_packaging
        with contextlib.redirect_stdout(io.StringIO()):
            rc = calc.run()

        self.assertEqual(0, rc)
        self.assertTrue(calc.converged)
        self.assertTrue(calc._single_dc_newton_block)
        self.assertEqual(0, calc.global_jac_raw_data.size)
        self.assertEqual(calc.dc_calc.iterations, calc.iterations)

    def test_single_dc_hybrid_run_delegates_to_dc_newton_core(self):
        import lfcore.hybrid_lf as hybrid_lf

        network = hybrid_lf._read_lf_network_from_file(ROOT / "data" / "model" / "dc" / "dc_net_30.e")
        calc = hybrid_lf.HybridPowerFlowCalc(network, verbose=False, result_mode="array")
        with contextlib.redirect_stdout(io.StringIO()):
            calc.prepare()

        calls = []
        original_dc_core = calc.dc_calc._run_newton_raphson
        original_factor = hybrid_lf._factor_jacobian

        def counted_dc_core():
            calls.append("core")
            return original_dc_core()

        def reject_dc_run(*_args, **_kwargs):
            raise AssertionError("single DC hybrid run should call DCPowerFlowCalc._run_newton_raphson directly")

        def reject_hybrid_factor(*_args, **_kwargs):
            raise AssertionError("single DC hybrid run should delegate to the DC Newton core")

        calc.dc_calc._run_newton_raphson = counted_dc_core
        calc.dc_calc.run = reject_dc_run
        hybrid_lf._factor_jacobian = reject_hybrid_factor
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                rc = calc.run()
        finally:
            calc.dc_calc._run_newton_raphson = original_dc_core
            hybrid_lf._factor_jacobian = original_factor

        self.assertEqual(0, rc)
        self.assertEqual(["core"], calls)
        self.assertTrue(calc.converged)
        self.assertIs(calc.result["dc"], calc.dc_calc.result)
        self.assertEqual(0, calc.result["ac"]["bus"].shape[0])

    def test_hybrid_run_lets_run_prepare_once(self):
        import lfcore.hybrid_lf as hybrid_lf

        original_run = hybrid_lf.HybridPowerFlowCalc.run
        x_sizes_before_run = []

        def counted_run(self, *args, **kwargs):
            x_sizes_before_run.append(int(self.x.size))
            return original_run(self, *args, **kwargs)

        network = hybrid_lf._read_lf_network_from_file(ROOT / "data" / "model" / "ac" / "ieee14.e")
        calc = hybrid_lf.HybridPowerFlowCalc(network, verbose=False, result_mode="array")
        hybrid_lf.HybridPowerFlowCalc.run = counted_run
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                rc = calc.run()
        finally:
            hybrid_lf.HybridPowerFlowCalc.run = original_run

        self.assertEqual(0, rc)
        self.assertEqual([0], x_sizes_before_run)
        self.assertTrue(calc.result["summary"]["converged"])

    def test_hybrid_power_flow_uses_dc_ppc_without_dc_object_topo(self):
        import lfcore.hybrid_lf as hybrid_lf

        network = hybrid_lf._read_lf_network_from_file(ROOT / "data" / "model" / "hybrid" / "hybrid_net_40.e")
        self.assertTrue(getattr(network.dc, "_lf_lightweight", False))
        self.assertFalse(hasattr(network.dc, "topo"))
        calc = hybrid_lf.HybridPowerFlowCalc(network, verbose=False)
        calc.skip_lf_result = True
        with contextlib.redirect_stdout(io.StringIO()):
            rc = calc.run()

        self.assertEqual(0, rc)
        self.assertFalse(hasattr(calc.dc_calc, "array_mode"))
        self.assertEqual("dc_ppc_v1", calc.dc_calc.ppc["format"])

    def test_pure_ac_hybrid_lf_load_uses_lightweight_ac_fast_path(self):
        import lfcore.hybrid_lf as hybrid_lf

        original_builder = hybrid_lf.build_hybrid_ppc_with_topology_from_efile_rows
        original_ac_file_builder = hybrid_lf.build_ac_ppc_with_topology_from_e_file
        original_rows_reader = hybrid_lf._read_efile_rows
        row_reads = 0

        def reject_full_hybrid_builder(*_args, **_kwargs):
            raise AssertionError("pure AC hybrid LF should not build full hybrid object model")

        def reject_ac_file_builder(*_args, **_kwargs):
            raise AssertionError("pure AC hybrid LF should reuse already-loaded E rows")

        def counted_rows_reader(*args, **kwargs):
            nonlocal row_reads
            row_reads += 1
            return original_rows_reader(*args, **kwargs)

        hybrid_lf.build_hybrid_ppc_with_topology_from_efile_rows = reject_full_hybrid_builder
        hybrid_lf.build_ac_ppc_with_topology_from_e_file = reject_ac_file_builder
        hybrid_lf._read_efile_rows = counted_rows_reader
        try:
            network = hybrid_lf._read_lf_network_from_file(ROOT / "data" / "model" / "ac" / "ieee14.e")
        finally:
            hybrid_lf.build_hybrid_ppc_with_topology_from_efile_rows = original_builder
            hybrid_lf.build_ac_ppc_with_topology_from_e_file = original_ac_file_builder
            hybrid_lf._read_efile_rows = original_rows_reader

        self.assertTrue(getattr(network.ac, "_lf_lightweight", False))
        self.assertTrue(hasattr(network, "_ac_ppc"))
        self.assertFalse(hasattr(network, "_dc_ppc"))
        self.assertIn("_topology_arrays", network._ac_ppc)
        self.assertEqual(14, network.total_nodes)
        self.assertEqual(1, row_reads)

    def test_pure_dc_hybrid_lf_load_uses_loaded_rows_once(self):
        import lfcore.hybrid_lf as hybrid_lf

        original_builder = hybrid_lf.build_hybrid_ppc_with_topology_from_efile_rows
        original_dc_file_builder = hybrid_lf.build_dc_ppc_with_topology_from_e_file
        original_rows_reader = hybrid_lf._read_efile_rows
        row_reads = 0

        def reject_full_hybrid_builder(*_args, **_kwargs):
            raise AssertionError("pure DC hybrid LF should not build full hybrid object model")

        def reject_dc_file_builder(*_args, **_kwargs):
            raise AssertionError("pure DC hybrid LF should reuse already-loaded E rows")

        def counted_rows_reader(*args, **kwargs):
            nonlocal row_reads
            row_reads += 1
            return original_rows_reader(*args, **kwargs)

        hybrid_lf.build_hybrid_ppc_with_topology_from_efile_rows = reject_full_hybrid_builder
        hybrid_lf.build_dc_ppc_with_topology_from_e_file = reject_dc_file_builder
        hybrid_lf._read_efile_rows = counted_rows_reader
        try:
            network = hybrid_lf._read_lf_network_from_file(ROOT / "data" / "model" / "dc" / "dc_net_30.e")
        finally:
            hybrid_lf.build_hybrid_ppc_with_topology_from_efile_rows = original_builder
            hybrid_lf.build_dc_ppc_with_topology_from_e_file = original_dc_file_builder
            hybrid_lf._read_efile_rows = original_rows_reader

        self.assertTrue(getattr(network.dc, "_lf_lightweight", False))
        self.assertFalse(hasattr(network, "_ac_ppc"))
        self.assertTrue(hasattr(network, "_dc_ppc"))
        self.assertIn("_topology_arrays", network._dc_ppc)
        self.assertEqual(30, network.total_nodes)
        self.assertEqual(1, row_reads)

    def test_hybrid_topology_builds_hybrid_islands(self):
        import hybrid_net_flow

        network = hybrid_net_flow.HybridPowerNetwork.read_from_file(
            ROOT / "data" / "model" / "hybrid" / "hybrid_net_40.e"
        )

        network.topo()

        from ac_model import ACPowerNetwork
        from dc_model import DCPowerNetwork

        self.assertIsInstance(network.ac, ACPowerNetwork)
        self.assertIsInstance(network.dc, DCPowerNetwork)
        self.assertEqual(len(network.ac.islands), 1)
        self.assertEqual(len(network.dc.islands), 3)
        self.assertEqual(len(network.hybrid_islands), 1)

        island = network.hybrid_islands[0]
        self.assertIsInstance(island, hybrid_net_flow.HybridIsland)
        self.assertTrue(island.is_alive)
        self.assertEqual(len(island.ac_islands), 1)
        self.assertEqual(len(island.dc_islands), 3)
        self.assertEqual(len(island.ac_nodes), 10)
        self.assertEqual(len(island.dc_nodes), 26)
        self.assertEqual(len(island.dcac_converters), 3)
        self.assertEqual(len(island.dcdc_converters), 9)
        self.assertEqual(len(island.acac_converters), 1)
        self.assertIs(network.ac.islands[0].hybrid_isl_obj, island)
        self.assertTrue(all(dc_isl.hybrid_isl_obj is island for dc_isl in network.dc.islands))

    def test_acac_terminal_controls_use_common_ac_enum(self):
        from model.ac_model import ACACConverter, ACAC_SIDE_CONTROL_TYPES

        self.assertEqual({"PQ", "PV", "PH", "NONE"}, ACAC_SIDE_CONTROL_TYPES)

        converter = ACACConverter(
            1,
            4,
            5,
            0.01,
            0.01,
            "Q",
            "V",
            5.0,
            0.0,
            0.0,
            1.0,
            1.0,
        )
        self.assertEqual("PQ", converter.i_control_type)
        self.assertEqual("PV", converter.j_control_type)
        self.assertEqual("PQV", converter.control_type)

    def test_acac_converter_is_solved_inside_hybrid_newton_system(self):
        import hybrid_net_flow

        with tempfile.TemporaryDirectory() as tmpdir:
            case_path = Path(tmpdir) / "hybrid_acac.e"
            source_path = ROOT / "data" / "model" / "hybrid" / "hybrid_net_40.e"
            source_text = source_path.read_text(encoding="utf-8").split("<ACACConverter>")[0].rstrip()
            acac_block = """

<ACACConverter>
@ idx name i_node j_node r1 r2 i_control_type j_control_type p_set i_q_set j_q_set i_v_set j_v_set run_stat i_p i_q j_p j_q i_i j_i
# 1 acac_4_5 4 5 0.01 0.01 PQ PQ 5.0 0.0 0.0 0.0 0.0 1 0.0 0.0 0.0 0.0 0.0 0.0
</ACACConverter>
"""
            case_path.write_text(source_text + acac_block, encoding="utf-8")

            result = _run_hybrid_lf(case_path, verbose=False)

        self.assertTrue(result.converged, (result.ac_errors, result.dc_errors, result.calc.normF))
        self.assertTrue(result.has_acac)
        self.assertEqual(len(result.network.acac_converters), 1)
        self.assertEqual(result.calc.N_acac, 1)
        self.assertEqual(
            result.global_jacobian_shape[0],
            result.calc.ac_eq + result.calc.dc_eq + result.calc.N_dcac * 3 + result.calc.N_acac * 4,
        )
        self.assertEqual(result.global_jacobian_shape[0], result.global_jacobian_shape[1])

        conv = result.network.acac_converters[0]
        self.assertEqual("PQ", conv.i_control_type)
        self.assertEqual("PQ", conv.j_control_type)
        self.assertAlmostEqual(conv.i_p, 0.05, places=8)
        self.assertAlmostEqual(conv.i_q, 0.0, places=8)
        self.assertAlmostEqual(conv.j_q, 0.0, places=8)
        self.assertIsNotNone(conv.j_p)
        self.assertGreater(conv.i_i, 0.0)
        self.assertGreater(conv.j_i, 0.0)

    def test_topology_counts_cross_ac_dc_converters_as_node_references(self):
        import hybrid_net_flow

        network = hybrid_net_flow.HybridPowerNetwork.read_from_file(
            ROOT / "data" / "model" / "hybrid" / "qinling.e"
        )

        ac_warnings, ac_errors, dc_warnings, dc_errors = network.prepare(verbose=False)

        self.assertEqual(ac_errors, [])
        self.assertEqual(dc_errors, [])
        self.assertEqual(ac_warnings, [])
        self.assertEqual(dc_warnings, [])

    def test_main_hybrid_power_flow_skips_topology_diagnostics(self):
        import hybrid_net_flow
        from ac_model import ACPowerNetwork
        from dc_model import DCPowerNetwork

        original_ac_check_topo = ACPowerNetwork.check_topo
        original_dc_check_topo = DCPowerNetwork.check_topo

        def reject_check_topo(*_args, **_kwargs):
            raise AssertionError("main hybrid load-flow path should not call check_topo")

        ACPowerNetwork.check_topo = reject_check_topo
        DCPowerNetwork.check_topo = reject_check_topo
        try:
            result = _run_hybrid_lf(
                ROOT / "data" / "model" / "hybrid" / "qinling.e",
                verbose=False,
            )
        finally:
            ACPowerNetwork.check_topo = original_ac_check_topo
            DCPowerNetwork.check_topo = original_dc_check_topo

        self.assertTrue(result.converged, (result.ac_errors, result.dc_errors, result.calc.normF))

    def test_node_run_stat_zero_removes_attached_converter_from_solution(self):
        import hybrid_net_flow

        text = (ROOT / "data" / "model" / "hybrid" / "qinling.e").read_text(encoding="utf-8")
        text = _ensure_block_column(text, "ACNode", "run_stat", "1")
        text = _ensure_block_column(text, "DCNode", "run_stat", "1")
        for block_name, row_idx in (
            ("ACNode", 1),
            ("ACNode", 11),
            ("DCNode", 2),
            ("DCNode", 29),
            ("ACBranch", 1),
            ("DCBranch", 1),
            ("ACGenerator", 1),
            ("DCBreak", 1),
        ):
            text = _set_block_value(text, block_name, row_idx, "run_stat", "0")

        with tempfile.TemporaryDirectory() as tmpdir:
            case_path = Path(tmpdir) / "qinling_node_out.e"
            case_path.write_text(text, encoding="utf-8")
            result = _run_hybrid_lf(case_path, verbose=False)

        self.assertTrue(result.converged, (result.ac_errors, result.dc_errors, result.calc.normF))
        self.assertEqual(len(result.network.dcac_converters), 11)
        self.assertEqual(result.calc.N_dcac, 10)
        self.assertNotIn(result.network.dcac_converters[0], [item[0] for item in result.calc.dcac_converters])


if __name__ == "__main__":
    unittest.main()
