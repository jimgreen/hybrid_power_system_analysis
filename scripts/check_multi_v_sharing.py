import argparse
import copy
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hybrid_power_system_analysis.lfcore.ac_lf import ACPowerFlowCalc, load_ac_ppc_from_e_file
from hybrid_power_system_analysis.lfcore.dc_lf import DCPowerFlowCalc, load_dc_ppc_from_e_file
from hybrid_power_system_analysis.lfcore.hybrid_lf import HybridPowerFlowCalc, _read_lf_network_from_file
from hybrid_power_system_analysis.model.ac_array_model import (
    BUS_COLS as AC_BUS_COLS,
    CTRL_PV,
    CTRL_SLACK,
    GEN_COLS as AC_GEN_COLS,
)
from hybrid_power_system_analysis.model.dc_array_model import (
    BUS_COLS as DC_BUS_COLS,
    CTRL_NONE as DC_CTRL_NONE,
    CTRL_V as DC_CTRL_V,
    DCDC_COLS as DC_DCDC_COLS,
    GEN_COLS as DC_GEN_COLS,
)
from hybrid_power_system_analysis.model.hybrid_array_model import (
    DCAC_AC_CONTROL_CODE,
    DCAC_COLS,
    DCAC_DC_CONTROL_CODE,
)


def append_row(table: np.ndarray, row: np.ndarray) -> np.ndarray:
    return np.vstack([table, row.reshape(1, -1)])


def append_name(names: Iterable[object], name: str) -> np.ndarray:
    base = np.asarray(list(names), dtype=object)
    return np.concatenate([base, np.asarray([name], dtype=object)])


def find_bus_row(bus: np.ndarray, idx_col: int, node_idx: int) -> int:
    rows = np.flatnonzero(bus[:, idx_col].astype(np.int64, copy=False) == int(node_idx))
    return int(rows[0]) if rows.size else -1


def ratio_text(values: np.ndarray) -> str:
    vals = np.asarray(values, dtype=float)
    if vals.size < 2:
        return "-"
    base = vals[0]
    if not np.isfinite(base) or abs(base) < 1e-12:
        return str(np.round(vals, 6).tolist())
    parts = []
    for value in vals:
        if not np.isfinite(value):
            parts.append("nan")
        else:
            parts.append(f"{value / base:.3f}")
    return " : ".join(parts)


def print_case_header(title: str) -> None:
    print("=" * 90)
    print(title)


def print_case_result(rc: int, converged: bool, iterations: int, norm_f: float) -> None:
    print(f"rc={rc}, converged={converged}, iter={iterations}, normF={norm_f:.6e}")


def run_ac_dual_pv_case(root: Path) -> bool:
    print_case_header("AC stable case: same-bus dual PV, average V + Q sharing")
    ppc = copy.deepcopy(load_ac_ppc_from_e_file(root / "data/model/ac/ieee1w.e"))
    gen = ppc["gen"].copy()
    row = int(np.flatnonzero(gen[:, AC_GEN_COLS["control_type"]].astype(np.int64) == CTRL_PV)[0])
    node = int(gen[row, AC_GEN_COLS["node"]])

    gen[row, AC_GEN_COLS["v_set"]] = 1.00
    gen[row, AC_GEN_COLS["alpha"]] = 1.0

    new_row = gen[row].copy()
    new_row[AC_GEN_COLS["idx"]] = np.max(gen[:, AC_GEN_COLS["idx"]]) + 1000
    new_row[AC_GEN_COLS["v_set"]] = 1.04
    new_row[AC_GEN_COLS["alpha"]] = 3.0
    gen = append_row(gen, new_row)

    ppc["gen"] = gen
    ppc["gen_name"] = append_name(ppc.get("gen_name", []), "pv_dup")

    calc = ACPowerFlowCalc(ppc, linear_solver="pyklu", result_mode="full", verbose=False)
    rc = calc.run()
    print_case_result(rc, calc.converged, calc.iterations, float(calc.normF))
    if rc != 0:
        return False

    res = calc.result
    sub = res["gen"][res["gen"][:, AC_GEN_COLS["node"]].astype(np.int64) == node]
    bus_row = find_bus_row(res["bus"], AC_BUS_COLS["idx"], node)
    print(f"node={node}, gen_idx={sub[:, AC_GEN_COLS['idx']].astype(int).tolist()}")
    print(
        "v_set=",
        np.round(sub[:, AC_GEN_COLS["v_set"]], 6).tolist(),
        "avg=",
        round(float(np.mean(sub[:, AC_GEN_COLS["v_set"]])), 6),
        "bus_V=",
        round(float(res["bus"][bus_row, AC_BUS_COLS["voltage"]]), 6),
    )
    print("P=", np.round(sub[:, AC_GEN_COLS["p"]], 6).tolist(), "(PV keeps p_set)")
    print(
        "Q=",
        np.round(sub[:, AC_GEN_COLS["q"]], 6).tolist(),
        "Q-share=",
        ratio_text(sub[:, AC_GEN_COLS["q"]]),
        "(expect about 1:3)",
    )
    return True


def run_dc_dual_v_gen_case(root: Path) -> bool:
    print_case_header("DC stable case: same-bus dual V generator, average V + equal P sharing")
    ppc = copy.deepcopy(load_dc_ppc_from_e_file(root / "data/model/dc/dc_net_1w.e"))
    gen = ppc["gen"].copy()
    row = int(np.flatnonzero(gen[:, DC_GEN_COLS["control_type"]].astype(np.int64) == DC_CTRL_V)[0])
    node = int(gen[row, DC_GEN_COLS["node"]])

    gen[row, DC_GEN_COLS["v_set"]] = 0.98
    new_row = gen[row].copy()
    new_row[DC_GEN_COLS["idx"]] = np.max(gen[:, DC_GEN_COLS["idx"]]) + 1002
    new_row[DC_GEN_COLS["v_set"]] = 1.02
    gen = append_row(gen, new_row)

    ppc["gen"] = gen
    ppc["gen_name"] = append_name(ppc.get("gen_name", []), "dc_v_dup")

    calc = DCPowerFlowCalc(ppc, linear_solver="pyklu", result_mode="full", verbose=False)
    rc = calc.run()
    print_case_result(rc, calc.converged, calc.iterations, float(calc.normF))
    if rc != 0:
        return False

    res = calc.result
    sub = res["gen"][res["gen"][:, DC_GEN_COLS["node"]].astype(np.int64) == node]
    bus_row = find_bus_row(res["bus"], DC_BUS_COLS["idx"], node)
    print(f"node={node}, gen_idx={sub[:, DC_GEN_COLS['idx']].astype(int).tolist()}")
    print(
        "v_set=",
        np.round(sub[:, DC_GEN_COLS["v_set"]], 6).tolist(),
        "avg=",
        round(float(np.mean(sub[:, DC_GEN_COLS["v_set"]])), 6),
        "bus_V=",
        round(float(res["bus"][bus_row, DC_BUS_COLS["voltage"]]), 6),
    )
    print(
        "P=",
        np.round(sub[:, DC_GEN_COLS["p"]], 6).tolist(),
        "P-share=",
        ratio_text(sub[:, DC_GEN_COLS["p"]]),
        "(expect about 1:1)",
    )
    return True


def run_hybrid_dual_dcac_acv_case(root: Path) -> bool:
    print_case_header("Hybrid stable case: same-AC-bus dual DCAC ACV, average V + equal P/Q sharing")
    network = _read_lf_network_from_file(root / "data/model/hybrid/qinling.e")
    ppc = network.ppc
    ppc["dcac"] = ppc["dcac"].copy()

    row = 0
    ac_node = int(ppc["dcac"][row, DCAC_COLS["ac_node"]])
    ppc["dcac"][row, DCAC_COLS["ac_control_type"]] = DCAC_AC_CONTROL_CODE["PH"]
    ppc["dcac"][row, DCAC_COLS["dc_control_type"]] = DCAC_DC_CONTROL_CODE["NONE"]
    ppc["dcac"][row, DCAC_COLS["v_ac_set"]] = 0.99
    ppc["dcac"][row, DCAC_COLS["q_ac_set"]] = 0.0

    new_row = ppc["dcac"][row].copy()
    new_row[DCAC_COLS["idx"]] = np.max(ppc["dcac"][:, DCAC_COLS["idx"]]) + 1010
    new_row[DCAC_COLS["v_ac_set"]] = 1.03
    ppc["dcac"] = append_row(ppc["dcac"], new_row)
    ppc["dcac_name"] = append_name(ppc.get("dcac_name", []), "synth_acv_dup")

    calc = HybridPowerFlowCalc(network, linear_solver="scipy", result_mode="full", verbose=False)
    rc = calc.run()
    print_case_result(rc, calc.converged, calc.iterations, float(calc.normF))
    if rc != 0:
        return False

    full_dcac = network.ppc["dcac"]
    sub = full_dcac[full_dcac[:, DCAC_COLS["ac_node"]].astype(np.int64) == ac_node]
    ac_bus = calc.result["ac"]["bus"]
    bus_row = find_bus_row(ac_bus, AC_BUS_COLS["idx"], ac_node)
    print(f"ac_node={ac_node}, dcac_idx={sub[:, DCAC_COLS['idx']].astype(int).tolist()}")
    print(
        "v_ac_set=",
        np.round(sub[:, DCAC_COLS["v_ac_set"]], 6).tolist(),
        "avg=",
        round(float(np.mean(sub[:, DCAC_COLS["v_ac_set"]])), 6),
        "bus_V=",
        round(float(ac_bus[bus_row, AC_BUS_COLS["voltage"]]), 6),
    )
    print(
        "ac_p=",
        np.round(sub[:, DCAC_COLS["ac_p"]], 6).tolist(),
        "share=",
        ratio_text(sub[:, DCAC_COLS["ac_p"]]),
        "(expect about 1:1)",
    )
    print(
        "ac_q=",
        np.round(sub[:, DCAC_COLS["ac_q"]], 6).tolist(),
        "share=",
        ratio_text(sub[:, DCAC_COLS["ac_q"]]),
        "(expect about 1:1)",
    )
    return True


def run_experimental_ac_dual_slack_case(root: Path) -> bool:
    print_case_header("Experimental AC case: same-bus dual slack")
    ppc = copy.deepcopy(load_ac_ppc_from_e_file(root / "data/model/ac/ieee1w.e"))
    gen = ppc["gen"].copy()
    row = int(np.flatnonzero(gen[:, AC_GEN_COLS["control_type"]].astype(np.int64) == CTRL_SLACK)[0])
    node = int(gen[row, AC_GEN_COLS["node"]])

    gen[row, AC_GEN_COLS["v_set"]] = 1.00
    gen[row, AC_GEN_COLS["alpha"]] = 1.0

    new_row = gen[row].copy()
    new_row[AC_GEN_COLS["idx"]] = np.max(gen[:, AC_GEN_COLS["idx"]]) + 1001
    new_row[AC_GEN_COLS["v_set"]] = 1.04
    new_row[AC_GEN_COLS["alpha"]] = 3.0
    gen = append_row(gen, new_row)

    ppc["gen"] = gen
    ppc["gen_name"] = append_name(ppc.get("gen_name", []), "slack_dup")

    calc = ACPowerFlowCalc(ppc, linear_solver="pyklu", result_mode="full", verbose=False)
    rc = calc.run()
    print_case_result(rc, calc.converged, calc.iterations, float(calc.normF))
    res = calc.result
    sub = res["gen"][res["gen"][:, AC_GEN_COLS["node"]].astype(np.int64) == node]
    bus_row = find_bus_row(res["bus"], AC_BUS_COLS["idx"], node)
    print(f"node={node}, gen_idx={sub[:, AC_GEN_COLS['idx']].astype(int).tolist()}")
    print(
        "v_set=",
        np.round(sub[:, AC_GEN_COLS["v_set"]], 6).tolist(),
        "avg=",
        round(float(np.mean(sub[:, AC_GEN_COLS["v_set"]])), 6),
        "bus_V=",
        round(float(res["bus"][bus_row, AC_BUS_COLS["voltage"]]), 6),
    )
    print("P=", np.round(sub[:, AC_GEN_COLS["p"]], 6).tolist(), "P-share=", ratio_text(sub[:, AC_GEN_COLS["p"]]))
    print("Q=", np.round(sub[:, AC_GEN_COLS["q"]], 6).tolist(), "Q-share=", ratio_text(sub[:, AC_GEN_COLS["q"]]))
    return rc == 0


def run_experimental_dc_dcdc_v_case(root: Path) -> bool:
    print_case_header("Experimental DC case: synthetic same-bus dual DCDC V side")
    ppc = copy.deepcopy(load_dc_ppc_from_e_file(root / "data/model/dc/dc_net_30.e"))
    dcdc = ppc["dcdc"].copy()

    row = 0
    node = int(dcdc[row, DC_DCDC_COLS["i_node"]])
    dcdc[row, DC_DCDC_COLS["i_control_type"]] = DC_CTRL_V
    dcdc[row, DC_DCDC_COLS["j_control_type"]] = DC_CTRL_NONE
    dcdc[row, DC_DCDC_COLS["v_set"]] = 0.99

    new_row = dcdc[row].copy()
    new_row[DC_DCDC_COLS["idx"]] = np.max(dcdc[:, DC_DCDC_COLS["idx"]]) + 1003
    new_row[DC_DCDC_COLS["v_set"]] = 1.03
    dcdc = append_row(dcdc, new_row)

    ppc["dcdc"] = dcdc
    ppc["dcdc_name"] = append_name(ppc.get("dcdc_name", []), "synth_v_dup")

    calc = DCPowerFlowCalc(ppc, linear_solver="pyklu", result_mode="full", verbose=False)
    rc = calc.run()
    print_case_result(rc, calc.converged, calc.iterations, float(calc.normF))
    if rc == 0:
        res = calc.result
        sub = res["dcdc"][res["dcdc"][:, DC_DCDC_COLS["i_node"]].astype(np.int64) == node]
        bus_row = find_bus_row(res["bus"], DC_BUS_COLS["idx"], node)
        print(f"node={node}, dcdc_idx={sub[:, DC_DCDC_COLS['idx']].astype(int).tolist()}")
        print(
            "v_set=",
            np.round(sub[:, DC_DCDC_COLS["v_set"]], 6).tolist(),
            "avg=",
            round(float(np.mean(sub[:, DC_DCDC_COLS["v_set"]])), 6),
            "bus_V=",
            round(float(res["bus"][bus_row, DC_BUS_COLS["voltage"]]), 6),
        )
        print(
            "i_p=",
            np.round(sub[:, DC_DCDC_COLS["i_p"]], 6).tolist(),
            "share=",
            ratio_text(sub[:, DC_DCDC_COLS["i_p"]]),
        )
    return rc == 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Check multi-device same-bus voltage-control averaging and power sharing."
    )
    parser.add_argument(
        "--include-experimental",
        action="store_true",
        help="Also run aggressive synthetic cases that may be physically infeasible and fail to converge.",
    )
    args = parser.parse_args(argv)

    ok = True
    ok &= run_ac_dual_pv_case(ROOT_DIR)
    ok &= run_dc_dual_v_gen_case(ROOT_DIR)
    ok &= run_hybrid_dual_dcac_acv_case(ROOT_DIR)

    if args.include_experimental:
        ok &= run_experimental_ac_dual_slack_case(ROOT_DIR)
        ok &= run_experimental_dc_dcdc_v_case(ROOT_DIR)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
