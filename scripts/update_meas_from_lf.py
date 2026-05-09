"""Rewrite measurement values from solved load-flow results.

The E and MEAS files store named values, while the load-flow solvers work in
per-unit internally.  This utility solves each matching E file, converts the
solved P/Q/V/I values back to file units, and rewrites only the Measurement
value column.  Weight and valid flags are intentionally preserved.
"""

from __future__ import annotations

import contextlib
import io
import math
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src" / "hybrid_power_system_analysis"
MODEL_DIR = SRC_DIR / "model"
LFCORE_DIR = SRC_DIR / "lfcore"
for path in (SRC_DIR, MODEL_DIR, LFCORE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ac_lf import ACPowerFlowCalc
from ac_model import ACPowerNetwork
from dc_lf import DCPowerFlowCalc
from dc_array_model import DCPowerNetwork
from hybrid_lf import run_hybrid_power_flow
from unit_system import ac_current_base_ka, dc_current_base_ka


MEAS_HEADER = ("idx", "name", "dev_type", "dev_name", "meas_type", "weight", "valid", "value")
VALUE_TYPES = {
    "P",
    "Q",
    "I",
    "V",
    "P_FROM",
    "Q_FROM",
    "I_FROM",
    "V_FROM",
    "P_TO",
    "Q_TO",
    "I_TO",
    "V_TO",
    "P_GEN",
    "Q_GEN",
    "I_GEN",
    "V_GEN",
    "P_LOAD",
    "Q_LOAD",
    "I_LOAD",
    "V_LOAD",
    "P_DC",
    "I_DC",
    "V_DC",
    "P_AC",
    "Q_AC",
    "I_AC",
    "V_AC",
    "V_DIFF",
}
ANGLE_TYPES = {"ANGLE", "THETA", "ANGLE_DIFF", "THETA_DIFF"}


class Snapshot:
    """Measurement evaluator over one solved AC, DC, or hybrid network."""

    def __init__(self, root, ac_grid=None, dc_grid=None, dcac_converters=None, acac_converters=None):
        self.root = root
        self.ac = ac_grid
        self.dc = dc_grid
        self.dcac_converters = list(dcac_converters or [])
        self.acac_converters = list(acac_converters or [])
        self.p_base = float(root.p_base)
        self.p_base_kW = float(root.p_base_kW)
        self.u_scale = float(root.u_scale)
        self.i_scale = float(root.i_scale)

        self.ac_nodes = self._by_name(getattr(ac_grid, "nodes", []))
        self.dc_nodes = self._by_name(getattr(dc_grid, "nodes", []))
        self.ac_nodes_by_idx = {dev.idx: dev for dev in getattr(ac_grid, "nodes", [])}
        self.dc_nodes_by_idx = {dev.idx: dev for dev in getattr(dc_grid, "nodes", [])}

        self.ac_devices = {
            "ACBranch": self._by_name(getattr(ac_grid, "branches", [])),
            "ACTransformer": self._by_name(getattr(ac_grid, "transformers", [])),
            "ACSwitch": self._by_name(getattr(ac_grid, "switches", [])),
            "ACBreak": self._by_name(getattr(ac_grid, "breakers", [])),
            "ACZeroBranch": self._by_name(getattr(ac_grid, "zero_branches", [])),
            "ACGenerator": self._by_name(getattr(ac_grid, "generators", [])),
            "ACLoad": self._by_name(getattr(ac_grid, "loads", [])),
        }
        self.dc_devices = {
            "DCBranch": self._by_name(getattr(dc_grid, "branches", [])),
            "DCSwitch": self._by_name(getattr(dc_grid, "switches", [])),
            "DCBreak": self._by_name(getattr(dc_grid, "breakers", [])),
            "DCZeroBranch": self._by_name(getattr(dc_grid, "zero_branches", [])),
            "DCDCConverter": self._by_name(getattr(dc_grid, "dcdc_converters", [])),
            "DCGenerator": self._by_name(getattr(dc_grid, "generators", [])),
            "DCLoad": self._by_name(getattr(dc_grid, "loads", [])),
        }
        self.dcac_by_name = self._by_name(self.dcac_converters)
        self.acac_by_name = self._by_name(self.acac_converters)

    @staticmethod
    def _by_name(devices: Iterable) -> Dict[str, object]:
        return {str(dev.name): dev for dev in devices if hasattr(dev, "name")}

    @staticmethod
    def _float(value, default: float = 0.0) -> float:
        if value is None:
            return default
        return float(value)

    @staticmethod
    def _terminal_node(device, side: str):
        return getattr(device, "i_node_obj", None) if side == "from" else getattr(device, "j_node_obj", None)

    def _ac_terminal_node(self, device, side: str):
        node = self._terminal_node(device, side)
        if node is not None:
            return node
        attr = "i_node" if side == "from" else "j_node"
        return self.ac_nodes_by_idx.get(getattr(device, attr, None))

    def _dc_terminal_node(self, device, side: str):
        node = self._terminal_node(device, side)
        if node is not None:
            return node
        attr = "i_node" if side == "from" else "j_node"
        return self.dc_nodes_by_idx.get(getattr(device, attr, None))

    def power_to_file(self, value: float) -> float:
        return float(value) * self.p_base

    def ac_voltage_to_file(self, node) -> float:
        return self._float(node.voltage) * self.u_scale * self._float(node.vbase, 1.0)

    def dc_voltage_to_file(self, node) -> float:
        return self._float(node.voltage) * self.u_scale * self._float(node.vbase, 1.0)

    def ac_current_to_file(self, node, current: float) -> float:
        base = ac_current_base_ka(self.p_base_kW, self._float(node.vbase, 1.0))
        return float(current) * self.i_scale * base

    def dc_current_to_file(self, node, current: float) -> float:
        base = dc_current_base_ka(self.p_base_kW, self._float(node.vbase, 1.0))
        return float(current) * self.i_scale * base

    def value(self, dev_type: str, dev_name: str, meas_type: str) -> Optional[float]:
        meas_type = meas_type.upper()
        if meas_type in ANGLE_TYPES:
            return self._angle_value(dev_type, dev_name, meas_type)
        if meas_type not in VALUE_TYPES:
            return None
        if dev_type == "ACNode":
            node = self.ac_nodes.get(dev_name)
            return None if node is None or meas_type != "V" else self.ac_voltage_to_file(node)
        if dev_type == "DCNode":
            node = self.dc_nodes.get(dev_name)
            return None if node is None or meas_type != "V" else self.dc_voltage_to_file(node)
        if dev_type in ("ACBranch", "ACTransformer"):
            dev = self.ac_devices[dev_type].get(dev_name)
            return None if dev is None else self._ac_line_value(dev, meas_type)
        if dev_type in ("ACSwitch", "ACBreak", "ACZeroBranch"):
            dev = self.ac_devices[dev_type].get(dev_name)
            return None if dev is None else self._ac_zero_value(dev, meas_type)
        if dev_type == "ACGenerator":
            dev = self.ac_devices[dev_type].get(dev_name)
            return None if dev is None else self._ac_generator_value(dev, meas_type)
        if dev_type == "ACLoad":
            dev = self.ac_devices[dev_type].get(dev_name)
            return None if dev is None else self._ac_load_value(dev, meas_type)
        if dev_type == "DCBranch":
            dev = self.dc_devices[dev_type].get(dev_name)
            return None if dev is None else self._dc_line_value(dev, meas_type)
        if dev_type in ("DCSwitch", "DCBreak", "DCZeroBranch"):
            dev = self.dc_devices[dev_type].get(dev_name)
            return None if dev is None else self._dc_zero_value(dev, meas_type)
        if dev_type == "DCDCConverter":
            dev = self.dc_devices[dev_type].get(dev_name)
            return None if dev is None else self._dcdc_value(dev, meas_type)
        if dev_type == "DCGenerator":
            dev = self.dc_devices[dev_type].get(dev_name)
            return None if dev is None else self._dc_generator_value(dev, meas_type)
        if dev_type == "DCLoad":
            dev = self.dc_devices[dev_type].get(dev_name)
            return None if dev is None else self._dc_load_value(dev, meas_type)
        if dev_type == "DCACConverter":
            dev = self.dcac_by_name.get(dev_name)
            return None if dev is None else self._dcac_value(dev, meas_type)
        if dev_type == "ACACConverter":
            dev = self.acac_by_name.get(dev_name)
            return None if dev is None else self._acac_value(dev, meas_type)
        return None

    def _angle_value(self, dev_type: str, dev_name: str, meas_type: str) -> Optional[float]:
        if dev_type == "ACNode" and meas_type in ("ANGLE", "THETA"):
            node = self.ac_nodes.get(dev_name)
            return None if node is None else math.degrees(self._float(node.angle))
        if dev_type in ("ACSwitch", "ACZeroBranch") and meas_type in ("ANGLE_DIFF", "THETA_DIFF"):
            dev = self.ac_devices[dev_type].get(dev_name)
            if dev is None or dev.i_node_obj is None or dev.j_node_obj is None:
                return None
            return math.degrees(self._float(dev.i_node_obj.angle) - self._float(dev.j_node_obj.angle))
        return None

    def _ac_line_value(self, dev, meas_type: str) -> Optional[float]:
        i_node = self._ac_terminal_node(dev, "from")
        j_node = self._ac_terminal_node(dev, "to")
        if meas_type == "P_FROM":
            return self.power_to_file(self._float(dev.i_p))
        if meas_type == "Q_FROM":
            return self.power_to_file(self._float(dev.i_q))
        if meas_type == "V_FROM":
            if i_node is None:
                return None
            return self.ac_voltage_to_file(i_node)
        if meas_type == "I_FROM":
            if i_node is None:
                return None
            return self.ac_current_to_file(i_node, self._float(dev.i_c))
        if meas_type == "P_TO":
            return self.power_to_file(self._float(dev.j_p))
        if meas_type == "Q_TO":
            return self.power_to_file(self._float(dev.j_q))
        if meas_type == "V_TO":
            if j_node is None:
                return None
            return self.ac_voltage_to_file(j_node)
        if meas_type == "I_TO":
            if j_node is None:
                return None
            return self.ac_current_to_file(j_node, self._float(dev.j_c))
        return None

    def _ac_zero_value(self, dev, meas_type: str) -> Optional[float]:
        i_node = self._ac_terminal_node(dev, "from")
        j_node = self._ac_terminal_node(dev, "to")
        p_from = self._float(getattr(dev, "p", 0.0))
        q_from = self._float(getattr(dev, "q", 0.0))
        current_abs = abs(self._float(getattr(dev, "current", 0.0)))
        if meas_type == "P_FROM":
            return self.power_to_file(p_from)
        if meas_type == "Q_FROM":
            return self.power_to_file(q_from)
        if meas_type == "V_FROM":
            return None if i_node is None else self.ac_voltage_to_file(i_node)
        if meas_type == "I_FROM":
            return None if i_node is None else self.ac_current_to_file(i_node, current_abs)
        if meas_type == "P_TO":
            return self.power_to_file(-p_from)
        if meas_type == "Q_TO":
            return self.power_to_file(-q_from)
        if meas_type == "V_TO":
            return None if j_node is None else self.ac_voltage_to_file(j_node)
        if meas_type == "I_TO":
            return None if j_node is None else self.ac_current_to_file(j_node, current_abs)
        if meas_type == "V_DIFF":
            if i_node is None or j_node is None:
                return None
            return self.ac_voltage_to_file(i_node) - self.ac_voltage_to_file(j_node)
        return None

    def _ac_generator_value(self, dev, meas_type: str) -> Optional[float]:
        node = getattr(dev, "node_obj", None) or self.ac_nodes_by_idx.get(getattr(dev, "node", None))
        if meas_type == "P_GEN":
            return self.power_to_file(self._float(dev.p))
        if meas_type == "Q_GEN":
            return self.power_to_file(self._float(dev.q))
        if meas_type == "V_GEN":
            return None if node is None else self.ac_voltage_to_file(node)
        if meas_type == "I_GEN":
            return None if node is None else self.ac_current_to_file(node, self._float(dev.current))
        return None

    def _ac_load_value(self, dev, meas_type: str) -> Optional[float]:
        node = getattr(dev, "node_obj", None) or self.ac_nodes_by_idx.get(getattr(dev, "node", None))
        if meas_type == "P_LOAD":
            return self.power_to_file(self._float(dev.p))
        if meas_type == "Q_LOAD":
            return self.power_to_file(self._float(dev.q))
        if meas_type == "V_LOAD":
            return None if node is None else self.ac_voltage_to_file(node)
        if meas_type == "I_LOAD":
            return None if node is None else self.ac_current_to_file(node, self._float(dev.current))
        return None

    def _dc_line_value(self, dev, meas_type: str) -> Optional[float]:
        i_node = self._dc_terminal_node(dev, "from")
        j_node = self._dc_terminal_node(dev, "to")
        current = self._float(getattr(dev, "current", 0.0))
        if meas_type == "P_FROM":
            return self.power_to_file(self._float(dev.i_p))
        if meas_type == "V_FROM":
            return None if i_node is None else self.dc_voltage_to_file(i_node)
        if meas_type == "I_FROM":
            return None if i_node is None else self.dc_current_to_file(i_node, current)
        if meas_type == "P_TO":
            return self.power_to_file(self._float(dev.j_p))
        if meas_type == "V_TO":
            return None if j_node is None else self.dc_voltage_to_file(j_node)
        if meas_type == "I_TO":
            return None if j_node is None else self.dc_current_to_file(j_node, -current)
        return None

    def _dc_zero_value(self, dev, meas_type: str) -> Optional[float]:
        i_node = self._dc_terminal_node(dev, "from")
        j_node = self._dc_terminal_node(dev, "to")
        current = self._float(getattr(dev, "current", 0.0))
        if meas_type == "P_FROM":
            return self.power_to_file(self._float(getattr(dev, "p", 0.0)))
        if meas_type == "V_FROM":
            return None if i_node is None else self.dc_voltage_to_file(i_node)
        if meas_type == "I_FROM":
            return None if i_node is None else self.dc_current_to_file(i_node, current)
        if meas_type == "P_TO":
            if j_node is None:
                return None
            return self.power_to_file(-self._float(j_node.voltage) * current)
        if meas_type == "V_TO":
            return None if j_node is None else self.dc_voltage_to_file(j_node)
        if meas_type == "I_TO":
            return None if j_node is None else self.dc_current_to_file(j_node, -current)
        if meas_type == "V_DIFF":
            if i_node is None or j_node is None:
                return None
            return self.dc_voltage_to_file(i_node) - self.dc_voltage_to_file(j_node)
        return None

    def _dcdc_value(self, dev, meas_type: str) -> Optional[float]:
        i_node = self._dc_terminal_node(dev, "from")
        j_node = self._dc_terminal_node(dev, "to")
        if meas_type == "P_FROM":
            return self.power_to_file(self._float(dev.i_p))
        if meas_type == "V_FROM":
            return None if i_node is None else self.dc_voltage_to_file(i_node)
        if meas_type == "I_FROM":
            return None if i_node is None else self.dc_current_to_file(i_node, self._float(dev.i_c))
        if meas_type == "P_TO":
            return self.power_to_file(self._float(dev.j_p))
        if meas_type == "V_TO":
            return None if j_node is None else self.dc_voltage_to_file(j_node)
        if meas_type == "I_TO":
            return None if j_node is None else self.dc_current_to_file(j_node, self._float(dev.j_c))
        return None

    def _dc_generator_value(self, dev, meas_type: str) -> Optional[float]:
        node = getattr(dev, "node_obj", None) or self.dc_nodes_by_idx.get(getattr(dev, "node", None))
        if meas_type == "P_GEN":
            return self.power_to_file(self._float(dev.p))
        if meas_type == "V_GEN":
            return None if node is None else self.dc_voltage_to_file(node)
        if meas_type == "I_GEN":
            return None if node is None else self.dc_current_to_file(node, self._float(dev.current))
        return None

    def _dc_load_value(self, dev, meas_type: str) -> Optional[float]:
        node = getattr(dev, "node_obj", None) or self.dc_nodes_by_idx.get(getattr(dev, "node", None))
        if meas_type == "P_LOAD":
            return self.power_to_file(self._float(dev.p))
        if meas_type == "V_LOAD":
            return None if node is None else self.dc_voltage_to_file(node)
        if meas_type == "I_LOAD":
            return None if node is None else self.dc_current_to_file(node, self._float(dev.current))
        return None

    def _dcac_value(self, dev, meas_type: str) -> Optional[float]:
        if meas_type == "P_DC":
            return self.power_to_file(self._float(dev.dc_p))
        if meas_type == "V_DC":
            return self.dc_voltage_to_file(dev.dc_node_obj)
        if meas_type == "I_DC":
            return self.dc_current_to_file(dev.dc_node_obj, self._float(dev.dc_i))
        if meas_type == "P_AC":
            return self.power_to_file(self._float(dev.ac_p))
        if meas_type == "Q_AC":
            return self.power_to_file(self._float(dev.ac_q))
        if meas_type == "V_AC":
            return self.ac_voltage_to_file(dev.ac_node_obj)
        if meas_type == "I_AC":
            return self.ac_current_to_file(dev.ac_node_obj, self._float(dev.ac_i))
        return None

    def _acac_value(self, dev, meas_type: str) -> Optional[float]:
        if meas_type == "P_FROM":
            return self.power_to_file(self._float(dev.i_p))
        if meas_type == "Q_FROM":
            return self.power_to_file(self._float(dev.i_q))
        if meas_type == "V_FROM":
            return self.ac_voltage_to_file(dev.i_node_obj)
        if meas_type == "I_FROM":
            return self.ac_current_to_file(dev.i_node_obj, self._float(dev.i_i))
        if meas_type == "P_TO":
            return self.power_to_file(self._float(dev.j_p))
        if meas_type == "Q_TO":
            return self.power_to_file(self._float(dev.j_q))
        if meas_type == "V_TO":
            return self.ac_voltage_to_file(dev.j_node_obj)
        if meas_type == "I_TO":
            return self.ac_current_to_file(dev.j_node_obj, self._float(dev.j_i))
        return None


def solve_ac(e_file: Path) -> Tuple[Snapshot, str]:
    network = ACPowerNetwork()
    network.read_from_file(e_file)
    network.topo()
    calc = ACPowerFlowCalc(network)
    with contextlib.redirect_stdout(io.StringIO()):
        calc.prepare()
        rc = calc.run()
    if rc != 0 or not calc.converged:
        raise RuntimeError(f"AC load flow failed for {e_file}: rc={rc}, iter={calc.iterations}, normF={calc.normF:.3e}")
    return Snapshot(network, ac_grid=network), f"iter={calc.iterations}, normF={calc.normF:.3e}"


def solve_dc(e_file: Path) -> Tuple[Snapshot, str]:
    network = DCPowerNetwork()
    network.read_from_file(e_file)
    network.topo()
    calc = DCPowerFlowCalc(network)
    with contextlib.redirect_stdout(io.StringIO()):
        rc = calc.run()
    if rc != 0 or not calc.converged:
        raise RuntimeError(f"DC load flow failed for {e_file}: rc={rc}, iter={calc.iterations}, normF={calc.normF:.3e}")
    return Snapshot(network, dc_grid=network), f"iter={calc.iterations}, normF={calc.normF:.3e}"


def solve_hybrid(e_file: Path) -> Tuple[Snapshot, str]:
    with contextlib.redirect_stdout(io.StringIO()):
        result = run_hybrid_power_flow(e_file, verbose=False)
    if not result.converged:
        raise RuntimeError(
            f"Hybrid load flow failed for {e_file}: rc={result.rc}, "
            f"iter={result.calc.iterations}, normF={result.calc.normF:.3e}, "
            f"ac_errors={result.ac_errors}, dc_errors={result.dc_errors}"
        )
    network = result.network
    return (
        Snapshot(
            network,
            ac_grid=network.ac,
            dc_grid=network.dc,
            dcac_converters=network.dcac_converters,
            acac_converters=network.acac_converters,
        ),
        f"iter={result.calc.iterations}, normF={result.calc.normF:.3e}",
    )


def format_number(value: float) -> str:
    if not math.isfinite(value):
        return str(value)
    if abs(value) < 5e-13:
        value = 0.0
    text = f"{value:.10g}"
    return "0" if text == "-0" else text


def parse_measurement_rows(meas_file: Path) -> Tuple[List[str], List[List[str]], List[str]]:
    before: List[str] = []
    rows: List[List[str]] = []
    after: List[str] = []
    in_measurement = False
    seen_measurement = False
    with meas_file.open("rt", encoding="utf-8") as fp:
        for raw_line in fp:
            line = raw_line.strip()
            if line == "<Measurement>":
                in_measurement = True
                seen_measurement = True
                continue
            if line == "</Measurement>":
                in_measurement = False
                continue
            if not in_measurement:
                if not seen_measurement:
                    before.append(raw_line.rstrip("\n"))
                elif line:
                    after.append(raw_line.rstrip("\n"))
                continue
            if not line or line.startswith("@"):
                continue
            if line.startswith("#"):
                parts = line[1:].split()
                if len(parts) != len(MEAS_HEADER):
                    raise RuntimeError(f"Invalid measurement row in {meas_file}: {line}")
                rows.append(parts)
    if not seen_measurement:
        raise RuntimeError(f"{meas_file} does not contain a <Measurement> block")
    return before, rows, after


def render_measurement_file(before: Sequence[str], rows: Sequence[Sequence[str]], after: Sequence[str]) -> str:
    widths = [len(header) for header in MEAS_HEADER]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(str(cell)))

    parts: List[str] = []
    parts.extend(line + "\n" for line in before if line)
    parts.append("<Measurement>\n")
    header_line = "@ " + "  ".join(f"{MEAS_HEADER[idx]:<{widths[idx]}}" for idx in range(len(MEAS_HEADER))).rstrip()
    parts.append(header_line + "\n")
    for row in rows:
        formatted = ["#"]
        for idx, cell in enumerate(row):
            text = str(cell)
            if idx in (0, 6):
                formatted.append(f"{text:>{widths[idx]}}")
            elif idx in (5, 7):
                formatted.append(f"{text:>{widths[idx]}}")
            else:
                formatted.append(f"{text:<{widths[idx]}}")
        parts.append(" ".join(formatted).rstrip() + "\n")
    parts.append("</Measurement>\n")
    parts.extend(line + "\n" for line in after if line)
    return "".join(parts)


def rewrite_measurements(meas_file: Path, snapshot: Snapshot) -> Tuple[int, int]:
    before, rows, after = parse_measurement_rows(meas_file)
    updated = 0
    missing = 0
    for row in rows:
        dev_type, dev_name, meas_type = row[2], row[3], row[4].upper()
        value = snapshot.value(dev_type, dev_name, meas_type)
        if value is None:
            if meas_type in VALUE_TYPES or meas_type in ANGLE_TYPES:
                missing += 1
            continue
        row[7] = format_number(value)
        updated += 1
    meas_file.write_text(render_measurement_file(before, rows, after), encoding="utf-8")
    return updated, missing


def matching_e_file(meas_file: Path) -> Optional[Path]:
    try:
        rel = meas_file.resolve().relative_to(ROOT_DIR / "data" / "meas")
    except ValueError:
        rel = None
    if rel is not None:
        candidate = ROOT_DIR / "data" / "model" / rel.with_suffix(".e")
        if candidate.exists():
            return candidate
    candidate = meas_file.with_suffix(".e")
    return candidate if candidate.exists() else None


def solver_for(meas_file: Path) -> Callable[[Path], Tuple[Snapshot, str]]:
    parts = {part.lower() for part in meas_file.parts}
    if "hybrid" in parts:
        return solve_hybrid
    if "dc" in parts:
        return solve_dc
    if "ac" in parts:
        return solve_ac
    raise RuntimeError(f"Cannot infer solver type for {meas_file}")


def iter_measurement_files(paths: Sequence[str]) -> List[Path]:
    if paths:
        return [Path(path).resolve() for path in paths]
    return sorted((ROOT_DIR / "data").rglob("*.meas"))


def main(argv: Sequence[str]) -> int:
    meas_files = iter_measurement_files(argv)
    if not meas_files:
        print("No measurement files found")
        return 0

    failures = []
    for meas_file in meas_files:
        e_file = matching_e_file(meas_file)
        if e_file is None:
            failures.append((meas_file, "matching E file not found"))
            continue
        try:
            snapshot, info = solver_for(meas_file)(e_file)
            updated, missing = rewrite_measurements(meas_file, snapshot)
            rel = meas_file.relative_to(ROOT_DIR)
            print(f"{rel}: updated={updated}, missing={missing}, {info}")
        except Exception as exc:  # Keep batch progress visible; fail after the loop.
            failures.append((meas_file, str(exc)))
            print(f"{meas_file}: FAILED: {exc}")

    if failures:
        print("\nFailures:")
        for meas_file, reason in failures:
            print(f"  {meas_file}: {reason}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
