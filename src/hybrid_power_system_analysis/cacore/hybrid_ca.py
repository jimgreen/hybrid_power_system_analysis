import argparse
import math
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

ROOT_DIR = Path(__file__).resolve().parents[1]
LFCORE_DIR = ROOT_DIR / "lfcore"
for path in (ROOT_DIR, LFCORE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from efile_read import EBook
from lfcore.hybrid_lf import HybridPowerFlowCalc, _read_lf_network_from_file
from paths import DATA_DIR, model_file


DEFAULT_MODEL_FILE = model_file("hybrid", "qinling.e")
DEFAULT_LIMIT_FILE = DATA_DIR / "ca" / "qinling.calmt"
DEFAULT_N0_RESULT_FILE = ROOT_DIR / "data" / "ca" / "qinling.ca.n0rslt"
DEFAULT_N1_RESULT_FILE = ROOT_DIR / "data" / "ca" / "qinling.ca.n1rslt"
DEFAULT_CALC_INFO_FILE = ROOT_DIR / "data" / "ca" / "ca_calc_info.e"


@dataclass
class CARisk:
    risk_source: str
    risk_dist: str
    risk_type: str
    risk_value: float | str
    risk_info: str


def _as_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt(value) -> str:
    if isinstance(value, str):
        return value
    value = float(value)
    if abs(value) < 1e-12:
        value = 0.0
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


def _cell(value) -> str:
    text = str(value)
    if not text:
        return "''"
    if any(ch.isspace() for ch in text) or text.startswith("'") or text.startswith("#"):
        return "'" + text.replace("'", "_") + "'"
    return text


def _format_efile_block(table_name: str, header: list[str], rows: list[dict[str, str]]) -> str:
    rendered_rows = [[_cell(row[col]) for col in header] for row in rows]
    widths = [
        max(len(header[pos]), *(len(row[pos]) for row in rendered_rows)) if rendered_rows else len(header[pos])
        for pos in range(len(header))
    ]

    def render(prefix: str, values: list[str]) -> str:
        return prefix + " ".join(value.ljust(widths[pos]) for pos, value in enumerate(values)).rstrip() + "\n"

    parts = [f"<{table_name}>\n", render("@ ", header)]
    for row in rendered_rows:
        parts.append(render("# ", row))
    parts.append(f"</{table_name}>\n")
    return "".join(parts)


def read_constraints(limit_file: str | Path) -> dict[str, dict[str, dict[str, float]]]:
    book = EBook(limit_file)
    constraints: dict[str, dict[str, dict[str, float]]] = {}
    for table_name, block in book.data.items():
        table = constraints.setdefault(table_name, {})
        for row in block.data:
            name = str(row.get("name", row.get("idx", "")))
            table[name] = {
                key: _as_float(value)
                for key, value in row.items()
                if key not in {"idx", "name"}
            }
    return constraints


def _add_risk(
    risks: list[CARisk],
    source: str,
    dist: str,
    risk_type: str,
    value: float | str,
    info: str,
) -> None:
    risks.append(CARisk(source, dist, risk_type, value, info))


def _check_range(
    risks: list[CARisk],
    constraints: dict[str, dict[str, dict[str, float]]],
    table: str,
    name: str,
    field: str,
    value: float,
    source: str,
) -> None:
    limit = constraints.get(table, {}).get(name)
    if not limit:
        return
    max_key = f"{field}_max"
    min_key = f"{field}_min"
    if max_key in limit and value > limit[max_key]:
        _add_risk(
            risks,
            source,
            f"{table}.{name}.{field}",
            "越限",
            value,
            f"{field}={_fmt(value)}>{max_key}={_fmt(limit[max_key])}",
        )
    if min_key in limit and value < limit[min_key]:
        _add_risk(
            risks,
            source,
            f"{table}.{name}.{field}",
            "越限",
            value,
            f"{field}={_fmt(value)}<{min_key}={_fmt(limit[min_key])}",
        )


def _device_name(device) -> str:
    return str(getattr(device, "name", getattr(device, "idx", "")))


def _node_voltage(network, side: str, node_idx) -> float:
    grid = network.ac if side == "ac" else network.dc
    node = getattr(grid, "node_dict", {}).get(int(node_idx))
    return _physical_voltage(network, node)


def _physical_power(network, value: float) -> float:
    return _as_float(value) * _as_float(getattr(network, "p_base_kW", getattr(network, "p_base", 1.0)), 1.0)


def _physical_voltage(network, node) -> float:
    if node is None:
        return 0.0
    return (
        _as_float(getattr(node, "voltage", 0.0))
        * _as_float(getattr(node, "vbase", 1.0), 1.0)
        * _as_float(getattr(network, "u_scale", 1.0), 1.0)
    )


def _physical_current(network, value: float, node=None) -> float:
    if node is not None:
        vbase = _as_float(getattr(node, "vbase", 0.0))
        if vbase > 0.0:
            return _as_float(value) * _as_float(getattr(network, "p_base_kW", getattr(network, "p_base", 1.0)), 1.0) / vbase
    return _as_float(value) * _as_float(getattr(network, "i_scale", 1.0), 1.0)


def _terminal_node(network, side: str, node_idx):
    grid = network.ac if side == "ac" else network.dc
    return getattr(grid, "node_dict", {}).get(int(node_idx))


def _active_load_power(result) -> float:
    total = 0.0
    for load in getattr(result.ac_network, "loads", []):
        if getattr(load, "run_stat", 1) == 1 and getattr(load, "is_alive", True):
            total += abs(_physical_power(result.network, getattr(load, "p", 0.0)))
    for load in getattr(result.dc_network, "loads", []):
        if getattr(load, "run_stat", 1) == 1 and getattr(load, "is_alive", True):
            total += abs(_physical_power(result.network, getattr(load, "p", 0.0)))
    return total


def collect_limit_risks(result, constraints, source: str) -> list[CARisk]:
    risks: list[CARisk] = []
    network = result.network

    for node in getattr(result.ac_network, "nodes", []):
        _check_range(risks, constraints, "ACNode", _device_name(node), "v", _physical_voltage(network, node), source)
    for node in getattr(result.dc_network, "nodes", []):
        _check_range(risks, constraints, "DCNode", _device_name(node), "v", _physical_voltage(network, node), source)

    for gen in getattr(result.ac_network, "generators", []):
        name = _device_name(gen)
        _check_range(risks, constraints, "ACUnit", name, "p", _physical_power(network, getattr(gen, "p", 0.0)), source)
        _check_range(risks, constraints, "ACUnit", name, "v", _node_voltage(network, "ac", gen.node), source)
    for gen in getattr(result.dc_network, "generators", []):
        name = _device_name(gen)
        _check_range(risks, constraints, "DCUnit", name, "p", _physical_power(network, getattr(gen, "p", 0.0)), source)
        _check_range(risks, constraints, "DCUnit", name, "v", _node_voltage(network, "dc", gen.node), source)

    for load in getattr(result.ac_network, "loads", []):
        name = _device_name(load)
        _check_range(risks, constraints, "ACLoad", name, "p", abs(_physical_power(network, getattr(load, "p", 0.0))), source)
        _check_range(risks, constraints, "ACLoad", name, "v", _node_voltage(network, "ac", load.node), source)
    for load in getattr(result.dc_network, "loads", []):
        name = _device_name(load)
        _check_range(risks, constraints, "DCLoad", name, "p", abs(_physical_power(network, getattr(load, "p", 0.0))), source)
        _check_range(risks, constraints, "DCLoad", name, "v", _node_voltage(network, "dc", load.node), source)

    for branch in getattr(result.ac_network, "branches", []):
        name = _device_name(branch)
        i_node = _terminal_node(network, "ac", branch.i_node)
        j_node = _terminal_node(network, "ac", branch.j_node)
        current = max(
            abs(_physical_current(network, getattr(branch, "i_c", 0.0), i_node)),
            abs(_physical_current(network, getattr(branch, "j_c", 0.0), j_node)),
        )
        power = max(
            _physical_power(network, math.hypot(_as_float(getattr(branch, "i_p", 0.0)), _as_float(getattr(branch, "i_q", 0.0)))),
            _physical_power(network, math.hypot(_as_float(getattr(branch, "j_p", 0.0)), _as_float(getattr(branch, "j_q", 0.0)))),
        )
        _check_range(risks, constraints, "ACBranch", name, "i", current, source)
        _check_range(risks, constraints, "ACBranch", name, "p", power, source)
    for branch in getattr(result.dc_network, "branches", []):
        name = _device_name(branch)
        i_node = _terminal_node(network, "dc", branch.i_node)
        current = abs(_physical_current(network, getattr(branch, "current", 0.0), i_node))
        power = max(abs(_physical_power(network, getattr(branch, "i_p", 0.0))), abs(_physical_power(network, getattr(branch, "j_p", 0.0))))
        _check_range(risks, constraints, "DCBranch", name, "i", current, source)
        _check_range(risks, constraints, "DCBranch", name, "p", power, source)

    for brk in getattr(result.ac_network, "breakers", []):
        name = _device_name(brk)
        _check_range(risks, constraints, "ACBreaker", name, "i", abs(_physical_current(network, getattr(brk, "current", 0.0), _terminal_node(network, "ac", brk.i_node))), source)
        voltage = max(_node_voltage(network, "ac", brk.i_node), _node_voltage(network, "ac", brk.j_node))
        _check_range(risks, constraints, "ACBreaker", name, "v", voltage, source)
    for brk in getattr(result.dc_network, "breakers", []):
        name = _device_name(brk)
        _check_range(risks, constraints, "DCBreaker", name, "i", abs(_physical_current(network, getattr(brk, "current", 0.0), _terminal_node(network, "dc", brk.i_node))), source)
        voltage = max(_node_voltage(network, "dc", brk.i_node), _node_voltage(network, "dc", brk.j_node))
        _check_range(risks, constraints, "DCBreaker", name, "v", voltage, source)

    for conv in getattr(result.dc_network, "dcdc_converters", []):
        name = _device_name(conv)
        _check_range(risks, constraints, "DCDCConver", name, "i_c", _physical_current(network, getattr(conv, "i_c", 0.0), _terminal_node(network, "dc", conv.i_node)), source)
        _check_range(risks, constraints, "DCDCConver", name, "j_c", _physical_current(network, getattr(conv, "j_c", 0.0), _terminal_node(network, "dc", conv.j_node)), source)
        _check_range(risks, constraints, "DCDCConver", name, "i_v", _node_voltage(network, "dc", conv.i_node), source)
        _check_range(risks, constraints, "DCDCConver", name, "j_v", _node_voltage(network, "dc", conv.j_node), source)
    for conv in getattr(network, "dcac_converters", []):
        name = _device_name(conv)
        _check_range(risks, constraints, "DCACConver", name, "i_c", _physical_current(network, getattr(conv, "dc_i", 0.0), _terminal_node(network, "dc", conv.dc_node)), source)
        _check_range(risks, constraints, "DCACConver", name, "j_c", _physical_current(network, getattr(conv, "ac_i", 0.0), _terminal_node(network, "ac", conv.ac_node)), source)
        _check_range(risks, constraints, "DCACConver", name, "i_v", _node_voltage(network, "dc", conv.dc_node), source)
        _check_range(risks, constraints, "DCACConver", name, "j_v", _node_voltage(network, "ac", conv.ac_node), source)
    for conv in getattr(network, "acac_converters", []):
        name = _device_name(conv)
        _check_range(risks, constraints, "ACACConver", name, "i_c", _physical_current(network, getattr(conv, "i_i", 0.0), _terminal_node(network, "ac", conv.i_node)), source)
        _check_range(risks, constraints, "ACACConver", name, "j_c", _physical_current(network, getattr(conv, "j_i", 0.0), _terminal_node(network, "ac", conv.j_node)), source)
        _check_range(risks, constraints, "ACACConver", name, "i_v", _node_voltage(network, "ac", conv.i_node), source)
        _check_range(risks, constraints, "ACACConver", name, "j_v", _node_voltage(network, "ac", conv.j_node), source)

    return risks


def _run_case(model_file: str | Path):
    try:
        network = _read_lf_network_from_file(model_file)
        calc = HybridPowerFlowCalc(network, verbose=False, result_mode="full")
        calc.run()
        result = calc.lf_result
        return result, None
    except Exception as exc:  # pragma: no cover - exercised by real divergent cases.
        return None, exc


def _contingency_rows(model_file: str | Path) -> Iterable[tuple[str, str, str]]:
    book = EBook(model_file)
    for table, source_table in (
        ("ACGenerator", "ACUnit"),
        ("ACLoad", "ACLoad"),
        ("ACBranch", "ACBranch"),
        ("ACBreak", "ACBreaker"),
        ("ACTransformer", "ACTransformer"),
        ("DCGenerator", "DCUnit"),
        ("DCLoad", "DCLoad"),
        ("DCBranch", "DCBranch"),
        ("DCBreak", "DCBreaker"),
        ("DCDCConverter", "DCDCConver"),
        ("DCACConverter", "DCACConver"),
        ("ACACConverter", "ACACConver"),
    ):
        block = book.data.get(table)
        if block is None or "run_stat" not in block.header_list:
            continue
        for row in block.data:
            if int(_as_float(row.get("run_stat", 1), 1)) == 1:
                yield table, source_table, str(row.get("name", row.get("idx")))


def _write_outage_case(src_file: str | Path, dst_file: str | Path, table: str, name: str) -> None:
    book = EBook(src_file)
    block = book.data[table]
    for row in block.data:
        if str(row.get("name", row.get("idx"))) == name:
            row["run_stat"] = 0
            break
    else:
        raise ValueError(f"contingency device not found: {table}.{name}")
    book.apply_to_file(dst_file)


def write_risk_result(risks: list[CARisk], result_file: str | Path) -> None:
    result_path = Path(result_file)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["risk_source", "risk_dist", "risk_type", "risk_value", "risk_info"]
    rows = [
        {
            "risk_source": risk.risk_source,
            "risk_dist": risk.risk_dist,
            "risk_type": risk.risk_type,
            "risk_value": _fmt(risk.risk_value),
            "risk_info": risk.risk_info,
        }
        for risk in risks
    ]
    result_path.write_text(_format_efile_block("ca_risk_result", header, rows), encoding="utf8")


def _risk_level(n0_risks: list[CARisk], n1_risks: list[CARisk]) -> str:
    all_risks = [*n0_risks, *n1_risks]
    if any(risk.risk_type in {"潮流发散", "负荷损失"} for risk in all_risks):
        return "高"
    if all_risks:
        return "中"
    return "无"


def append_calc_info(
    info_file: str | Path,
    *,
    calc_start_time: str,
    calc_elapsed_seconds: float,
    n0_risk_count: int,
    n1_scan_count: int,
    n1_risk_count: int,
    risk_level: str,
) -> None:
    info_path = Path(info_file)
    info_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "calc_start_time",
        "calc_elapsed_seconds",
        "n0_risk_count",
        "n1_scan_count",
        "n1_risk_count",
        "risk_level",
    ]
    rows = []
    if info_path.exists() and info_path.stat().st_size > 0:
        try:
            block = EBook(info_path).data.get("ca_calc_info")
            if block is not None and block.header_list == header:
                rows.extend(block.data)
        except Exception:
            rows = []
    rows.append(
        {
            "calc_start_time": calc_start_time,
            "calc_elapsed_seconds": _fmt(calc_elapsed_seconds),
            "n0_risk_count": str(int(n0_risk_count)),
            "n1_scan_count": str(int(n1_scan_count)),
            "n1_risk_count": str(int(n1_risk_count)),
            "risk_level": risk_level,
        }
    )
    info_path.write_text(_format_efile_block("ca_calc_info", header, rows), encoding="utf8")


def run_hybrid_ca(
    model_file: str | Path = DEFAULT_MODEL_FILE,
    limit_file: str | Path = DEFAULT_LIMIT_FILE,
    n0_result_file: str | Path = DEFAULT_N0_RESULT_FILE,
    n1_result_file: str | Path = DEFAULT_N1_RESULT_FILE,
    calc_info_file: str | Path = DEFAULT_CALC_INFO_FILE,
    *,
    scan_n1: bool = True,
) -> tuple[list[CARisk], list[CARisk]]:
    start_clock = time.perf_counter()
    calc_start_time = datetime.now().isoformat(timespec="seconds")
    constraints = read_constraints(limit_file)
    n0_risks: list[CARisk] = []
    n1_risks: list[CARisk] = []
    n1_scan_count = 0

    base_result, base_error = _run_case(model_file)
    if base_error is not None or base_result is None or not base_result.converged:
        detail = str(base_error) if base_error is not None else "hybrid_power_flow_not_converged"
        _add_risk(n0_risks, "BASE", "HybridPowerFlow", "潮流发散", 1, detail.replace(" ", "_"))
        write_risk_result(n0_risks, n0_result_file)
        write_risk_result(n1_risks, n1_result_file)
        append_calc_info(
            calc_info_file,
            calc_start_time=calc_start_time,
            calc_elapsed_seconds=time.perf_counter() - start_clock,
            n0_risk_count=len(n0_risks),
            n1_scan_count=n1_scan_count,
            n1_risk_count=len(n1_risks),
            risk_level=_risk_level(n0_risks, n1_risks),
        )
        return n0_risks, n1_risks

    n0_risks.extend(collect_limit_risks(base_result, constraints, "BASE"))
    base_load = _active_load_power(base_result)

    if scan_n1:
        with tempfile.TemporaryDirectory(prefix="hybrid_ca_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            for table, source_table, name in _contingency_rows(model_file):
                n1_scan_count += 1
                source = f"N-1:{source_table}.{name}"
                outage_file = tmp_path / f"{table}_{name}.e"
                _write_outage_case(model_file, outage_file, table, name)
                result, error = _run_case(outage_file)
                if error is not None or result is None or not result.converged:
                    detail = str(error) if error is not None else "hybrid_power_flow_not_converged"
                    _add_risk(n1_risks, source, "HybridPowerFlow", "潮流发散", 1, detail.replace(" ", "_"))
                    continue
                n1_risks.extend(collect_limit_risks(result, constraints, source))
                load_loss = max(0.0, base_load - _active_load_power(result))
                if load_loss > 1e-6:
                    _add_risk(
                        n1_risks,
                        source,
                        "Load",
                        "负荷损失",
                        load_loss,
                        f"loss={_fmt(load_loss)};base_load={_fmt(base_load)}",
                    )

    write_risk_result(n0_risks, n0_result_file)
    write_risk_result(n1_risks, n1_result_file)
    append_calc_info(
        calc_info_file,
        calc_start_time=calc_start_time,
        calc_elapsed_seconds=time.perf_counter() - start_clock,
        n0_risk_count=len(n0_risks),
        n1_scan_count=n1_scan_count,
        n1_risk_count=len(n1_risks),
        risk_level=_risk_level(n0_risks, n1_risks),
    )
    return n0_risks, n1_risks


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Hybrid AC/DC N-1 contingency analysis")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_FILE), help="Hybrid network E file.")
    parser.add_argument("--limit", default=str(DEFAULT_LIMIT_FILE), help="Security constraint E file.")
    parser.add_argument("--n0-output", default=str(DEFAULT_N0_RESULT_FILE), help="Base-case risk result E file.")
    parser.add_argument("--n1-output", default=str(DEFAULT_N1_RESULT_FILE), help="N-1 risk result E file.")
    parser.add_argument("--info-output", default=str(DEFAULT_CALC_INFO_FILE), help="CA calculation info E file.")
    parser.add_argument("--no-n1", action="store_true", help="Only evaluate the base case.")
    args = parser.parse_args(argv)

    n0_risks, n1_risks = run_hybrid_ca(
        args.model,
        args.limit,
        args.n0_output,
        args.n1_output,
        args.info_output,
        scan_n1=not args.no_n1,
    )
    print(f"CA base risks: {len(n0_risks)}")
    print(f"CA N-1 risks: {len(n1_risks)}")
    print(f"Base result: {Path(args.n0_output).resolve()}")
    print(f"N-1 result: {Path(args.n1_output).resolve()}")
    print(f"Calc info: {Path(args.info_output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
