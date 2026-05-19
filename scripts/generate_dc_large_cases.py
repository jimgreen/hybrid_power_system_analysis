import argparse
import contextlib
import io
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

from lfcore.dc_lf import DCPowerFlowCalc
from model.dc_model import DCPowerNetwork
from unit_system import dc_current_base_ka


POWER_BASE = 100.0
U_SCALE = 1.0
P_SCALE = 1.0
I_SCALE = 1.0
VBASE = 100.0
NODES_PER_BLOCK = 10


def _fmt(value) -> str:
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        return f"{float(value):.10g}"
    return str(value)


def _format_block(name: str, headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    widths = [len(item) for item in headers]
    string_rows = [[_fmt(item) for item in row] for row in rows]
    for row in string_rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    lines = [f"<{name}>", "@ " + " ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers))]
    for row in string_rows:
        lines.append("# " + " ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)))
    lines.append(f"</{name}>")
    return "\n".join(lines)


def _block_base(block: int) -> int:
    return block * NODES_PER_BLOCK


def _build_case_tables(n_nodes: int) -> List[tuple]:
    if n_nodes <= 0 or n_nodes % NODES_PER_BLOCK != 0:
        raise ValueError(f"n_nodes must be a positive multiple of {NODES_PER_BLOCK}")

    n_blocks = n_nodes // NODES_PER_BLOCK
    nodes = []
    branches = []
    loads = []
    generators = []
    zero_branches = []
    switches = []
    dcdcs = []

    branch_idx = load_idx = gen_idx = zero_idx = switch_idx = dcdc_idx = 0
    ring_r = [0.08, 0.06, 0.07, 0.08, 0.06, 0.07, 0.08, 0.06, 0.07, 0.09]
    load_specs = [
        (1, 35.0, 4.0, 2.0),
        (2, 30.0, 3.0, 2.5),
        (4, 25.0, 4.0, 1.5),
        (6, 35.0, 3.0, 2.0),
        (8, 30.0, 4.0, 1.0),
    ]

    for block in range(n_blocks):
        base = _block_base(block)
        for local in range(NODES_PER_BLOCK):
            idx = base + local
            nodes.append((idx, f"nd_{idx + 1}", VBASE, 100.0, block, 1))

        for local in range(NODES_PER_BLOCK):
            i_node = base + local
            j_node = base + ((local + 1) % NODES_PER_BLOCK)
            branches.append((branch_idx, f"line_{i_node}_{j_node}", i_node, j_node, ring_r[local], 1))
            branch_idx += 1
        for i_local, j_local, r in ((1, 5, 0.12), (3, 7, 0.11)):
            i_node = base + i_local
            j_node = base + j_local
            branches.append((branch_idx, f"line_{i_node}_{j_node}", i_node, j_node, r, 1))
            branch_idx += 1

        for node_local, pv0, pv1, pv2 in load_specs:
            node = base + node_local
            loads.append((load_idx, f"load_{load_idx + 1}", node, 1.0, pv0, pv1, pv2, 1))
            load_idx += 1

        generators.append((gen_idx, f"gen_v_{block + 1}", base + 0, "V", 160.0, 0.0, 0.0, 1))
        gen_idx += 1
        generators.append((gen_idx, f"gen_p_{block + 1}", base + 3, "P", 100.0, 80.0, 0.0, 1))
        gen_idx += 1
        generators.append((gen_idx, f"gen_i_{block + 1}", base + 7, "I", 100.0, 0.0, 0.00055, 1))
        gen_idx += 1

        zero_branches.append((zero_idx, f"zbr_{base + 1}_{base + 2}", base + 1, base + 2, 1))
        zero_idx += 1
        switches.append((switch_idx, f"sw_{base + 2}_{base + 3}", base + 2, base + 3, 1, 1))
        switch_idx += 1

        dcdcs.append(
            (
                dcdc_idx,
                f"conv_{block + 1}",
                base + 4,
                base + 5,
                0.04,
                0.04,
                "P",
                35.0,
                0.0,
                0.0,
                1,
            )
        )
        dcdc_idx += 1

    return [
        ("PowerBase", ("p_base", "u_scale", "p_scale", "i_scale"), [(POWER_BASE, U_SCALE, P_SCALE, I_SCALE)]),
        ("DCNode", ("idx", "name", "vbase", "voltage", "isl", "run_stat"), nodes),
        ("DCBranch", ("idx", "name", "i_node", "j_node", "r", "run_stat"), branches),
        ("DCLoad", ("idx", "name", "node", "pbase", "pv0", "pv1", "pv2", "run_stat"), loads),
        ("DCGenerator", ("idx", "name", "node", "control_type", "v_set", "p_set", "i_set", "run_stat"), generators),
        ("DCZeroBranch", ("idx", "name", "i_node", "j_node", "run_stat"), zero_branches),
        ("DCSwitch", ("idx", "name", "i_node", "j_node", "status", "run_stat"), switches),
        (
            "DCDCConverter",
            ("idx", "name", "i_node", "j_node", "r1", "r2", "control_type", "p_set", "i_set", "v_set", "run_stat"),
            dcdcs,
        ),
    ]


def _write_e_file(n_nodes: int, e_file: Path) -> None:
    blocks = [_format_block(name, headers, rows) for name, headers, rows in _build_case_tables(n_nodes)]
    e_file.parent.mkdir(parents=True, exist_ok=True)
    e_file.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def _power_to_file(network: DCPowerNetwork, power: float) -> float:
    return float(power) * float(network.p_base)


def _voltage_to_file(network: DCPowerNetwork, node_idx: int, voltage: float) -> float:
    node = network.node_dict[node_idx]
    return float(voltage) * float(network.u_scale) * float(node.vbase)


def _current_to_file(network: DCPowerNetwork, node_idx: int, current: float) -> float:
    node = network.node_dict[node_idx]
    base = dc_current_base_ka(float(network.p_base_kW), float(node.vbase))
    return float(current) * float(network.i_scale) * base


def _weight(rng: np.random.Generator) -> float:
    return float(rng.uniform(0.1, 10.0))


def _add_meas(rows: List[Sequence[object]], rng: np.random.Generator, name: str, dev_type: str, dev_name: str, meas_type: str, value: float) -> None:
    rows.append((len(rows), name, dev_type, dev_name, meas_type, f"{_weight(rng):.4f}", 1, value))


def _build_measurement_rows(network: DCPowerNetwork, seed: int = 20260504) -> List[Sequence[object]]:
    rng = np.random.default_rng(seed)
    rows: List[Sequence[object]] = []

    for node in sorted(network.nodes, key=lambda item: item.idx):
        _add_meas(rows, rng, f"v_{node.name}", "DCNode", node.name, "V", _voltage_to_file(network, node.idx, node.voltage))

    for br in sorted(network.branches, key=lambda item: item.idx):
        _add_meas(rows, rng, f"p_from_{br.name}", "DCBranch", br.name, "P_FROM", _power_to_file(network, br.i_p))
        _add_meas(rows, rng, f"v_from_{br.name}", "DCBranch", br.name, "V_FROM", _voltage_to_file(network, br.i_node, br.i_node_obj.voltage))
        _add_meas(rows, rng, f"i_from_{br.name}", "DCBranch", br.name, "I_FROM", _current_to_file(network, br.i_node, br.current))
        _add_meas(rows, rng, f"p_to_{br.name}", "DCBranch", br.name, "P_TO", _power_to_file(network, br.j_p))
        _add_meas(rows, rng, f"v_to_{br.name}", "DCBranch", br.name, "V_TO", _voltage_to_file(network, br.j_node, br.j_node_obj.voltage))
        _add_meas(rows, rng, f"i_to_{br.name}", "DCBranch", br.name, "I_TO", _current_to_file(network, br.j_node, -br.current))

    for gen in sorted(network.generators, key=lambda item: item.idx):
        _add_meas(rows, rng, f"p_gen_{gen.name}", "DCGenerator", gen.name, "P_GEN", _power_to_file(network, gen.p))
        _add_meas(rows, rng, f"v_gen_{gen.name}", "DCGenerator", gen.name, "V_GEN", _voltage_to_file(network, gen.node, gen.node_obj.voltage))
        _add_meas(rows, rng, f"i_gen_{gen.name}", "DCGenerator", gen.name, "I_GEN", _current_to_file(network, gen.node, gen.current))

    for load in sorted(network.loads, key=lambda item: item.idx):
        _add_meas(rows, rng, f"p_load_{load.name}", "DCLoad", load.name, "P_LOAD", _power_to_file(network, load.p))
        _add_meas(rows, rng, f"v_load_{load.name}", "DCLoad", load.name, "V_LOAD", _voltage_to_file(network, load.node, load.node_obj.voltage))
        _add_meas(rows, rng, f"i_load_{load.name}", "DCLoad", load.name, "I_LOAD", _current_to_file(network, load.node, load.current))

    for sw in sorted(network.switches, key=lambda item: item.idx):
        _add_meas(rows, rng, f"p_from_{sw.name}", "DCSwitch", sw.name, "P_FROM", _power_to_file(network, sw.p))
        _add_meas(rows, rng, f"v_from_{sw.name}", "DCSwitch", sw.name, "V_FROM", _voltage_to_file(network, sw.i_node, sw.i_node_obj.voltage))
        _add_meas(rows, rng, f"i_from_{sw.name}", "DCSwitch", sw.name, "I_FROM", _current_to_file(network, sw.i_node, sw.current))

    for conv in sorted(network.dcdc_converters, key=lambda item: item.idx):
        _add_meas(rows, rng, f"p_from_{conv.name}", "DCDCConverter", conv.name, "P_FROM", _power_to_file(network, conv.i_p))
        _add_meas(rows, rng, f"v_from_{conv.name}", "DCDCConverter", conv.name, "V_FROM", _voltage_to_file(network, conv.i_node, conv.i_node_obj.voltage))
        _add_meas(rows, rng, f"i_from_{conv.name}", "DCDCConverter", conv.name, "I_FROM", _current_to_file(network, conv.i_node, conv.i_c))
        _add_meas(rows, rng, f"p_to_{conv.name}", "DCDCConverter", conv.name, "P_TO", _power_to_file(network, conv.j_p))
        _add_meas(rows, rng, f"v_to_{conv.name}", "DCDCConverter", conv.name, "V_TO", _voltage_to_file(network, conv.j_node, conv.j_node_obj.voltage))
        _add_meas(rows, rng, f"i_to_{conv.name}", "DCDCConverter", conv.name, "I_TO", _current_to_file(network, conv.j_node, conv.j_c))

    return rows


def _solve_dc_case(e_file: Path) -> DCPowerNetwork:
    network = DCPowerNetwork()
    network.read_from_file(e_file)
    network.topo()
    with contextlib.redirect_stdout(io.StringIO()):
        warnings, errors = network.check_topo()
    if errors:
        raise RuntimeError(f"Topology check failed for {e_file}: {errors}")

    calc = DCPowerFlowCalc(network)
    with contextlib.redirect_stdout(io.StringIO()):
        rc = calc.run()
    if rc != 0 or not calc.converged:
        raise RuntimeError(f"Power flow failed for {e_file}: rc={rc}, iter={calc.iterations}, normF={calc.normF:.3e}")
    return network


def generate_dc_case_files(n_nodes: int, e_file: Path, meas_file: Path, seed: int = 20260504) -> None:
    e_file = Path(e_file)
    meas_file = Path(meas_file)
    _write_e_file(n_nodes, e_file)
    network = _solve_dc_case(e_file)
    rows = _build_measurement_rows(network, seed=seed)
    meas_file.parent.mkdir(parents=True, exist_ok=True)
    meas_file.write_text(
        _format_block(
            "Measurement",
            ("idx", "name", "dev_type", "dev_name", "meas_type", "weight", "valid", "value"),
            rows,
        )
        + "\n",
        encoding="utf-8",
    )


def generate_default_cases(output_dir: Path = Path("data") / "dc") -> None:
    generate_dc_case_files(1000, output_dir / "dc_net_1000.e", output_dir / "dc_net_1000.meas", seed=1000)
    generate_dc_case_files(3000, output_dir / "dc_net_3000.e", output_dir / "dc_net_3000.meas", seed=3000)


def main(argv: Iterable[str] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate large DC E/meas cases based on dc_net_30 patterns.")
    parser.add_argument("--nodes", type=int, choices=(1000, 3000), default=None, help="Generate one case with this node count.")
    parser.add_argument("--output-dir", default=str(Path("data") / "dc"), help="Output directory for generated E/meas files.")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    if args.nodes is None:
        generate_default_cases(output_dir)
    else:
        generate_dc_case_files(args.nodes, output_dir / f"dc_net_{args.nodes}.e", output_dir / f"dc_net_{args.nodes}.meas", seed=args.nodes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
