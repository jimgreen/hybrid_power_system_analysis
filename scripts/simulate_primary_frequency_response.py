"""Run a primary frequency response case and export the curve as CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hybrid_power_system_analysis.simu.primary_frequency_response import (  # noqa: E402
    DieselGovernor,
    Disturbance,
    GridFormingStorage,
    SystemFrequencyModel,
    simulate_primary_frequency_response,
)


CSV_COLUMNS = [
    "time_s",
    "frequency_hz",
    "delta_frequency_hz",
    "diesel_power_mw",
    "storage_power_mw",
    "storage_soc",
    "power_deficit_mw",
]


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_distinct_file_paths(args.input, args.output, args.summary_output)
        if args.input is not None:
            system, disturbance, duration_s, dt_s, diesels, storages = _load_case_file(args.input)
        else:
            diesels, storages = _build_units(args)
            system = SystemFrequencyModel(
                f_nom_hz=args.f_nom_hz,
                s_base_mw=args.s_base_mw,
                inertia_s=args.inertia_s,
                damping_mw_per_hz=args.damping_mw_per_hz,
            )
            disturbance = Disturbance(
                start_s=args.disturbance_start_s,
                deficit_mw=args.deficit_mw,
            )
            duration_s = args.duration_s
            dt_s = args.dt_s
    except ValueError as exc:
        parser.error(str(exc))

    result = simulate_primary_frequency_response(
        system=system,
        diesels=diesels,
        storages=storages,
        disturbance=disturbance,
        duration_s=duration_s,
        dt_s=dt_s,
    )

    include_unit_columns = args.input is not None or args.case == "parallel" or len(diesels) > 1 or len(storages) > 1
    _write_csv(args.output, result, include_unit_columns=include_unit_columns)
    if args.summary_output is not None:
        _write_summary_json(args.summary_output, result, args.input, args.output)
    print(
        "nadir_hz={:.6f}, nadir_time_s={:.3f}, final_frequency_hz={:.6f}, "
        "max_diesel_power_mw={:.6f}, max_storage_power_mw={:.6f}, diesel_units={}, storage_units={}, input={}, "
        "output={}, summary={}".format(
            result.nadir_hz,
            result.nadir_time_s,
            result.final_frequency_hz,
            result.max_diesel_power_mw,
            result.max_storage_power_mw,
            len(result.diesel_unit_names),
            len(result.storage_unit_names),
            args.input,
            args.output,
            args.summary_output,
        )
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="一次频率响应曲线计算：功率缺额、柴油发电机、构网储能。")
    parser.add_argument("--case", choices=["single", "parallel"], default="single")
    parser.add_argument("--input", type=Path, help="JSON 输入算例文件。配置该项时，系统、扰动和设备参数从文件读取。")
    parser.add_argument("--output", type=Path, default=ROOT_DIR / "output" / "primary_frequency_response.csv")
    parser.add_argument("--summary-output", type=Path, help="JSON 摘要输出文件。")
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--dt-s", type=float, default=0.02)
    parser.add_argument("--f-nom-hz", type=float, default=50.0)
    parser.add_argument("--s-base-mw", type=float, default=10.0)
    parser.add_argument("--inertia-s", type=float, default=4.0)
    parser.add_argument("--damping-mw-per-hz", type=float, default=1.0)
    parser.add_argument("--disturbance-start-s", type=float, default=1.0)
    parser.add_argument("--deficit-mw", type=float, default=2.0)
    parser.add_argument("--diesel-reserve-mw", type=float, default=2.0)
    parser.add_argument("--diesel-droop-mw-per-hz", type=float, default=0.8)
    parser.add_argument("--diesel-time-constant-s", type=float, default=2.0)
    parser.add_argument("--diesel-ramp-mw-per-s", type=float, default=0.5)
    parser.add_argument("--diesel-deadband-hz", type=float, default=0.03)
    parser.add_argument("--storage-discharge-limit-mw", type=float, default=2.0)
    parser.add_argument("--storage-charge-limit-mw", type=float, default=2.0)
    parser.add_argument("--storage-droop-mw-per-hz", type=float, default=1.5)
    parser.add_argument("--storage-inertia-mw-s-per-hz", type=float, default=0.4)
    parser.add_argument("--storage-response-time-s", type=float, default=0.1)
    parser.add_argument("--storage-energy-mwh", type=float, default=4.0)
    parser.add_argument("--storage-initial-soc", type=float, default=0.8)
    parser.add_argument("--storage-min-soc", type=float, default=0.2)
    parser.add_argument("--storage-max-soc", type=float, default=0.95)
    parser.add_argument("--storage-deadband-hz", type=float, default=0.01)
    return parser


def _load_case_file(
    input_file: Path,
) -> tuple[SystemFrequencyModel, Disturbance, float, float, list[DieselGovernor], list[GridFormingStorage]]:
    with input_file.open("r", encoding="utf-8") as fp:
        config = json.load(fp)

    simulation = config.get("simulation", {})
    system = SystemFrequencyModel(**config.get("system", {}))
    disturbance = Disturbance(**config["disturbance"])
    diesels = [DieselGovernor(**item) for item in config.get("diesels", [])]
    storages = [GridFormingStorage(**item) for item in config.get("storages", [])]
    return (
        system,
        disturbance,
        float(simulation.get("duration_s", 60.0)),
        float(simulation.get("dt_s", 0.02)),
        diesels,
        storages,
    )


def _validate_distinct_file_paths(input_file: Path | None, output_file: Path, summary_output: Path | None) -> None:
    named_paths = [
        ("input", input_file),
        ("output", output_file),
        ("summary", summary_output),
    ]
    resolved: dict[Path, str] = {}
    for name, path in named_paths:
        if path is None:
            continue
        normalized = path.resolve()
        if normalized in resolved:
            raise ValueError(f"{resolved[normalized]} and {name} must be different files")
        resolved[normalized] = name


def _build_units(args) -> tuple[list[DieselGovernor], list[GridFormingStorage]]:
    if args.case == "parallel":
        return _parallel_case_units()
    return [
        DieselGovernor(
            reserve_mw=args.diesel_reserve_mw,
            droop_mw_per_hz=args.diesel_droop_mw_per_hz,
            time_constant_s=args.diesel_time_constant_s,
            ramp_mw_per_s=args.diesel_ramp_mw_per_s,
            deadband_hz=args.diesel_deadband_hz,
            name="diesel_1",
        )
    ], [
        GridFormingStorage(
            discharge_limit_mw=args.storage_discharge_limit_mw,
            charge_limit_mw=args.storage_charge_limit_mw,
            droop_mw_per_hz=args.storage_droop_mw_per_hz,
            inertia_mw_s_per_hz=args.storage_inertia_mw_s_per_hz,
            response_time_s=args.storage_response_time_s,
            energy_mwh=args.storage_energy_mwh,
            initial_soc=args.storage_initial_soc,
            min_soc=args.storage_min_soc,
            max_soc=args.storage_max_soc,
            deadband_hz=args.storage_deadband_hz,
            name="bess_1",
        )
    ]


def _parallel_case_units() -> tuple[list[DieselGovernor], list[GridFormingStorage]]:
    return [
        DieselGovernor(
            name="diesel_slow",
            reserve_mw=1.2,
            droop_mw_per_hz=0.55,
            time_constant_s=3.0,
            ramp_mw_per_s=0.25,
            deadband_hz=0.03,
        ),
        DieselGovernor(
            name="diesel_fast",
            reserve_mw=0.8,
            droop_mw_per_hz=0.45,
            time_constant_s=1.2,
            ramp_mw_per_s=0.55,
            deadband_hz=0.03,
        ),
    ], [
        GridFormingStorage(
            name="bess_1",
            discharge_limit_mw=1.5,
            charge_limit_mw=1.0,
            droop_mw_per_hz=1.0,
            inertia_mw_s_per_hz=0.25,
            response_time_s=0.08,
            energy_mwh=3.0,
            initial_soc=0.8,
            min_soc=0.2,
            max_soc=0.95,
            deadband_hz=0.01,
        ),
        GridFormingStorage(
            name="bess_2",
            discharge_limit_mw=0.8,
            charge_limit_mw=0.8,
            droop_mw_per_hz=0.7,
            inertia_mw_s_per_hz=0.15,
            response_time_s=0.15,
            energy_mwh=1.5,
            initial_soc=0.65,
            min_soc=0.25,
            max_soc=0.9,
            deadband_hz=0.01,
        ),
    ]


def _write_csv(output: Path, result, *, include_unit_columns: bool = False) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = list(CSV_COLUMNS)
    if include_unit_columns:
        columns.extend(f"diesel_power_mw__{name}" for name in result.diesel_unit_names)
        columns.extend(f"storage_power_mw__{name}" for name in result.storage_unit_names)
        columns.extend(f"storage_soc__{name}" for name in result.storage_unit_names)
    with output.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        for idx, t_s in enumerate(result.time_s):
            row = {
                "time_s": _fmt(t_s),
                "frequency_hz": _fmt(result.frequency_hz[idx]),
                "delta_frequency_hz": _fmt(result.delta_frequency_hz[idx]),
                "diesel_power_mw": _fmt(result.diesel_power_mw[idx]),
                "storage_power_mw": _fmt(result.storage_power_mw[idx]),
                "storage_soc": _fmt(result.storage_soc[idx]),
                "power_deficit_mw": _fmt(result.power_deficit_mw[idx]),
            }
            if include_unit_columns:
                for unit_idx, name in enumerate(result.diesel_unit_names):
                    row[f"diesel_power_mw__{name}"] = _fmt(result.diesel_units_power_mw[unit_idx][idx])
                for unit_idx, name in enumerate(result.storage_unit_names):
                    row[f"storage_power_mw__{name}"] = _fmt(result.storage_units_power_mw[unit_idx][idx])
                    row[f"storage_soc__{name}"] = _fmt(result.storage_units_soc[unit_idx][idx])
            writer.writerow(row)


def _write_summary_json(output: Path, result, input_file: Path | None, curve_output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "input_file": str(input_file) if input_file is not None else None,
        "curve_output_file": str(curve_output),
        "nadir_hz": result.nadir_hz,
        "nadir_time_s": result.nadir_time_s,
        "final_frequency_hz": result.final_frequency_hz,
        "final_delta_frequency_hz": result.final_delta_frequency_hz,
        "max_diesel_power_mw": result.max_diesel_power_mw,
        "max_storage_power_mw": result.max_storage_power_mw,
        "diesel_unit_names": result.diesel_unit_names,
        "storage_unit_names": result.storage_unit_names,
    }
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _fmt(value: float) -> str:
    return f"{value:.9g}"


if __name__ == "__main__":
    raise SystemExit(main())
