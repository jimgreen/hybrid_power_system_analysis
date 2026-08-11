# hybrid_power_system_analysis

Hybrid AC/DC, heat, gas, hydrogen, and steam power-flow/state-estimation tools, plus simulation, AGC, and contingency analysis.

## Project Layout

- `src/hybrid_power_system_analysis/` contains importable Python packages and modules.
- `data/` contains sample AC, DC, hybrid, and contingency-analysis input files.
- `tests/` contains the pytest test suite.
- `docs/` contains technical notes and implementation plans.
- `scripts/` contains one-off generation and comparison utilities.
- `benchmarks/` contains benchmark drivers and performance comparison scripts.
- `power_plan/` is a standalone monitoring UI/server area.

## Multi-Energy Networks

The steady-state multi-energy entry points follow the same `prepare()` / `run()` contract as the AC/DC solvers:

```powershell
python src/hybrid_power_system_analysis/lfcore/heat_lf.py data/model/heat/heat_net_3.e
python src/hybrid_power_system_analysis/lfcore/heat_lf.py data/model/heat/heat_explicit_return.e
python src/hybrid_power_system_analysis/lfcore/heat_lf.py data/model/heat/heat_three_port_exchanger.e
python src/hybrid_power_system_analysis/lfcore/gas_lf.py data/model/gas/gas_net_3.e
python src/hybrid_power_system_analysis/lfcore/hydro_lf.py data/model/hydro/hydro_net_3.e
python src/hybrid_power_system_analysis/lfcore/steam_lf.py data/model/steam/steam_net_5.e

python src/hybrid_power_system_analysis/secore/heat_se.py data/model/heat/heat_net_3.e --meas data/meas/heat/heat_net_3.meas --flat-start
python src/hybrid_power_system_analysis/secore/heat_se.py data/model/heat/heat_explicit_return.e --meas data/meas/heat/heat_explicit_return.meas --flat-start
python src/hybrid_power_system_analysis/secore/gas_se.py data/model/gas/gas_net_3.e --meas data/meas/gas/gas_net_3.meas --flat-start
python src/hybrid_power_system_analysis/secore/hydro_se.py data/model/hydro/hydro_net_3.e --meas data/meas/hydro/hydro_net_3.meas --flat-start
python src/hybrid_power_system_analysis/secore/steam_se.py data/model/steam/steam_net_5.e --meas data/meas/steam/steam_net_5.meas --flat-start
```

`hydro_lf.py` and `hydro_se.py` are the requested historical names for the hydrogen-network modules; they do not model hydroelectric generation. See `docs/fluid_network_lf_se.md` for E-block and equation definitions.

Generate and benchmark complete heat, gas, hydrogen, and steam networks at 10 through 5000 nodes:

```powershell
python scripts/check_fluid_scale_lf_se.py
```

The command writes generated E files below `data/model/<network>/`, generated measurement snapshots below `data/meas/<network>/`, and phase timings plus LF/SE consistency metrics to `output/fluid_scale_benchmark/results.json`, `results.csv`, and `summary.md`.

## Development

Create or reuse a Python environment, then install the package in editable mode:

```powershell
python -m pip install -e ".[dev]"
```

Run tests:

```powershell
python -m pytest
```

Run lint checks:

```powershell
python -m ruff check .
```

Format Python files:

```powershell
python -m ruff format .
```

