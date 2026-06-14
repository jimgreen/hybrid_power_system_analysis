# hybrid_power_system_analysis

Hybrid AC/DC power-flow, state-estimation, simulation, AGC, and contingency-analysis tools.

## Project Layout

- `src/hybrid_power_system_analysis/` contains importable Python packages and modules.
- `data/` contains sample AC, DC, hybrid, and contingency-analysis input files.
- `tests/` contains the pytest test suite.
- `docs/` contains technical notes and implementation plans.
- `scripts/` contains one-off generation and comparison utilities.
- `benchmarks/` contains benchmark drivers and performance comparison scripts.
- `power_plan/` is a standalone monitoring UI/server area.

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

