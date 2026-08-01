# Primary Frequency Response Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python program that calculates primary frequency response curves after a power deficit disturbance, including diesel generator governor response and grid-forming storage regulation.

**Architecture:** Add a focused simulation module under `src/hybrid_power_system_analysis/simu/` with dataclasses for system, diesel, storage, and disturbance parameters. The solver uses explicit time-domain integration with clamps, governor lag, diesel ramp limits, storage droop/inertia support, and SOC energy bounds. A CLI script runs a default case and writes CSV output for plotting or web use.

**Tech Stack:** Python 3.11, standard library, `unittest`, optional CSV output through `csv`.

---

### Task 1: Core Model API

**Files:**
- Create: `src/hybrid_power_system_analysis/simu/__init__.py`
- Create: `src/hybrid_power_system_analysis/simu/primary_frequency_response.py`
- Test: `tests/test_primary_frequency_response.py`

- [ ] **Step 1: Write the failing test**

```python
def test_power_deficit_makes_frequency_drop_and_storage_responds_first():
    from hybrid_power_system_analysis.simu.primary_frequency_response import (
        DieselGovernor,
        Disturbance,
        GridFormingStorage,
        SystemFrequencyModel,
        simulate_primary_frequency_response,
    )

    result = simulate_primary_frequency_response(
        system=SystemFrequencyModel(f_nom_hz=50.0, s_base_mw=10.0, inertia_s=4.0, damping_mw_per_hz=0.2),
        diesel=DieselGovernor(reserve_mw=2.0, droop_mw_per_hz=0.8, time_constant_s=2.0, ramp_mw_per_s=0.5),
        storage=GridFormingStorage(
            discharge_limit_mw=2.0,
            charge_limit_mw=2.0,
            droop_mw_per_hz=1.5,
            inertia_mw_s_per_hz=0.4,
            response_time_s=0.1,
            energy_mwh=4.0,
            initial_soc=0.8,
            min_soc=0.2,
            max_soc=0.95,
        ),
        disturbance=Disturbance(start_s=1.0, deficit_mw=2.0),
        duration_s=20.0,
        dt_s=0.02,
    )

    assert result.nadir_hz < 50.0
    assert result.nadir_time_s > 1.0
    assert max(result.storage_power_mw) > max(result.diesel_power_mw[:20])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_primary_frequency_response`
Expected: FAIL or ERROR because `primary_frequency_response` is not implemented.

- [ ] **Step 3: Write minimal implementation**

Create dataclasses and the simulator, returning arrays and summary properties.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_primary_frequency_response`
Expected: PASS.

### Task 2: Limits, SOC, and Input Validation

**Files:**
- Modify: `src/hybrid_power_system_analysis/simu/primary_frequency_response.py`
- Test: `tests/test_primary_frequency_response.py`

- [ ] **Step 1: Write failing tests**

```python
def test_storage_soc_limit_prevents_discharge_below_minimum():
    ...

def test_invalid_time_step_is_rejected():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_primary_frequency_response`
Expected: FAIL due missing SOC and validation behavior.

- [ ] **Step 3: Implement limits and validation**

Clamp diesel reserve/ramp, storage charge/discharge, and SOC bounds; raise `ValueError` for invalid simulation inputs.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_primary_frequency_response`
Expected: PASS.

### Task 3: CLI CSV Program

**Files:**
- Create: `scripts/simulate_primary_frequency_response.py`
- Test: `tests/test_primary_frequency_response.py`

- [ ] **Step 1: Write failing CLI test**

```python
def test_cli_writes_csv_with_frequency_and_resource_columns():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_primary_frequency_response`
Expected: FAIL because the CLI script does not exist.

- [ ] **Step 3: Implement CLI**

Add argparse options for duration, step, nominal frequency, deficit, diesel limits, storage limits, and output CSV path.

- [ ] **Step 4: Run full verification**

Run: `python -m unittest tests.test_primary_frequency_response`
Expected: PASS.

### Task 4: Parallel Diesel and Grid-Forming Storage Units

**Files:**
- Modify: `src/hybrid_power_system_analysis/simu/primary_frequency_response.py`
- Modify: `scripts/simulate_primary_frequency_response.py`
- Test: `tests/test_primary_frequency_response.py`

- [x] **Step 1: Write failing tests**

Add tests that call `simulate_primary_frequency_response` with two `DieselGovernor` units and two
`GridFormingStorage` units with different parameters. Assert that per-unit names, per-unit power curves,
per-unit SOC curves, and total power curves are returned and that the totals equal the sum of the individual units.

- [x] **Step 2: Implement parallel resource state**

Normalize the original single-resource arguments into unit lists for backward compatibility, keep one dynamic state
per diesel and storage unit, sum all diesel and storage powers for the system frequency equation, and return both
aggregate and per-unit curves.

- [x] **Step 3: Extend CLI typical case**

Add `--case parallel` to run a typical two-diesel/two-storage case. CSV output keeps the original aggregate columns
and adds per-unit power/SOC columns when the parallel case is selected.

- [x] **Step 4: Verify**

Run: `python -m pytest tests/test_primary_frequency_response.py -q -p no:cacheprovider`
Expected: PASS.
