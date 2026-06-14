# Standard Package Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the repository into a standard Python package layout while preserving existing algorithm behavior and test entry points.

**Architecture:** Move importable project code under `src/hybrid_power_system_analysis/` and keep `data/`, `docs/`, `tests/`, and runtime outputs outside the package. Configure `pyproject.toml` so pytest and editable installs can find both the package and legacy flat imports during this transition.

**Tech Stack:** Python 3.13, setuptools, pytest, ruff, NumPy, SciPy.

---

### Task 1: Add Packaging And Tooling Config

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Modify: `README.md`

- [ ] Add `pyproject.toml` with setuptools package discovery from `src`, pytest configuration, and ruff formatting/lint defaults.
- [ ] Add `.gitignore` entries for Python caches, pytest caches, local IDE files, build outputs, logs, temporary run directories, profile files, and generated benchmark outputs.
- [ ] Update `README.md` with the package layout and common development commands.
- [ ] Verify `python -m pytest --collect-only -q` can collect tests through the configured paths.

### Task 2: Move Importable Code Into `src`

**Files:**
- Create directory: `src/hybrid_power_system_analysis/`
- Move: `agc/`, `cacore/`, `lfcore/`, `model/`, `secore/`, `simu/`
- Move: `algorithm_parameters.py`, `efile_read.py`, `unit_system.py`, `hybrid_net_flow.py`

- [ ] Create the package directory and `__init__.py`.
- [ ] Move core packages and root modules under `src/hybrid_power_system_analysis/`.
- [ ] Add missing `__init__.py` files to moved subpackages.
- [ ] Verify a package import: `python -c "import hybrid_power_system_analysis; import lfcore.ac_lf; import secore.ac_se"`.

### Task 3: Sort Scripts And Generated Artifacts

**Files:**
- Create directory: `scripts/`
- Create directory: `benchmarks/`
- Move root utility scripts into `scripts/` or `benchmarks/`
- Leave `data/`, `docs/`, `tests/`, `power_plan/`, parameter files, and sample `.e` files in place.

- [ ] Move benchmark scripts to `benchmarks/`.
- [ ] Move generation/comparison utility scripts to `scripts/`.
- [ ] Keep generated outputs ignored rather than tracked by new package configuration.
- [ ] Verify no importable project module remains at repository root except intentional config files.

### Task 4: Update Tests And Source Path References

**Files:**
- Modify tests that read source files by old relative paths.
- Modify script command references that call moved source files directly.

- [ ] Replace old source path checks such as `model/ac_array_model.py` with `src/hybrid_power_system_analysis/model/ac_array_model.py`.
- [ ] Replace command references such as `lfcore/ac_lf.py` with package/module-compatible paths where practical.
- [ ] Run focused tests for packaging-sensitive imports.
- [ ] Run the full pytest suite or report the first concrete blocker.

### Task 5: Verify And Summarize

**Files:**
- No new files expected beyond config, docs, moved package files, scripts, and benchmarks.

- [ ] Run `python -m pytest --collect-only -q`.
- [ ] Run `python -m pytest tests/test_network_factory_loading.py tests/test_hybrid_net_flow_self_contained.py -q`.
- [ ] Run `python -m ruff check .` if ruff is available; if not, report the missing tool and the configured command.
- [ ] Summarize moved paths, config added, and verification results.

