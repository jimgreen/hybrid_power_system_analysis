# Src Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean production Python code under `src/hybrid_power_system_analysis` by removing clearly unused code and adding concise comments where control flow or data layout is non-obvious.

**Architecture:** Use a conservative static scan over production code plus tests to build a candidate list of unused top-level definitions, then manually review each candidate before deleting. Keep behavior stable by only touching production `.py` files and by validating with targeted compile/test passes after edits.

**Tech Stack:** Python 3.11, `ast`, `unittest`, package code under `src/hybrid_power_system_analysis`

---

### Task 1: Build cleanup candidate inventory

**Files:**
- Modify: `docs/superpowers/plans/2026-05-09-src-cleanup.md`
- Inspect: `src/hybrid_power_system_analysis/**/*.py`
- Inspect: `tests/**/*.py`

- [ ] Scan production Python files excluding `*_benchmark.py`, `simu/*`, and `__pycache__`.
- [ ] Build a conservative definition/reference inventory for top-level functions, classes, and constants.
- [ ] Exclude dynamic or exported entry points (`main`, `__all__`, CLI hooks, `getattr`-driven hooks, dataclasses, public API types).
- [ ] Produce a reviewed delete-candidate list before editing code.

### Task 2: Remove unused production code

**Files:**
- Modify: exact files selected from `src/hybrid_power_system_analysis/**/*.py`

- [ ] Delete only definitions with clear zero-use evidence across production code and tests.
- [ ] Remove now-dead helper branches and imports created by those deletions.
- [ ] Keep changes scoped; do not restructure unrelated modules.

### Task 3: Add targeted comments

**Files:**
- Modify: exact files selected from `src/hybrid_power_system_analysis/**/*.py`

- [ ] Add short orienting comments only around complex prepare/topology/layout/cache logic.
- [ ] Do not add narration comments for obvious assignments or loops.

### Task 4: Verify cleanup

**Files:**
- Verify: updated production files and affected tests

- [ ] Run `python -m py_compile` on touched production files.
- [ ] Run targeted `unittest` modules covering touched subsystems.
- [ ] Summarize deleted definitions and comment additions.
