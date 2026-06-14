# Qinling Coupled Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rework the Qinling diagram layout so the generated topology and power-flow figures visually read as one coupled hybrid system rather than two loosely aligned side panels.

**Architecture:** Keep the existing plotting script and LF pipeline, but replace the independent AC/DC spring layout with a converter-centered layered placement. Build a shared positional scaffold from AC/DC bus adjacency plus DCAC couplings, then reuse that scaffold for both topology and power-flow figures.

**Tech Stack:** Python, matplotlib, networkx, existing hybrid_model / hybrid_lf modules.

---

### Task 1: Inspect coupling structure and define placement anchors

**Files:**
- Modify: `D:\codex\elec_power_flow\hybrid_power_system_analysis\scripts\plot_qinling_diagrams.py`
- Create: `D:\codex\elec_power_flow\hybrid_power_system_analysis\docs\superpowers\plans\2026-05-09-qinling-coupled-layout.md`

- [ ] Identify how DCAC converters connect AC/DC buses and derive a stable group ordering.
- [ ] Define a shared bus placement algorithm that uses converter anchors first, then spreads remaining buses by side-local topology.
- [ ] Keep topology drawing and power-flow drawing on the same coordinate map.

### Task 2: Implement coupled layered layout in the plotting script

**Files:**
- Modify: `D:\codex\elec_power_flow\hybrid_power_system_analysis\scripts\plot_qinling_diagrams.py`

- [ ] Replace the old left/right independent spring layout with a coupled layout builder.
- [ ] Keep the existing rendering pipeline for devices and labels, changing only what is needed for placement readability.
- [ ] Preserve script CLI and output filenames.

### Task 3: Regenerate and verify output artifacts

**Files:**
- Modify: `D:\codex\elec_power_flow\hybrid_power_system_analysis\artifacts\qinling_topology.svg`
- Modify: `D:\codex\elec_power_flow\hybrid_power_system_analysis\artifacts\qinling_topology.png`
- Modify: `D:\codex\elec_power_flow\hybrid_power_system_analysis\artifacts\qinling_powerflow.svg`
- Modify: `D:\codex\elec_power_flow\hybrid_power_system_analysis\artifacts\qinling_powerflow.png`

- [ ] Run `python scripts\plot_qinling_diagrams.py`.
- [ ] Confirm artifacts are regenerated and visually better reflect the coupled hybrid network structure.
