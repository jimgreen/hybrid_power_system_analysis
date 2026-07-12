# Automatic PH Generator Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow every AC topology island without an explicit angle reference to promote its highest-capacity online PV generator to a runtime PH generator, then run the desktop `qinling.e` case successfully.

**Architecture:** Extend the AC generator data contract with optional per-unit `p_max`, select one deterministic runtime reference row per unreferenced island during PPC topology preparation, and pass those selected rows to `ACPowerFlowCalc` through PPC metadata. Keep the source E model unchanged and retain `<PowerBase>` compatibility while preferring `<Model>`.

**Tech Stack:** Python 3, NumPy PPC arrays, SciPy topology helpers, unittest/pytest, Ruff.

---

## File Map

- `src/hybrid_power_system_analysis/model/array_common.py`: Read base units from `<Model>` with legacy `<PowerBase>` fallback.
- `src/hybrid_power_system_analysis/unit_system.py`: Normalize object models using the same base-block precedence and scale generator `p_max`.
- `src/hybrid_power_system_analysis/model/ac_model.py`: Add optional `p_max` to the AC generator object contract.
- `src/hybrid_power_system_analysis/model/ac_array_model.py`: Add `p_max` to PPC arrays and E/object/MATPOWER conversions.
- `src/hybrid_power_system_analysis/model/topology.py`: Select one automatic PH generator row per AC island and attach runtime metadata.
- `src/hybrid_power_system_analysis/lfcore/ac_lf.py`: Treat automatic PH rows exactly like explicit slack rows while building equations.
- `tests/test_named_unit_files.py`: Cover `<Model>` base blocks and named-unit `p_max` scaling.
- `tests/test_ac_ppc_flow.py`: Cover `p_max` E/PPC/object/MATPOWER round trips and load-flow use of automatic references.
- `tests/test_topology_helpers.py`: Cover per-island ranking, explicit-reference precedence, and no-PV behavior.
- `docs/ac_lf.md`: Document optional `p_max` and automatic PH selection.

### Task 1: Finish `<Model>` Base-Block Compatibility

**Files:**
- Modify: `src/hybrid_power_system_analysis/model/array_common.py:249`
- Modify: `src/hybrid_power_system_analysis/unit_system.py:5-85`
- Test: `tests/test_named_unit_files.py`

- [ ] **Step 1: Keep the existing failing-first tests for array and object models**

```python
def test_array_unit_settings_accept_model_block(self):
    rows = {
        "Model": {
            "header_list": ["path", "name", "p_base", "u_unit", "p_unit", "i_unit"],
            "rows": [["IEEE", "qinling", "100", "V", "kW", "A"]],
        }
    }
    self.assertEqual((100.0, 1000.0, 1.0, 1000.0, 100.0), _base_from_rows(rows))
```

- [ ] **Step 2: Verify the tests pass with `<Model>` preferred and `<PowerBase>` retained**

Run: `python -m pytest tests/test_named_unit_files.py -k "model_block or power_base_scales_file_units" -q`

Expected: `3 passed`.

- [ ] **Step 3: Commit only the base-block compatibility files**

```powershell
git add -- src/hybrid_power_system_analysis/model/array_common.py src/hybrid_power_system_analysis/unit_system.py tests/test_named_unit_files.py
git commit -m "fix: accept Model power base definitions"
```

### Task 2: Add Optional AC Generator `p_max`

**Files:**
- Modify: `src/hybrid_power_system_analysis/model/ac_model.py:113-125,360-375,647-650`
- Modify: `src/hybrid_power_system_analysis/model/ac_array_model.py:140-151,588-611,1213-1230,1386-1405,1601-1613,1887-1898`
- Modify: `src/hybrid_power_system_analysis/unit_system.py:234-237`
- Test: `tests/test_named_unit_files.py`
- Test: `tests/test_ac_ppc_flow.py`

- [ ] **Step 1: Write failing tests for E/PPC/object and named-unit conversion**

Add a temporary E model whose generator header contains `p_max` and assert:

```python
ppc = build_ac_ppc_from_e_file(case_path)
self.assertAlmostEqual(ppc["gen"][0, GEN_COLS["p_max"]], 0.8)

network = build_ac_network_from_ppc(ppc)
self.assertAlmostEqual(network.generators[0].p_max, 0.8)

round_trip = build_ac_ppc_from_network(network)
self.assertAlmostEqual(round_trip["gen"][0, GEN_COLS["p_max"]], 0.8)
```

Also assert a legacy generator without `p_max` stores `NaN`, and object named-unit normalization divides `p_max` by `p_base`.

- [ ] **Step 2: Run the new tests and verify they fail because `GEN_COLS["p_max"]` does not exist**

Run: `python -m pytest tests/test_ac_ppc_flow.py tests/test_named_unit_files.py -k "p_max" -q`

Expected: FAIL with missing `p_max` support.

- [ ] **Step 3: Extend the generator object without breaking positional callers**

Use the compatible signature:

```python
class ACGenerator:
    def __init__(self, idx, node, control_type, p_set, q_set, v_set, alpha=None, run_stat=1, p_max=None):
        ...
        self.p_max = p_max
```

Add `p_max=None` to `_AC_ROW_DEFAULT_ATTRS["ACGenerator"]` and to `add_generator()` after `run_stat`.

- [ ] **Step 4: Append `p_max` to PPC without changing existing column numbers**

```python
GEN_COLS = {
    ...
    "current": 10,
    "p_max": 11,
}
```

E files scale `p_max` by `p_base`; missing values become `np.nan`. Object conversion preserves `None` as `np.nan`. MATPOWER import reads `PMAX/baseMVA`, and MATPOWER export writes finite `p_max` or retains the existing large unconstrained default.

- [ ] **Step 5: Scale object-model `p_max` with other generator powers**

```python
_scale_power_attrs_in_dict(gen, ("p_set", "q_set", "p_max", "p", "q"), p_base)
```

- [ ] **Step 6: Run focused and conversion regression tests**

Run: `python -m pytest tests/test_ac_ppc_flow.py tests/test_named_unit_files.py -k "p_max or build_ac_ppc_from_network or matpower" -q`

Expected: all selected tests pass.

- [ ] **Step 7: Commit the generator capacity contract**

```powershell
git add -- src/hybrid_power_system_analysis/model/ac_model.py src/hybrid_power_system_analysis/model/ac_array_model.py src/hybrid_power_system_analysis/unit_system.py tests/test_named_unit_files.py tests/test_ac_ppc_flow.py
git commit -m "feat: add AC generator maximum capacity"
```

### Task 3: Select One Automatic PH Generator Per Island

**Files:**
- Modify: `src/hybrid_power_system_analysis/model/topology.py:822-1000`
- Test: `tests/test_topology_helpers.py`

- [ ] **Step 1: Write failing topology tests**

Build small PPC cases and assert:

```python
arrays = prepare_ac_topology_ppc(ppc)
self.assertEqual([selected_row], ppc["_auto_slack_gen_rows"].tolist())
self.assertTrue(arrays.island_alive_mask[island_pos])
self.assertEqual(expected_bus_pos, arrays.island_reference_bus_pos[island_pos])
```

Cover largest finite `p_max`, fallback to largest `abs(p_set)`, then largest `alpha`, then smallest `idx`; one selected row per island; no automatic row when an explicit slack/external reference exists; and a no-PV island remaining dead.

- [ ] **Step 2: Run the topology tests and verify they fail because metadata is absent**

Run: `python -m pytest tests/test_topology_helpers.py -k "auto_slack or automatic_ph" -q`

Expected: FAIL because `_auto_slack_gen_rows` is not produced.

- [ ] **Step 3: Add a deterministic row-selection helper**

Implement a NumPy-oriented helper that receives generator rows, generator island positions, online/valid masks, and the current island reference mask. For each unreferenced island, restrict candidates to online `CTRL_PV` rows and rank with this exact key:

```python
has_p_max = np.isfinite(p_max)
capacity = p_max if any candidate has_p_max else np.abs(p_set)
winner = min(candidates, key=lambda row: (-capacity[row], -alpha[row], idx[row]))
```

When no candidate in the island has `p_max`, use `abs(p_set)` as capacity. Missing/non-finite `alpha` ranks as zero.

- [ ] **Step 4: Attach runtime metadata and activate selected islands**

After explicit generator and external references are processed:

```python
ppc["_auto_slack_gen_rows"] = selected_rows
topology.island_alive_mask[selected_islands] = True
_mark_reference_bus(...)
```

Always replace stale metadata with the newly computed array, including an empty array.

- [ ] **Step 5: Run all topology helper tests**

Run: `python -m pytest tests/test_topology_helpers.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit topology selection**

```powershell
git add -- src/hybrid_power_system_analysis/model/topology.py tests/test_topology_helpers.py
git commit -m "feat: select automatic PH generators per island"
```

### Task 4: Apply Automatic PH Rows in AC Load Flow

**Files:**
- Modify: `src/hybrid_power_system_analysis/lfcore/ac_lf.py:966-1058`
- Test: `tests/test_ac_ppc_flow.py`
- Test: `tests/test_hybrid_net_flow_self_contained.py`

- [ ] **Step 1: Write a failing two-island load-flow test**

Construct an AC PPC with two disconnected energized islands, each containing PV generation and load but no explicit slack. Assert that `prepare()` marks both selected buses as `AC_NODE_TYPE_SLACK`, and that `run()` converges.

Also construct a hybrid case using the same AC PPC path and assert the selected rows survive hybrid topology preparation.

- [ ] **Step 2: Run the new tests and verify equation preparation does not yet honor automatic rows**

Run: `python -m pytest tests/test_ac_ppc_flow.py tests/test_hybrid_net_flow_self_contained.py -k "automatic_ph or auto_slack" -q`

Expected: FAIL because selected PV rows remain PV in the AC equation builder.

- [ ] **Step 3: Merge explicit and automatic slack masks in `_prepare_ppc_devices`**

```python
auto_rows = np.asarray(self.ppc.get("_auto_slack_gen_rows", ()), dtype=np.int32)
auto_mask = np.isin(self.ppc_gen_rows, auto_rows)
slack_mask = (controls == CTRL_SLACK) | auto_mask
```

Use the selected generator's `v_set` and bus angle for the voltage and angle constraints. Leave the underlying PPC `control_type` column unchanged.

- [ ] **Step 4: Run AC and hybrid focused tests**

Run: `python -m pytest tests/test_ac_ppc_flow.py tests/test_hybrid_net_flow_self_contained.py -k "automatic_ph or auto_slack or topology" -q`

Expected: all selected tests pass.

- [ ] **Step 5: Commit load-flow integration**

```powershell
git add -- src/hybrid_power_system_analysis/lfcore/ac_lf.py tests/test_ac_ppc_flow.py tests/test_hybrid_net_flow_self_contained.py
git commit -m "feat: apply automatic PH references in load flow"
```

### Task 5: Document and Validate the Qinling Case

**Files:**
- Modify: `docs/ac_lf.md`
- Test: `tests/test_named_unit_files.py`
- Test: `tests/test_ac_ppc_flow.py`
- Test: `tests/test_topology_helpers.py`
- Test: `tests/test_hybrid_net_flow_self_contained.py`

- [ ] **Step 1: Document the runtime behavior**

Add `p_max` to the `ACGenerator` parameter table and state that an island without explicit `V/SLACK/PH` selects the online PV generator by `p_max`, `abs(p_set)`, `alpha`, and `idx` in that order.

- [ ] **Step 2: Run the complete focused regression set**

Run:

```powershell
$env:PYTHONPATH='src\hybrid_power_system_analysis'
python -m pytest tests\test_named_unit_files.py tests\test_ac_ppc_flow.py tests\test_topology_helpers.py tests\test_hybrid_net_flow_self_contained.py -q
```

Expected: feature tests pass; any unrelated pre-existing state-estimator constructor failures must be reported separately rather than changed in this feature.

- [ ] **Step 3: Run static checks**

```powershell
python -m ruff check src\hybrid_power_system_analysis\model\array_common.py src\hybrid_power_system_analysis\unit_system.py src\hybrid_power_system_analysis\model\ac_model.py src\hybrid_power_system_analysis\model\ac_array_model.py src\hybrid_power_system_analysis\model\topology.py src\hybrid_power_system_analysis\lfcore\ac_lf.py tests\test_named_unit_files.py tests\test_ac_ppc_flow.py tests\test_topology_helpers.py tests\test_hybrid_net_flow_self_contained.py
python -m py_compile src\hybrid_power_system_analysis\model\array_common.py src\hybrid_power_system_analysis\unit_system.py src\hybrid_power_system_analysis\model\ac_model.py src\hybrid_power_system_analysis\model\ac_array_model.py src\hybrid_power_system_analysis\model\topology.py src\hybrid_power_system_analysis\lfcore\ac_lf.py
git diff --check
```

Expected: all commands exit zero.

- [ ] **Step 4: Execute the desktop Qinling model once**

```powershell
$env:PYTHONPATH='src\hybrid_power_system_analysis'
python src\hybrid_power_system_analysis\lfcore\hybrid_lf.py 'C:\Users\wangbin\Desktop\qinling.e'
```

Expected: the parser accepts `<Model>`, eleven AC islands receive runtime references, and the command reports either convergence with numerical summaries or a new precisely diagnosed model/solver error.

- [ ] **Step 5: Capture a concise machine-readable diagnostic summary without rerunning the solver**

Use the result produced by Step 4 to report selected generator indices, convergence status, iteration count, residual norm, AC/DC voltage ranges, generation/load totals, and warnings. Do not alter the source E file.

- [ ] **Step 6: Commit documentation and any final test-only adjustments**

```powershell
git add -- docs/ac_lf.md tests/test_named_unit_files.py tests/test_ac_ppc_flow.py tests/test_topology_helpers.py tests/test_hybrid_net_flow_self_contained.py
git commit -m "docs: describe automatic PH selection"
```
