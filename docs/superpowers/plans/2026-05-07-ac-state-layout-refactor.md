# AC State Layout Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `HybridStateEstimator` delegate the AC subproblem to `ACStateEstimator` with a compatible state layout, while keeping DC delegation and converter coupling logic inside `HybridStateEstimator`.

**Architecture:** Align the AC portion of the hybrid estimator with the AC estimator's state layout and state bookkeeping so AC Jacobian/evaluate/estimate can be reused directly. Keep `HybridStateEstimator` as the top-level orchestrator that splits AC, DC, and coupling measurements, maps between hybrid and subsystem state vectors, and recombines results.

**Tech Stack:** Python, NumPy, SciPy sparse matrices, existing `secore/*_se.py` estimators, `pytest`/`unittest`.

---

### Task 1: Freeze the current AC/Hybrid layout contract with regression tests

**Files:**
- Modify: `tests/test_hybrid_state_estimation.py`

- [ ] **Step 1: Write the failing regression tests**

```python
def test_ieee39_flat_start_ac_delegation_matches_ac_state_estimator(self):
    from secore.ac_se import ACStateEstimator
    from secore.hybrid_se import HybridStateEstimator

    kwargs = dict(
        e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
        meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
        flat_start=True,
    )

    ac_estimator = ACStateEstimator(**kwargs)
    hybrid_estimator = HybridStateEstimator(**kwargs)

    ac_result = ac_estimator.estimate(verbose=False, final_diagnostics=False)
    hybrid_result = hybrid_estimator.estimate(verbose=False)

    self.assertIsInstance(hybrid_estimator._ac_sub_estimator, ACStateEstimator)
    self.assertEqual(ac_result.iterations, hybrid_result.iterations)
    self.assertAlmostEqual(ac_result.objective, hybrid_result.objective, places=14)
    self.assertAlmostEqual(ac_result.residual_inf, hybrid_result.residual_inf, places=12)


def test_ieee3k_flat_start_hybrid_uses_ac_sub_state_layout(self):
    from secore.hybrid_se import HybridStateEstimator

    estimator = HybridStateEstimator(
        e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee3k.e",
        meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee3k.meas",
        flat_start=True,
    )

    self.assertTrue(hasattr(estimator, "ac_theta_state_col"))
    self.assertGreater(estimator.ac_theta_state_col.size, 0)
```

- [ ] **Step 2: Run the targeted tests to confirm current failure**

Run: `python -m unittest tests.test_hybrid_state_estimation.HybridStateEstimationTest.test_ieee39_flat_start_ac_delegation_matches_ac_state_estimator_path tests.test_hybrid_state_estimation.HybridStateEstimationTest.test_ieee3k_flat_start_hybrid_uses_ac_sub_state_layout`
Expected: FAIL because the hybrid AC path still uses its own compact layout.

- [ ] **Step 3: Keep the tests as the contract for the refactor**

No code yet in this task.

- [ ] **Step 4: Re-run the targeted tests after implementation**

Run the same unittest command.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_hybrid_state_estimation.py
git commit -m "test: lock AC hybrid state layout delegation contract"
```

### Task 2: Expose AC state-layout helpers from `ACStateEstimator`

**Files:**
- Modify: `secore/ac_se.py`

- [ ] **Step 1: Add explicit layout helpers around the existing AC state vector**

Implement small accessors that describe the AC estimator state in a reusable way:

```python
def state_layout(self) -> dict[str, np.ndarray | int]:
    return {
        "state_labels": self.state_labels,
        "angle_col": self.angle_col,
        "voltage_col": self.voltage_col,
        "n_state": self.n_state,
        "references": self.references,
    }
```

- [ ] **Step 2: Add a mapping helper that converts AC nodes to state columns**

```python
def state_cols_for_nodes(self, nodes: Sequence) -> tuple[np.ndarray, np.ndarray]:
    theta_cols = np.asarray([int(self.angle_col[self.node_pos[node.idx]]) for node in nodes], dtype=np.int32)
    voltage_cols = np.asarray([int(self.voltage_col[self.node_pos[node.idx]]) for node in nodes], dtype=np.int32)
    return theta_cols, voltage_cols
```

- [ ] **Step 3: Keep the existing estimator behavior unchanged**

Do not alter the `initial_state()`, `evaluate()`, `jacobian()`, or `estimate()` contracts.

- [ ] **Step 4: Run the AC estimator tests**

Run: `python -m unittest tests.test_ac_state_estimation tests.test_ac_ppc_flow`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add secore/ac_se.py
git commit -m "refactor: expose AC state layout helpers"
```

### Task 3: Rebuild the hybrid AC bookkeeping around the AC estimator layout

**Files:**
- Modify: `secore/hybrid_se.py`

- [ ] **Step 1: Replace the hybrid-local AC compact layout with AC-estimator-compatible mappings**

Build AC state column maps from `ACStateEstimator`'s layout instead of using a parallel hybrid-only AC layout.

```python
def _build_ac_state_adapter(self) -> None:
    ac = self._ac_sub_estimator
    self.ac_theta_state_col, self.ac_voltage_state_col = ac.state_cols_for_nodes(self.ac_nodes)
    self.ac_state_labels = list(ac.state_labels)
    self.ac_n_state = int(ac.n_state)
```

- [ ] **Step 2: Remove direct dependence on the old AC compact packing in hybrid code paths**

Update:
- `_build_estimation_state_layout()`
- `_build_state_column_lookup_arrays()`
- `_expand_state()`
- `_pack_estimation_state()`
- `_write_state()`

so AC rows/columns are delegated through the AC estimator instead of being rebuilt in hybrid-specific form.

- [ ] **Step 3: Preserve DC and coupling layout unchanged**

Do not disturb:
- DC node state handling
- DCDC state handling
- DCAC/ACAC coupling state handling
- zero-branch and breaker pseudo-measurement logic

- [ ] **Step 4: Run focused hybrid regression tests**

Run:
`python -m unittest tests.test_hybrid_state_estimation.HybridStateEstimationTest.test_ieee39_flat_start_ac_delegation_matches_ac_state_estimator_path tests.test_hybrid_state_estimation.HybridStateEstimationTest.test_mixed_network_reuses_dc_state_estimator_jacobian_block tests.test_hybrid_state_estimation.HybridStateEstimationTest.test_hybrid_jacobian_uses_direct_derivatives_without_repeated_evaluation`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add secore/hybrid_se.py
git commit -m "refactor: align hybrid AC layout with AC estimator"
```

### Task 4: Delegate pure AC hybrid cases directly to `ACStateEstimator`

**Files:**
- Modify: `secore/hybrid_se.py`

- [ ] **Step 1: Route pure AC cases through the AC estimator path**

When the hybrid case has no DC side and no cross-domain coupling, let `HybridStateEstimator` call `ACStateEstimator` for `initial_state()`, `evaluate()`, `jacobian()`, `jacobian_sparse()`, `observability_analysis()`, and `estimate()`.

```python
def _delegate(self):
    if self._sub_estimators_enabled and self._ac_sub_estimator is not None and not self.dc_nodes and not (self.dcac_by_name or self.acac_by_name):
        return self._ac_sub_estimator
    if self._sub_estimators_enabled and self._dc_sub_estimator is not None and not self.ac_nodes and not (self.dcac_by_name or self.acac_by_name):
        return self._dc_sub_estimator
    return None
```

- [ ] **Step 2: Keep hybrid-only measurement handling out of the delegated AC path**

The delegate path must not rebuild hybrid-specific AC zero-branch, converter, or DC constraint rows.

- [ ] **Step 3: Verify pure AC result parity**

Run:
`python -m unittest tests.test_hybrid_state_estimation.HybridStateEstimationTest.test_ieee39_flat_start_ac_delegation_matches_ac_state_estimator_path`

Expected: PASS with identical AC result metrics.

- [ ] **Step 4: Commit**

```bash
git add secore/hybrid_se.py
git commit -m "feat: delegate pure ac hybrid cases to ac estimator"
```

### Task 5: Re-verify mixed AC/DC hybrid behavior and close out the refactor

**Files:**
- Modify: `tests/test_hybrid_state_estimation.py`
- Modify: `secore/hybrid_se.py` if needed

- [ ] **Step 1: Add or update mixed-network tests that cover AC/DC coexistence**

Include at least one mixed-case regression that proves:
- AC side still evaluates correctly
- DC delegation still works
- converter coupling rows are still assembled inside `HybridStateEstimator`

```python
def test_qinling_mixed_network_still_uses_coupling_rows(self):
    from secore.hybrid_se import HybridStateEstimator

    estimator = HybridStateEstimator(
        e_file=ROOT_DIR / "data" / "model" / "hybrid" / "qinling.e",
        meas_file=ROOT_DIR / "data" / "meas" / "hybrid" / "qinling.meas",
        flat_start=True,
    )

    self.assertGreaterEqual(len(estimator.active_measurements), estimator.n_state)
    self.assertTrue(any(meas.device_type in ("DCACConverter", "ACACConverter") for meas in estimator.active_measurements))
```

- [ ] **Step 2: Run the full hybrid estimator test file**

Run: `python -m unittest tests.test_hybrid_state_estimation`
Expected: PASS.

- [ ] **Step 3: Run one AC and one mixed-case smoke test**

Run:
`python -m unittest tests.test_ac_state_estimation`

Run:
`python -m unittest tests.test_hybrid_state_estimation.HybridStateEstimationTest.test_qinling_hybrid_state_estimation_with_converter_measurements`

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add secore/hybrid_se.py tests/test_hybrid_state_estimation.py
git commit -m "refactor: delegate ac hybrid path to ac estimator"
```

## Self-Review

- AC layout reuse is explicit and test-backed.
- DC delegation remains intact and untouched unless a test proves otherwise.
- Hybrid-only coupling logic stays in `HybridStateEstimator`.
- No placeholder requirements remain.
- The plan is narrow enough to complete in one implementation cycle.
