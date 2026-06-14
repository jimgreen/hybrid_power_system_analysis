# AC State Layout Refactor Design

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the hybrid estimator delegate the AC subsystem to `ACStateEstimator` using a compatible AC state layout, while preserving the existing DC delegation and hybrid coupling behavior.

**Architecture:** Treat `HybridStateEstimator` as an orchestrator. AC and DC subsystem math should live in their respective estimators, while `HybridStateEstimator` owns cross-domain measurement routing, converter coupling rows, and state-vector composition. The main refactor is to make the hybrid AC bookkeeping use the same state layout and node-to-column mapping as `ACStateEstimator`, so the AC estimator can own AC evaluation and Jacobian assembly directly.

**Tech Stack:** Python, NumPy, SciPy sparse matrices, existing `secore/ac_se.py`, `secore/dc_se.py`, `secore/hybrid_se.py`, and unittest-based regression tests.

---

## Current Problem

`HybridStateEstimator` currently keeps a compact AC-specific layout that differs from `ACStateEstimator`'s internal layout. That means the hybrid code still has to duplicate AC state packing, AC column mapping, and AC derivative assembly. The result is that AC cannot be delegated cleanly, and hybrid AC code drifts from the tested AC path.

## Design Principles

- AC subsystem logic should have one owner: `ACStateEstimator`.
- Hybrid should orchestrate, not reimplement AC math.
- DC delegation already works and should remain intact.
- Coupling logic between AC and DC should stay in `HybridStateEstimator`.
- State mapping must stay explicit and testable.

## Proposed Shape

### 1. AC state layout as a reusable contract

`ACStateEstimator` already defines the authoritative AC state ordering, reference rebasing, and node-to-state-column mapping. The refactor will expose small helper accessors so hybrid code can consume those mappings without duplicating the layout rules.

Expected helpers:
- a state-layout description
- a node-to-state-column mapping for theta and voltage
- no changes to the numerical behavior of `initial_state()`, `evaluate()`, `jacobian()`, or `estimate()`

### 2. Hybrid AC bookkeeping becomes adapter-based

`HybridStateEstimator` will keep its overall hybrid state vector, but its AC slice will be mapped through the AC estimator's layout. This means:
- AC theta/voltage columns in hybrid are derived from AC estimator state metadata
- AC measurements are routed to the AC sub-estimator when the case is pure AC
- mixed cases can still use the hybrid orchestrator for coupling rows

### 3. Pure AC hybrid cases delegate directly

For cases with no DC side and no cross-domain converter coupling, `HybridStateEstimator` should call into `ACStateEstimator` for:
- initial state
- evaluation
- Jacobian assembly
- observability
- estimation

This makes pure AC behavior identical to the AC estimator path and keeps the hybrid wrapper thin.

### 4. Mixed AC/DC cases keep hybrid coupling logic

Mixed-network cases still need hybrid-side assembly for:
- converter coupling rows
- cross-domain pseudo-measurements
- DC/AC interaction constraints

However, the AC-side derivative calculations should now be driven by AC estimator-compatible layout mappings rather than a duplicate hybrid AC layout.

## Testing Strategy

The refactor should be validated with three layers of regression tests:

1. Pure AC delegation parity
   - `ieee39` flat-start hybrid result matches AC estimator result

2. Mixed network preservation
   - `qinling` still converges
   - DC delegation remains active
   - coupling rows are still present

3. Layout contract checks
   - hybrid AC column mappings are populated
   - AC angle and voltage columns are stable under flat start and non-flat start

## Out of Scope

- Rewriting DC state layout
- Changing converter physics
- Changing measurement file formats
- Removing hybrid coupling logic

## Risks

- If AC layout metadata is exposed too loosely, hybrid and AC could diverge again later.
- If pure AC delegation is wired too early, mixed-case code paths may accidentally inherit the delegated behavior.
- If the layout adapter is incomplete, a few AC measurement types may still be assembled by hybrid code instead of the AC estimator.

## Acceptance Criteria

- Pure AC hybrid cases use `ACStateEstimator` directly.
- AC Jacobian and evaluation behavior match the AC estimator path.
- Mixed AC/DC hybrid cases still run end-to-end.
- Existing DC delegation regression tests still pass.
- No duplicated AC state layout remains in the hybrid path.

