# AC Topology Island Automatic PH Generator Selection

## Goal

When an energized AC topology island has no explicit `V`, `SLACK`, or `PH`
generator, automatically select one online `PV` generator as the island angle
reference so that the island can participate in load-flow calculation.

The selection is runtime-only. The source E file is not rewritten and an
explicit PH/slack generator always takes precedence.

## Generator Capacity

`ACGenerator` gains an optional `p_max` field in the E-file, object model, and
array model. Named-unit E-file values are converted to per unit using `p_base`.
Missing `p_max` values are represented as unavailable rather than zero so that
legacy files remain distinguishable from generators whose declared maximum is
actually zero.

For each island without an explicit reference, online PV generators are ranked
in this order:

1. Largest available `p_max`.
2. If the island has no available `p_max`, largest `abs(p_set)`.
3. If still tied, largest `alpha`.
4. If still tied, smallest generator `idx`.

If an island contains no online PV generator, it retains the existing dead-
island/error behavior.

## Runtime Topology Behavior

The AC topology preparation step identifies each generator's island before
computing the island alive mask. It first records all explicit slack references,
then selects one automatic reference for every remaining island that contains
an online PV generator.

Selected generator rows are stored as runtime PPC metadata. The topology uses
their buses as island references, and `ACPowerFlowCalc` treats those rows as PH
generators when building voltage and angle constraints. The original generator
control code remains `PV`, so reading or running a model does not silently edit
the source definition.

This is performed independently for every topology island. A system with ten
isolated wind-generator/converter islands and one main AC island therefore gets
eleven reference buses when none were explicitly configured.

## Compatibility

- Existing E files without `p_max` continue to load.
- Existing files with explicit `V`, `SLACK`, or `PH` generators retain their
  current reference selection.
- AC-only and hybrid load flow share the same PPC topology path and therefore
  receive identical automatic-reference behavior.
- Object-model import/export and array-model import/export preserve `p_max`
  when present.
- The preferred base-information block is `<Model>`; legacy `<PowerBase>`
  remains accepted.

## Validation

Tests will cover:

- `<Model>` and legacy `<PowerBase>` unit settings.
- `p_max` import, scaling, object conversion, and export.
- Largest-`p_max` selection in an island with multiple PV generators.
- Fallback ranking by `abs(p_set)`, `alpha`, and generator `idx`.
- One automatically selected PH generator per island.
- Explicit slack generators overriding automatic selection.
- Islands without an online PV generator remaining unavailable.
- AC and hybrid load-flow convergence with automatically selected references.
- A direct run of `C:\Users\wangbin\Desktop\qinling.e`, including convergence,
  iteration count, node-voltage range, and power summary.
