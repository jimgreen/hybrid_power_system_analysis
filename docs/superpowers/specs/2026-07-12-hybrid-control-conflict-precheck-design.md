# Hybrid Load-Flow Control-Conflict Precheck

## Goal

Run a deterministic control-configuration precheck before the hybrid Newton
system is assembled. The precheck must identify two model configurations that
are known to create redundant voltage-control equations:

1. An automatically selected AC PH generator and one or more other online PV
   generators connected through ideal AC elements to the same voltage
   component.
2. Two or more online `DCACConverter` devices using `DCV` control in the same
   energized DC topology island.

These findings are model-control conflicts, not load-flow divergence. By
default they block the calculation before the first Newton iteration. Callers
may explicitly allow the calculation to continue for diagnosis.

The precheck reports conflicts only. It must not rewrite E files, alter PPC
control codes, or modify device objects to resolve a conflict automatically.

## Scope

This feature is integrated into `HybridPowerFlowCalc`, including its normal
object-backed path, lightweight PPC-backed path, `from_file_fast()` factory,
result modes, and command-line interface.

The first version does not attempt to find every possible cause of a singular
Jacobian. It is deliberately limited to the two confirmed duplicate-control
patterns above. Jacobian rank analysis is not part of the precheck, and the
feature does not choose which user-configured controller should be changed.

## Public Behavior

`HybridPowerFlowCalc` and `HybridPowerFlowCalc.from_file_fast()` gain the
following argument:

```python
allow_control_conflicts: bool = False
```

The default is blocking behavior. The command-line interface gains the matching
flag:

```text
--allow-control-conflicts
```

When conflicts exist and the flag is false:

- `prepare()` raises `HybridControlConflictError` with the structured issues.
- `run()` catches that specific exception and returns `-2`.
- No Newton residual or Jacobian is built.
- `converged` is false, `iterations` is zero, and the result records that the
  calculation was blocked by precheck rather than classified as divergent.
- AC findings are exposed through `ac_errors`; DC findings are exposed through
  `dc_errors`.

When conflicts exist and the flag is true:

- The same structured issues are retained.
- AC findings are exposed through `ac_warnings`; DC findings are exposed
  through `dc_warnings`.
- Preparation and Newton iteration continue without changing any control mode.
- The final load-flow return code remains `0` for convergence or `-1` for
  non-convergence.

`--quiet` suppresses console output but never discards structured issues or
warning/error lists.

## Architecture

Add a focused module:

```text
src/hybrid_power_system_analysis/lfcore/hybrid_control_precheck.py
```

It contains a small immutable issue record and pure inspection functions:

```python
@dataclass(frozen=True)
class ControlConflictIssue:
    domain: str
    code: str
    island_id: int
    device_ids: tuple[int, ...]
    device_names: tuple[str, ...]
    node_ids: tuple[int, ...]
    setpoints: tuple[float, ...]
    message: str


def check_ac_voltage_control_conflicts(ac_calc) -> list[ControlConflictIssue]: ...


def check_dc_voltage_control_conflicts(calc) -> list[ControlConflictIssue]: ...


def run_hybrid_control_precheck(calc) -> list[ControlConflictIssue]: ...
```

The checks consume arrays and topology metadata already prepared by the AC and
DC subcalculators. They do not require a full `HybridPowerNetwork` object graph,
so the lightweight PPC path keeps its current startup characteristics.

`HybridPowerFlowCalc` stores the latest issues in `precheck_issues`. It also
stores the derived `ac_warnings`, `ac_errors`, `dc_warnings`, and `dc_errors`
lists used by result and command-line consumers.

`HybridLFResult` gains `precheck_issues` and receives the four warning/error
lists. Array and summary result modes expose a serializable `precheck` list so
that callers do not need to parse display strings.

## Preparation Sequence

`HybridPowerFlowCalc.prepare()` uses this order:

1. Prepare the AC PPC subcalculator and topology. This includes automatic PH
   selection and ideal AC component construction.
2. Prepare the DC PPC subcalculator and topology.
3. Map live DCAC converters to their active AC and DC solver terminals.
4. Run the AC and DC control-conflict checks.
5. Store and classify all issues.
6. If issues exist and `allow_control_conflicts` is false, raise
   `HybridControlConflictError` before global state-vector and Jacobian
   preparation.
7. Otherwise continue with ACAC preparation, global state-vector assembly, and
   Newton-system cache construction.

This ordering guarantees that the checks use the same runtime topology and
device eligibility rules as the equations they protect.

## AC Duplicate-Voltage Check

The AC check considers only generator rows that are online and attached to an
alive solver node. It reads the automatically promoted generator rows from
`_auto_slack_gen_rows` and maps those rows through the prepared generator
positions.

For each automatic PH generator, the check determines its ideal voltage
component from the AC solver's prepared `comp_nodes`. Those components are
formed from active ideal AC connections, including closed switches, closed
breakers, and `ACZeroBranch` devices.

An AC issue is emitted when all of the following are true:

- The component contains an automatically selected PH generator.
- Another online `ACGenerator` retains `PV` control.
- That PV generator is attached to a different solver node in the same ideal
  component.

Multiple generators attached directly to the same solver node are not reported
by this check. The AC solver creates one node-voltage specification in that
case, so it is not the redundant ideal-link equation pattern addressed here.

The issue code is:

```text
AC_CONTROL_CONFLICT
```

One issue is generated per affected automatic-PH ideal component. Its message
identifies the AC island, selected PH generator, duplicate PV generators,
nodes, and all `v_set` values. Equal voltage setpoints are described as
redundant constraints. Unequal values additionally state that the setpoints
conflict.

Open devices, offline generators, dead nodes, PV generators in other islands,
and PV generators outside the automatic PH's ideal component are ignored.

## DC Duplicate-DCV Check

The DC check works from the live converter arrays produced by
`_prepare_dcac_converters()`. A converter participates only when:

- Its own `run_stat` is online.
- Its AC terminal maps to an alive AC solver node.
- Its DC terminal maps to an alive DC solver node.
- Its normalized control code is `DCV`.

Each participating converter's original DC node is mapped through the prepared
DC `GridTopologyArrays.node_to_island_pos` data. Converters are grouped by the
corresponding energized `island_id`.

An issue is emitted for every DC island containing two or more participating
DCV converters. The issue code is:

```text
DC_CONTROL_CONFLICT
```

The message identifies the DC island, number of DCV controllers, converter
names and identifiers, DC nodes, and `v_dc_set` values. It recommends retaining
one principal DC-voltage controller and reviewing the remaining control modes,
but it does not select or modify a controller.

Converters in different DC islands are independent and are not reported
together. Offline converters and converters with inactive or unmapped
terminals are ignored.

## Error and Result Propagation

`HybridControlConflictError` derives from `RuntimeError` and exposes its issues
as an immutable sequence. Direct callers of `prepare()` therefore receive a
clear, inspectable failure instead of a later sparse-solver exception.

`run()` handles only this dedicated precheck exception. Other preparation
errors retain their existing behavior.

For a blocked run, `run()` creates the result metadata needed by the selected
result mode without fabricating solved voltages or powers. Full-result printing
recognizes return code `-2`, prints the precheck section, reports that no Newton
iteration was attempted, and skips normal solved-value sections.

Console output uses these labels:

```text
[BLOCK][AC_CONTROL_CONFLICT]
[BLOCK][DC_CONTROL_CONFLICT]
```

When override is enabled, the labels become:

```text
[WARNING][AC_CONTROL_CONFLICT]
[WARNING][DC_CONTROL_CONFLICT]
```

Chinese explanatory text may follow the stable machine-readable code. The
machine-readable fields, rather than the localized message, are the API
contract.

## Testing

Unit tests use small repository-local PPC or E-file fixtures and cover:

- Automatic PH plus PV generators linked by a closed AC breaker.
- Automatic PH plus PV generators linked by a closed switch or
  `ACZeroBranch`.
- An open breaker removing the AC conflict.
- PV generators in different AC islands.
- Offline PV generators and dead AC nodes.
- Multiple generators directly attached to one node without a false positive.
- Equal and unequal `v_set` values with correct issue details.
- Two or more live DCV converters in one alive DC island.
- DCV converters in different DC islands.
- A single DCV converter.
- Offline DCAC converters and inactive AC or DC terminals.
- Correct names, identifiers, nodes, setpoints, island IDs, and issue counts.

Integration tests cover:

- Default `run()` returning `-2` with zero iterations.
- Verification that Newton-system construction is not called after a blocking
  finding.
- Direct `prepare()` raising `HybridControlConflictError`.
- Override mode retaining warnings and entering Newton execution.
- Full, array, summary, and none result-mode behavior.
- `from_file_fast()` argument forwarding.
- Command-line blocking and `--allow-control-conflicts` behavior.
- Existing conflict-free models retaining their prior return codes and output.

Final verification runs the focused tests, AC/DC/hybrid load-flow regressions,
the full maintained test command, Ruff, Python syntax compilation, and
`git diff --check`.

## Qinling Acceptance Check

After automated tests pass, run the unmodified file:

```text
C:\Users\wangbin\Desktop\qinling.e
```

With default settings, the expected behavior is:

- One AC conflict identifying `diesel generator-1` as the automatic PH and
  `diesel generator-2` through `diesel generator-4` as duplicate PV controls
  in the same ideal AC voltage component.
- One DC conflict identifying all 12 DCV `DCACConverter` devices in the main
  DC island.
- Return code `-2`, zero Newton iterations, and no divergence classification.

With `allow_control_conflicts=True`, the same two findings remain available as
warnings and the solver proceeds with the unchanged model controls. The source
E file is never modified.
