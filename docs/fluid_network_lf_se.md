# Heat, Gas, Hydrogen, and Steam LF/SE

## Scope

The new solvers implement steady-state network calculations with the same high-level lifecycle used by the electrical solvers:

1. Parse an E file once into an array-oriented network.
2. `prepare()` topology, islands, state layout, and sparse incidence arrays.
3. `run()` the sparse Newton load flow or WLS state estimator.
4. Store device results in an LF result object or an `EstimateResult` plus `SEResult`.

The project name `hydro` denotes hydrogen in these modules.

## Gas and Hydrogen Equations

For a passive pipe or open valve from node `i` to node `j`:

```text
q_ij = C_ij sign(pi_i - pi_j) sqrt(abs(pi_i - pi_j))
pi_i = p_i^2
```

`C_ij` is the E-file conductance multiplied by the medium factor. Gas and hydrogen share the sparse equation kernel but retain independent density, compressibility, molar mass, temperature, and optional `flow_factor` values.

Each non-pressure-controlled node contributes one mass-balance equation. A ratio-controlled compressor contributes one pressure-ratio equation and one explicit compressor-flow state. Fixed-flow devices contribute directly to node balance.

Supported E blocks are:

- `GasMedium`, `GasNode`, `GasSource`, `GasLoad`, `GasPipe`, `GasValve`, `GasCompressor`
- `HydroMedium`, `HydroNode`, `HydroSource`, `HydroLoad`, `HydroPipe`, `HydroValve`, `HydroCompressor`

## Heat Equations

Heat-network hydraulic flow uses pressure rather than pressure squared:

```text
q_ij = C_ij sign(p_i - p_j) sqrt(abs(p_i - p_j))
```

After hydraulic convergence, pipe supply and return attenuation is:

```text
T_out = T_ambient + (T_in - T_ambient) exp(-heat_loss / abs(q))
```

Heat networks support both implicit and explicit return-water layouts. The two layouts can coexist on different hydraulic components and can be thermally coupled by a three-port heat exchanger.

### Implicit Return Water

The compact layout keeps the existing E-file contract:

- `HeatSource.node`: one combined supply/return node
- `HeatLoad.node`: one combined supply/return node
- `HeatExchanger.i_node/j_node`: one primary-side node and one secondary-side node

Every implicit `HeatNode` has separate `T_SUPPLY` and `T_RETURN` states. The return hydraulic network is not listed in the E file; it is assumed to be isomorphic to the supply network, while return heat transport is solved in the reverse direction.

### Explicit Return Water

The detailed layout lists both the supply and return hydraulic networks:

- `HeatSource.supply_node/return_node`: water enters at the return port and leaves at the supply port
- `HeatLoad.supply_node/return_node`: water enters at the supply port and leaves at the return port
- `HeatExchanger.primary_supply_node/primary_return_node/secondary_return_node/secondary_supply_node`: four physical terminals

The explicit-return parser also accepts modeling-platform aliases `HeatSource2`
and `HeatLoad2`. Their `i_node/j_node` fields are mapped to
`supply_node/return_node`; canonical port names remain valid in either block.
Measurements named `HeatSource2` or `HeatLoad2` are normalized to the shared
`HeatSource`/`HeatLoad` calculation devices.

Every explicit `HeatNode` has one `TEMPERATURE` state. Supply and return pipes are both ordinary `HeatPipe`, `HeatValve`, or `HeatPump` rows with their actual terminal nodes. The source, load, and heat-exchanger terminal flows participate directly in node mass balance.

An explicit pressure-controlled heat source fixes its supply-port pressure. Its circulation flow is solved from the combined supply/return KCL. If several such sources control the same supply node, `flow_set`, directional limits, and `alpha` determine the flow sharing.

Ideal fixed-flow sources, loads, and exchanger sides transfer mass but do not impose a pressure-drop equation. Pressure-reference islands are therefore formed from pipes, valves, and pumps; a component without a pressure source receives the existing node-pressure anchor. A separate thermal-circuit island map includes explicit device terminal connections.

### Heat Exchanger Ports

`HeatExchanger` supports three layouts:

- Two ports: `i_node`, `j_node`; both sides use implicit return water.
- Four ports: all four explicit terminal fields; both sides use explicit return water.
- Three ports: one side uses its explicit supply/return pair and the other side uses `i_node` or `j_node` as an implicit supply/return node.

Each side must define exactly one complete layout. A pipe-connected hydraulic component cannot mix explicit and implicit nodes. The heat exchanger couples energy only and never merges its primary and secondary hydraulic islands.

Sparse mixing equations determine all compact temperature states. Heat loads withdraw specified mass flow and heat power; in the implicit layout the local load outlet temperature can differ from the mixed return-node temperature.

Supported E blocks are `HeatMedium`, `HeatNode`, `HeatSource`, `HeatSource2`,
`HeatLoad`, `HeatLoad2`, `HeatPipe`, `HeatValve`, `HeatPump`, and
`HeatExchanger`.

The heat-exchanger thermal equations support:

- `EFFECTIVENESS`: `Q = effectiveness * min(m_primary, m_secondary) * cp * (Ts_primary - Tr_secondary)`
- `HEAT`: fixed primary heat transfer `heat_set`; the surrounding loop must provide an independent temperature anchor

`heat_loss` is the fraction of primary heat not delivered to the secondary side.

## Control Types

- Source: `PRESSURE` or `FLOW`
- Pipe: passive
- Valve: `OPEN`/`PASSIVE`, `FLOW`, or `CLOSED`
- Gas/hydrogen compressor: `RATIO` or `FLOW`
- Heat pump: `GAIN` or `FLOW`

If several pressure sources control the same node, their pressure setpoints are averaged before forming the pressure equation. Their solved source flow is allocated from `flow_set`, then by directional headroom and `alpha`, reusing the existing limited-residual allocation contract.

## State Estimation

Supported measurement types are:

- Pressure: `PRESSURE`, `PRESSURE_FROM`, `PRESSURE_TO`
- Flow: `FLOW`, `FLOW_FROM`, `FLOW_TO`
- Heat temperature: `TEMPERATURE`, `T_FROM`, `T_TO`, `T_SUPPLY`, `T_RETURN`, `TS_FROM`, `TS_TO`, `TR_FROM`, `TR_TO`
- Heat load: `HEAT`
- Heat exchanger: `FLOW_FROM`, `FLOW_TO`, `TS_FROM`, `TR_FROM`, `TR_TO`, `TS_TO`, and `HEAT`

Explicit heat nodes contribute one `TEMPERATURE` state; implicit heat nodes contribute separate `T_SUPPLY` and `T_RETURN` states. The compact mixed-mode mapping prevents duplicate explicit-node temperature states from making the WLS gain matrix singular.

SE performs observability analysis before WLS. If the measurement Jacobian is rank deficient, it greedily adds only low-weight node-pressure, node-temperature, node-enthalpy, or regulated-edge-flow pseudo measurements that increase rank. The final calculation populates both `EstimateResult` and `SEResult`, including pseudo, bad-data, normal, and prefiltered measurement statistics.

## Steam Network

Steam hydraulic flow reuses the compressible pressure-squared equation used by the gas kernel. Steam energy transport uses node specific enthalpy rather than a district-heating supply/return pair:

```text
h_out = h_ambient + (h_in - h_ambient) exp(-heat_loss / abs(m))
```

Incoming steam streams and positive source injections are mixed by mass-weighted enthalpy. Approximate temperature is reported from the configured reference state:

```text
T = T_reference + (h - h_reference) / cp
```

This temperature conversion is a steady single-phase engineering approximation, not a replacement for a full steam-table property package. Pressure, mass flow, and enthalpy remain the calculation states.

Supported E blocks are `SteamMedium`, `SteamNode`, `SteamSource`, `SteamLoad`, `SteamPipe`, `SteamValve`, and `SteamPressureReducer`. The reducer supports `RATIO` or `FLOW` control through the shared regulated-edge state layout.

Steam SE supports `PRESSURE`, `FLOW`, `FLOW_FROM`, `FLOW_TO`, `ENTHALPY`, `TEMPERATURE`, `H_FROM`, `H_TO`, `T_FROM`, `T_TO`, and load `HEAT` measurements. Missing node-enthalpy observability is restored with low-weight `SteamNode/ENTHALPY` pseudo measurements.

## Scale Benchmark

`scripts/check_fluid_scale_lf_se.py` generates independent heat, gas, hydrogen, and steam E files at 10, 50, 100, 500, 1000, 2000, and 5000 nodes. Every generated case contains a source, distributed loads, passive pipes, valves, and the medium-specific regulated device. Heat cases additionally contain explicit supply/return nodes, implicit return-water nodes, dual-port heat sources and loads, pumps, and a three-port heat exchanger.

For each case the benchmark:

1. Loads the generated E file and runs LF with a `1e-9` residual tolerance.
2. Generates an exact measurement snapshot from the LF result and writes a `.meas` file.
3. Reloads both network and measurements for flat-start SE.
4. Runs observability analysis, WLS, bad-data diagnostics, and `SEResult` generation.
5. Compares estimated pressure, edge flow, source flow, temperature, and enthalpy states against LF.

Each case runs in an independent Python process. Reports separate model loading, `prepare()`, LF solve, measurement generation, SE loading, observability, WLS, diagnostics, interpreter/import overhead, and total process time. Generated models are written below `data/model/<network>/`, measurement snapshots below `data/meas/<network>/`, and reports below `output/fluid_scale_benchmark/`.
