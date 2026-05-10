# Qinling Network Diagram Design

## Goal

Generate two layered diagrams for `data/model/hybrid/qinling.e`:

1. A topology diagram that reads like a simplified hybrid single-line diagram.
2. A power-flow diagram that reuses the same layout and overlays solved LF results.

## Scope

- Use the existing hybrid network and hybrid LF pipeline.
- Do not change LF/SE/model behavior.
- Produce static `SVG` and `PNG` outputs.

## Layout

- Build topology from bus-level objects after hybrid topology preparation.
- Split AC and DC buses into separate visual bands:
  - AC buses on the left.
  - DC buses on the right.
  - DCAC converters in the middle.
- Use a deterministic spring layout inside each side, then normalize into fixed side ranges.
- Keep one shared position map for both diagrams.

## Visual Encoding

### Topology diagram

- AC buses: blue circles
- DC buses: orange squares
- AC branches: blue lines
- AC transformers: purple lines
- DC branches: orange lines
- DCDC converters: orange dashed lines
- DCAC converters: green links across AC/DC sides

### Power-flow diagram

- Reuse the topology layout.
- Show node operating values:
  - AC: `V` and `angle`
  - DC: `V`
- Show branch and converter flow labels:
  - AC branches / transformers: `P/Q`
  - DC branches / DCDC: `P`
  - DCAC: `Pdc / Pac / Qac`
- Use line width and color intensity to reflect flow magnitude.

## Outputs

- `artifacts/qinling_topology.svg`
- `artifacts/qinling_topology.png`
- `artifacts/qinling_powerflow.svg`
- `artifacts/qinling_powerflow.png`

## Implementation

- Add a standalone plotting script under `scripts/`.
- Script should support:
  - input case file
  - output directory
  - optional flat-start LF seed for the power-flow diagram

