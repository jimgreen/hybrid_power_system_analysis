from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass(frozen=True, slots=True)
class StateMeta:
    """Structured description for one state vector column."""

    side: str
    kind: str
    device_type: str
    device_name: str
    terminal: str = ""
    component: str = ""
    legacy_label: str = ""


def state_labels_from_metadata(state_meta: Sequence[StateMeta]) -> List[str]:
    """Return display labels derived from structured state metadata."""
    labels: List[str] = []
    for idx, meta in enumerate(state_meta):
        if meta.legacy_label:
            labels.append(meta.legacy_label)
            continue
        parts = [meta.side, meta.kind, meta.device_type, meta.device_name, meta.terminal, meta.component]
        labels.append(":".join(part for part in parts if part) or f"state_{idx}")
    return labels


def state_sides_from_metadata(state_meta: Sequence[StateMeta]) -> List[str]:
    return [meta.side for meta in state_meta]


def state_meta_at(state_meta: Sequence[StateMeta], state_idx) -> Optional[StateMeta]:
    try:
        idx = int(state_idx)
    except (TypeError, ValueError):
        return None
    if idx < 0 or idx >= len(state_meta):
        return None
    return state_meta[idx]


def with_legacy_label(meta: StateMeta, legacy_label: str, side: Optional[str] = None) -> StateMeta:
    return StateMeta(
        meta.side if side is None else side,
        meta.kind,
        meta.device_type,
        meta.device_name,
        terminal=meta.terminal,
        component=meta.component,
        legacy_label=legacy_label,
    )
