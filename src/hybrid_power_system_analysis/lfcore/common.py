import numpy as np


def device_key(device) -> str:
    """Return the stable result-map key for a device object."""
    return str(getattr(device, "name", "") or getattr(device, "idx", id(device)))


def find_spanning_tree_edges(edges, n_nodes: int):
    """Return edge indices selected by Kruskal union-find for a spanning forest."""
    parent = np.arange(n_nodes, dtype=np.int32)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> bool:
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        parent[ry] = rx
        return True

    tree_indices = []
    for idx, (u, v) in enumerate(edges):
        if union(int(u), int(v)):
            tree_indices.append(idx)
    return tree_indices


def normalize_result_mode(result_mode: str, context: str) -> str:
    mode = str(result_mode or "full").strip().lower()
    if mode not in {"full", "summary", "array", "none"}:
        raise ValueError(f"Unsupported {context} result_mode: {result_mode!r}")
    return mode


def optional_ppc_vector(ppc, key: str, size: int, default=np.nan) -> np.ndarray:
    """Return a length-``size`` optional PPC vector without shape surprises."""
    values = np.asarray((ppc or {}).get(key, ()), dtype=np.float64).reshape(-1)
    result = np.full(int(size), float(default), dtype=np.float64)
    if values.size:
        result[: min(result.size, values.size)] = values[: result.size]
    return result


def allocate_limited_residual(
    baseline,
    target_total: float,
    *,
    lower=None,
    upper=None,
    alpha=None,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Apply setpoints first, then distribute the residual in two stages.

    The first stage respects directional operating limits. Finite headroom is
    multiplied by ``alpha``; saturated devices are removed and the residual is
    redistributed in subsequent passes. Missing limits remain unbounded and
    fall back to the alpha weight, preserving legacy E files without limits.

    If all available headroom is exhausted before the requested total is met,
    the second stage deliberately exceeds the limits and allocates the remaining
    residual by normalized ``alpha``. This keeps the solved network power-balanced
    while leaving the limit violation visible to higher-level controls.
    """
    result = np.asarray(baseline, dtype=np.float64).reshape(-1).copy()
    if result.size == 0:
        return result

    lower_values = (
        np.full(result.size, -np.inf, dtype=np.float64)
        if lower is None
        else np.asarray(lower, dtype=np.float64).reshape(-1)
    )
    upper_values = (
        np.full(result.size, np.inf, dtype=np.float64)
        if upper is None
        else np.asarray(upper, dtype=np.float64).reshape(-1)
    )
    if lower_values.size != result.size or upper_values.size != result.size:
        raise ValueError("分配上下限必须与基准设定数组长度一致")
    # NaN denotes an omitted optional limit. Keep explicit infinities intact.
    lower_values = np.where(np.isnan(lower_values), -np.inf, lower_values)
    upper_values = np.where(np.isnan(upper_values), np.inf, upper_values)

    if alpha is None:
        alpha_values = np.ones(result.size, dtype=np.float64)
    else:
        alpha_values = np.asarray(alpha, dtype=np.float64).reshape(-1)
        if alpha_values.size != result.size:
            raise ValueError("分配 alpha 必须与基准设定数组长度一致")
    alpha_values = np.nan_to_num(alpha_values, nan=0.0, posinf=0.0, neginf=0.0)
    alpha_values = np.maximum(alpha_values, 0.0)
    if not np.any(alpha_values > 0.0):
        alpha_values.fill(1.0)

    remaining = float(target_total) - float(np.sum(result))
    active = np.ones(result.size, dtype=bool)
    for _ in range(result.size + 1):
        if abs(remaining) <= tolerance:
            break
        if remaining > 0.0:
            headroom = upper_values - result
        else:
            headroom = result - lower_values
        usable = active & (headroom > tolerance)
        if not np.any(usable):
            break

        finite_headroom = usable & np.isfinite(headroom)
        weights = np.zeros(result.size, dtype=np.float64)
        weights[finite_headroom] = headroom[finite_headroom] * alpha_values[finite_headroom]
        unbounded = usable & ~finite_headroom
        weights[unbounded] = alpha_values[unbounded]
        if not np.any(weights > 0.0):
            weights[usable] = 1.0
        proposals = abs(remaining) * weights / float(np.sum(weights))
        saturated = finite_headroom & (proposals >= headroom - tolerance)
        applied = proposals.copy()
        applied[saturated] = headroom[saturated]
        applied[~usable] = 0.0
        if remaining > 0.0:
            result += applied
        else:
            result -= applied
        remaining -= float(np.sum(applied)) if remaining > 0.0 else -float(np.sum(applied))
        active[saturated] = False
        if not np.any(saturated) and np.all(applied <= tolerance):
            break

    remaining = float(target_total) - float(np.sum(result))
    if abs(remaining) > tolerance:
        overflow_weights = alpha_values.copy()
        if not np.any(overflow_weights > 0.0):
            overflow_weights.fill(1.0)
        result += remaining * overflow_weights / float(np.sum(overflow_weights))

        # Remove the final floating-point accumulation error without changing
        # the intended participation set.
        correction = float(target_total) - float(np.sum(result))
        if abs(correction) > tolerance:
            participating = np.flatnonzero(overflow_weights > 0.0)
            result[int(participating[-1])] += correction
    return result
