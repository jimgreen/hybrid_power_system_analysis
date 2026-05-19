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
