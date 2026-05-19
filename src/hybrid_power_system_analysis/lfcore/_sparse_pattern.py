import numpy as np


def build_compressed_pattern_from_raw_coords(raw_major, raw_minor, n_major: int):
    """Build a compressed sparse pattern and raw-entry-to-slot map.

    ``raw_major`` is row for CSR and column for CSC. Duplicate coordinates are
    collapsed in the returned pattern; ``raw_to_pos`` maps each raw entry to the
    collapsed slot so runtime values can be reduced without rebuilding sparsity.
    """
    raw_major = np.asarray(raw_major, dtype=np.int32)
    raw_minor = np.asarray(raw_minor, dtype=np.int32)
    if raw_major.size != raw_minor.size:
        raise ValueError("raw_major and raw_minor must have the same length")
    raw_count = int(raw_major.size)
    if raw_count == 0:
        return (
            np.array([], dtype=np.int32),
            np.zeros(int(n_major) + 1, dtype=np.int32),
            np.array([], dtype=np.intp),
        )

    order = np.lexsort((raw_minor, raw_major))
    sorted_major = raw_major[order]
    sorted_minor = raw_minor[order]

    is_new = np.empty(raw_count, dtype=bool)
    is_new[0] = True
    is_new[1:] = (sorted_major[1:] != sorted_major[:-1]) | (sorted_minor[1:] != sorted_minor[:-1])
    sorted_group = np.cumsum(is_new, dtype=np.intp) - 1
    raw_to_pos = np.empty(raw_count, dtype=np.intp)
    raw_to_pos[order] = sorted_group

    indices = sorted_minor[is_new].astype(np.int32, copy=True)
    major_counts = np.bincount(sorted_major[is_new], minlength=int(n_major))
    indptr = np.empty(int(n_major) + 1, dtype=np.int32)
    indptr[0] = 0
    indptr[1:] = np.cumsum(major_counts, dtype=np.int64)
    return indices, indptr, raw_to_pos


def build_raw_sum_plan(raw_to_pos, n_pos: int):
    """Precompute direct-copy and duplicate-reduction slots for raw values."""
    raw_to_pos = np.asarray(raw_to_pos, dtype=np.intp)
    empty = np.array([], dtype=np.intp)
    if raw_to_pos.size == 0 or int(n_pos) == 0:
        return empty, empty, empty, empty, empty

    counts = np.bincount(raw_to_pos, minlength=int(n_pos))
    source_by_pos = np.empty(int(n_pos), dtype=np.intp)
    source_by_pos[raw_to_pos] = np.arange(raw_to_pos.size, dtype=np.intp)

    direct_pos = np.flatnonzero(counts == 1).astype(np.intp, copy=False)
    direct_raw = source_by_pos[direct_pos].astype(np.intp, copy=False)

    duplicate_raw = np.flatnonzero(counts[raw_to_pos] > 1).astype(np.intp, copy=False)
    if duplicate_raw.size == 0:
        return direct_pos, direct_raw, empty, empty, empty

    order = np.argsort(raw_to_pos[duplicate_raw], kind="stable")
    duplicate_raw = duplicate_raw[order].astype(np.intp, copy=False)
    duplicate_pos_sorted = raw_to_pos[duplicate_raw]
    duplicate_starts = np.empty(np.count_nonzero(np.r_[True, duplicate_pos_sorted[1:] != duplicate_pos_sorted[:-1]]), dtype=np.intp)
    duplicate_starts[0] = 0
    if duplicate_starts.size > 1:
        duplicate_starts[1:] = np.flatnonzero(duplicate_pos_sorted[1:] != duplicate_pos_sorted[:-1]) + 1
    duplicate_pos = duplicate_pos_sorted[duplicate_starts].astype(np.intp, copy=False)
    return direct_pos, direct_raw, duplicate_pos, duplicate_raw, duplicate_starts


def apply_raw_sum_plan(out, raw, plan):
    """Refresh compressed data from raw values using a precomputed sum plan."""
    direct_pos, direct_raw, duplicate_pos, duplicate_raw, duplicate_starts = plan
    if direct_pos.size:
        out[direct_pos] = raw[direct_raw]
    if duplicate_pos.size:
        out[duplicate_pos] = np.add.reduceat(raw[duplicate_raw], duplicate_starts)
    return out
