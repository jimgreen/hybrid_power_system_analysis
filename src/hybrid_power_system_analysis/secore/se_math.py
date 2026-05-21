import hashlib
import math
import warnings
from typing import List, Optional, Sequence, Tuple

import numpy as np

from scipy.linalg import cho_factor as CHO_FACTOR
from scipy.linalg import cho_solve as CHO_SOLVE
from scipy.linalg.lapack import dposv as DPOSV
from scipy.linalg.lapack import dpotrf as DPOTRF
from scipy.sparse import coo_matrix as SP_COO_MATRIX
from scipy.sparse import csc_matrix as SP_CSC_MATRIX
from scipy.sparse import csr_matrix as SP_CSR_MATRIX
from scipy.sparse import issparse as SP_ISSPARSE
from scipy.sparse import tril as SP_TRIL
from scipy.sparse.csgraph import connected_components as SP_CONNECTED_COMPONENTS
from scipy.sparse.csgraph import structural_rank as SP_STRUCTURAL_RANK
from scipy.sparse.linalg import MatrixRankWarning as SP_MATRIX_RANK_WARNING
from scipy.sparse.linalg import eigsh as SP_EIGSH
from scipy.sparse.linalg import splu as SP_SPLU
from scipy.sparse.linalg import spsolve as SP_SPSOLVE

try:
    from sksparse.cholmod import cholesky as CHOLMOD_CHOLESKY
    from sksparse.cholmod import analyze as CHOLMOD_ANALYZE
except Exception:
    CHOLMOD_CHOLESKY = None
    CHOLMOD_ANALYZE = None

try:
    from sksparse.cholmod import cholesky_AAt as CHOLMOD_CHOLESKY_AAT
    from sksparse.cholmod import analyze_AAt as CHOLMOD_ANALYZE_AAT
except Exception:
    CHOLMOD_CHOLESKY_AAT = None
    CHOLMOD_ANALYZE_AAT = None


ANGLE_MEASUREMENT_TYPES = frozenset(("ANGLE", "THETA", "ANGLE_DIFF", "THETA_DIFF"))
_NORMAL_EQUATION_PATTERN_CACHE = {}
_SPARSE_PATTERN_EXPANSION_CACHE = {}
_SPARSE_PATTERN_LINEAR_INDEX_CACHE = {}
_SPARSE_CSR_ROW_INDEX_CACHE = {}
_MAX_DIRECT_NORMAL_ASSEMBLY_PAIRS = 2_000_000

def targeted_redundancy_count(state_count: int, ratio: float) -> int:
    """Return the configured pseudo-measurement redundancy target as a row count."""
    return max(0, int(math.ceil(max(0.0, float(ratio)) * max(0, int(state_count)))))


def observability_weak_direction(
    H,
    state_count: int,
    weak_states: Sequence[Tuple[int, float]] = (),
    dense_svd_limit: int = 2000,
) -> np.ndarray:
    """Return a normalized state-space direction that is currently weakest."""
    state_count = int(state_count)
    if state_count <= 0:
        return np.array([], dtype=np.float64)

    direction = np.zeros(state_count, dtype=np.float64)
    if weak_states:
        for state_idx, score in weak_states:
            try:
                pos = int(state_idx)
            except (TypeError, ValueError):
                continue
            if 0 <= pos < state_count:
                direction[pos] = max(abs(float(score)), direction[pos])
        norm = float(np.linalg.norm(direction))
        if norm > 0.0:
            return direction / norm

    if H is None or H.shape[0] == 0:
        direction[0] = 1.0
        return direction

    try:
        if state_count <= dense_svd_limit:
            H_arr = H.toarray() if is_sparse_matrix(H) else np.asarray(H, dtype=np.float64)
            if H_arr.size:
                _u, _s, vh = np.linalg.svd(H_arr, full_matrices=False)
                if vh.size:
                    direction = np.asarray(vh[-1, :], dtype=np.float64)
                    norm = float(np.linalg.norm(direction))
                    if norm > 0.0:
                        return direction / norm
        gram = H.T @ H
        if is_sparse_matrix(gram):
            values, vectors = SP_EIGSH(gram.tocsc(), k=1, which="SM", tol=1e-6, maxiter=500)
            if values.size and vectors.size:
                direction = np.asarray(vectors[:, 0], dtype=np.float64)
                norm = float(np.linalg.norm(direction))
                if norm > 0.0:
                    return direction / norm
        else:
            values, vectors = np.linalg.eigh(np.asarray(gram, dtype=np.float64))
            if values.size and vectors.size:
                direction = np.asarray(vectors[:, int(np.argmin(values))], dtype=np.float64)
                norm = float(np.linalg.norm(direction))
                if norm > 0.0:
                    return direction / norm
    except Exception:
        pass

    gram = H.T @ H
    diag = np.asarray(gram.diagonal() if is_sparse_matrix(gram) else np.diag(gram), dtype=np.float64)
    if diag.size:
        direction[int(np.argmin(diag))] = 1.0
    else:
        direction[0] = 1.0
    return direction


def _as_array_dtype(values, dtype):
    if isinstance(values, np.ndarray) and values.dtype == dtype:
        return values
    return np.asarray(values, dtype=dtype)


def angle_residual_mask(measurements: Sequence[object]) -> np.ndarray:
    """Mark measurement rows whose residual lives on the circular angle domain."""
    return np.fromiter(
        (getattr(meas, "meas_type", None) in ANGLE_MEASUREMENT_TYPES for meas in measurements),
        dtype=bool,
        count=len(measurements),
    )


def wrap_angle_residual(residual: np.ndarray) -> np.ndarray:
    """Map angle residuals to [-pi, pi) so 2*pi-equivalent angles stay close."""
    return (residual + np.pi) % (2.0 * np.pi) - np.pi


def measurement_residual(
    z: np.ndarray,
    z_est: np.ndarray,
    angle_mask: np.ndarray = None,
    has_angle_residuals: Optional[bool] = None,
) -> np.ndarray:
    """Return z - h(x), wrapping only rows that represent angle measurements."""
    residual = np.asarray(z, dtype=np.float64) - np.asarray(z_est, dtype=np.float64)
    if has_angle_residuals is None:
        has_angle_residuals = bool(angle_mask is not None and np.any(angle_mask))
    if has_angle_residuals:
        residual = residual.copy()
        residual[angle_mask] = wrap_angle_residual(residual[angle_mask])
    return residual


class SparseJacobianBuilder:
    """Collect Jacobian entries as COO triplets while preserving ndarray-style writes.

    The state estimators already express every analytical derivative as sparse
    row/column writes. This adapter lets those writes create sparse triplets
    directly instead of first allocating a mostly-zero dense measurement matrix.
    """

    def __init__(self, shape: Tuple[int, int]):
        self.shape = tuple(int(item) for item in shape)
        self.size = self.shape[0] * self.shape[1]
        self._cached_pattern_linear: Optional[np.ndarray] = None
        self._cached_pattern_indptr: Optional[np.ndarray] = None
        self._cached_pattern_indices: Optional[np.ndarray] = None
        self._cached_slot_positions: Optional[np.ndarray] = None
        self._cached_unique_slot_mask: Optional[np.ndarray] = None
        self._cached_duplicate_slot_mask: Optional[np.ndarray] = None
        self._cached_unique_slots: Optional[np.ndarray] = None
        self._cached_unique_data_positions: Optional[np.ndarray] = None
        self._cached_duplicate_slots: Optional[np.ndarray] = None
        self._cached_duplicate_data_positions: Optional[np.ndarray] = None
        self._cached_chunk_slices: Optional[List[Tuple[int, int]]] = None
        self._cached_chunk_slot_plans: Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = None
        self._cached_direct_chunk_slots: Optional[List[Optional[np.ndarray]]] = None
        self._cached_duplicate_dest_slots: Optional[np.ndarray] = None
        self._cached_slots_cover_pattern = False
        self._cached_has_unique_slots = False
        self._cached_has_duplicate_slots = False
        self._cached_csr_data: Optional[np.ndarray] = None
        self._cached_csr_matrix = None
        self._data_buffer: Optional[np.ndarray] = None
        self._assume_fixed_pattern = False
        self._data_only_refresh_enabled = False
        self._data_only_refresh_active = False
        self._data_only_direct_active = False
        self._data_only_direct_initialized = False
        self._data_only_chunk_cursor = 0
        self.reset()

    def _set_cached_slot_positions(self, slot: np.ndarray) -> None:
        self._cached_slot_positions = slot
        self._cached_chunk_slices = self._current_chunk_slices()
        self._cached_chunk_slot_plans = []
        self._cached_direct_chunk_slots = []
        if slot.size == 0:
            self._cached_unique_slot_mask = np.array([], dtype=bool)
            self._cached_duplicate_slot_mask = np.array([], dtype=bool)
            self._cached_unique_slots = np.array([], dtype=np.int64)
            self._cached_unique_data_positions = np.array([], dtype=np.int64)
            self._cached_duplicate_slots = np.array([], dtype=np.int64)
            self._cached_duplicate_data_positions = np.array([], dtype=np.int64)
            self._cached_duplicate_dest_slots = np.array([], dtype=np.int64)
            self._cached_slots_cover_pattern = self._cached_pattern_linear is not None and self._cached_pattern_linear.size == 0
            self._cached_has_unique_slots = False
            self._cached_has_duplicate_slots = False
            return
        counts = np.bincount(slot, minlength=int(slot.max()) + 1)
        self._cached_unique_slot_mask = counts[slot] == 1
        self._cached_duplicate_slot_mask = ~self._cached_unique_slot_mask
        self._cached_has_unique_slots = bool(np.any(self._cached_unique_slot_mask))
        self._cached_has_duplicate_slots = bool(np.any(self._cached_duplicate_slot_mask))
        if self._cached_has_unique_slots:
            self._cached_unique_data_positions = np.nonzero(self._cached_unique_slot_mask)[0].astype(np.int64, copy=False)
            self._cached_unique_slots = slot[self._cached_unique_data_positions]
        else:
            self._cached_unique_data_positions = np.array([], dtype=np.int64)
            self._cached_unique_slots = np.array([], dtype=np.int64)
        if self._cached_has_duplicate_slots:
            self._cached_duplicate_data_positions = np.nonzero(self._cached_duplicate_slot_mask)[0].astype(
                np.int64,
                copy=False,
            )
            self._cached_duplicate_slots = slot[self._cached_duplicate_data_positions]
        else:
            self._cached_duplicate_data_positions = np.array([], dtype=np.int64)
            self._cached_duplicate_slots = np.array([], dtype=np.int64)
        self._cached_duplicate_dest_slots = np.unique(self._cached_duplicate_slots).astype(np.int64, copy=False)
        self._cached_slots_cover_pattern = (
            self._cached_pattern_linear is not None
            and np.count_nonzero(counts) == int(self._cached_pattern_linear.size)
        )
        for start, end in self._cached_chunk_slices:
            if start == end:
                empty = np.array([], dtype=np.int64)
                self._cached_chunk_slot_plans.append((empty, empty, empty, empty))
                self._cached_direct_chunk_slots.append(empty)
                continue
            chunk_unique_mask = self._cached_unique_slot_mask[start:end]
            unique_pos = np.nonzero(chunk_unique_mask)[0].astype(np.int64, copy=False)
            duplicate_pos = np.nonzero(~chunk_unique_mask)[0].astype(np.int64, copy=False)
            chunk_slots = slot[start:end]
            unique_slots = chunk_slots[unique_pos].astype(np.int64, copy=False)
            duplicate_slots = chunk_slots[duplicate_pos].astype(np.int64, copy=False)
            self._cached_chunk_slot_plans.append((unique_pos, unique_slots, duplicate_pos, duplicate_slots))
            if duplicate_pos.size == 0 and unique_pos.size == chunk_slots.size:
                self._cached_direct_chunk_slots.append(chunk_slots.astype(np.int64, copy=False))
            else:
                self._cached_direct_chunk_slots.append(None)

    def _current_chunk_slices(self) -> List[Tuple[int, int]]:
        slices: List[Tuple[int, int]] = []
        cursor = len(self.data)
        for chunk in self._data_chunks:
            size = int(chunk.size)
            slices.append((cursor, cursor + size))
            cursor += size
        return slices

    def reset(self) -> None:
        self.rows = []
        self.cols = []
        self.data = []
        self._row_chunks = []
        self._col_chunks = []
        self._data_chunks = []
        self._data_only_chunk_cursor = 0
        self._data_only_refresh_active = bool(
            self._data_only_refresh_enabled
            and self._assume_fixed_pattern
            and self._cached_pattern_indptr is not None
            and self._cached_pattern_indices is not None
            and self._cached_chunk_slices is not None
        )
        first_chunk_starts_after_scalars = (
            self._cached_chunk_slices is not None
            and len(self._cached_chunk_slices) > 0
            and int(self._cached_chunk_slices[0][0]) == 0
        )
        self._data_only_direct_active = bool(self._data_only_refresh_active and first_chunk_starts_after_scalars)
        self._data_only_direct_initialized = False

    def _initialize_data_only_direct_buffer(self) -> Optional[np.ndarray]:
        if not self._data_only_direct_active:
            return None
        pattern = self._cached_pattern_linear
        if pattern is None:
            return None
        if self._cached_csr_data is None or self._cached_csr_data.size != pattern.size:
            self._cached_csr_data = np.zeros(pattern.size, dtype=np.float64)
            self._cached_csr_matrix = None
        values = self._cached_csr_data
        if not self._data_only_direct_initialized:
            if not self._cached_slots_cover_pattern:
                values.fill(0.0)
            else:
                duplicate_dest_slots = self._cached_duplicate_dest_slots
                if duplicate_dest_slots is not None and duplicate_dest_slots.size:
                    values[duplicate_dest_slots] = 0.0
            self._data_only_direct_initialized = True
        return values

    def _append_data_only_chunk(self, values: np.ndarray) -> bool:
        if not self._data_only_refresh_active:
            return False
        chunk_slices = self._cached_chunk_slices
        if chunk_slices is None or self._data_only_chunk_cursor >= len(chunk_slices):
            raise RuntimeError("SparseJacobianBuilder fixed CSR data refresh received too many chunks")
        start, end = chunk_slices[self._data_only_chunk_cursor]
        expected = int(end) - int(start)
        if int(values.size) != expected:
            raise RuntimeError(
                "SparseJacobianBuilder fixed CSR data refresh pattern changed: "
                f"expected chunk size {expected}, got {int(values.size)}"
            )
        values = _as_array_dtype(values, np.float64)
        target = self._initialize_data_only_direct_buffer()
        if target is not None:
            direct_slots = (
                self._cached_direct_chunk_slots[self._data_only_chunk_cursor]
                if self._cached_direct_chunk_slots is not None
                and self._data_only_chunk_cursor < len(self._cached_direct_chunk_slots)
                else None
            )
            chunk_plan = (
                self._cached_chunk_slot_plans[self._data_only_chunk_cursor]
                if self._cached_chunk_slot_plans is not None
                and self._data_only_chunk_cursor < len(self._cached_chunk_slot_plans)
                else None
            )
            if direct_slots is not None:
                if direct_slots.size:
                    target[direct_slots] = values
            elif chunk_plan is not None:
                unique_pos, unique_slots, duplicate_pos, duplicate_slots = chunk_plan
                if unique_pos.size:
                    target[unique_slots] = values[unique_pos]
                if duplicate_pos.size:
                    np.add.at(target, duplicate_slots, values[duplicate_pos])
            else:
                slot = self._cached_slot_positions[int(start) : int(end)]
                np.add.at(target, slot, values)
        else:
            self._data_chunks.append(values)
        self._data_only_chunk_cursor += 1
        return True

    def _append_arrays(self, rows: np.ndarray, cols: np.ndarray, values: np.ndarray) -> None:
        """Keep vectorized writes as NumPy chunks until final COO/CSR construction."""
        if rows.size == 0:
            return
        rows = _as_array_dtype(rows, np.int32)
        cols = _as_array_dtype(cols, np.int32)
        values = _as_array_dtype(values, np.float64)
        mask = cols >= 0
        if not mask.any():
            return
        if mask.all():
            if self._append_data_only_chunk(values):
                return
            self._row_chunks.append(rows)
            self._col_chunks.append(cols)
            self._data_chunks.append(values)
            return
        masked_values = values[mask].astype(np.float64, copy=False)
        if self._append_data_only_chunk(masked_values):
            return
        self._row_chunks.append(rows[mask].astype(np.int32, copy=False))
        self._col_chunks.append(cols[mask].astype(np.int32, copy=False))
        self._data_chunks.append(masked_values)

    def _coo_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.rows and not self._row_chunks:
            return (
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
                np.array([], dtype=np.float64),
            )
        scalar_count = len(self.rows)
        chunk_sizes = [int(chunk.size) for chunk in self._row_chunks]
        total_count = scalar_count + sum(chunk_sizes)
        if total_count == 0:
            return (
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
                np.array([], dtype=np.float64),
            )
        out_rows = np.empty(total_count, dtype=np.int32)
        out_cols = np.empty(total_count, dtype=np.int32)
        out_data = np.empty(total_count, dtype=np.float64)
        cursor = 0
        if scalar_count:
            end = scalar_count
            out_rows[:end] = np.asarray(self.rows, dtype=np.int32)
            out_cols[:end] = np.asarray(self.cols, dtype=np.int32)
            out_data[:end] = np.asarray(self.data, dtype=np.float64)
            cursor = end
        for rows, cols, data, size in zip(self._row_chunks, self._col_chunks, self._data_chunks, chunk_sizes):
            if size == 0:
                continue
            end = cursor + size
            out_rows[cursor:end] = rows
            out_cols[cursor:end] = cols
            out_data[cursor:end] = data
            cursor = end
        if cursor != total_count:
            return out_rows[:cursor], out_cols[:cursor], out_data[:cursor]
        return out_rows, out_cols, out_data

    def _data_array(self) -> np.ndarray:
        if not self.data and not self._data_chunks:
            return np.array([], dtype=np.float64)
        scalar_count = len(self.data)
        chunk_sizes = [int(chunk.size) for chunk in self._data_chunks]
        total_count = scalar_count + sum(chunk_sizes)
        if self._data_buffer is None or self._data_buffer.size < total_count:
            self._data_buffer = np.empty(total_count, dtype=np.float64)
        out_data = self._data_buffer
        cursor = 0
        if scalar_count:
            out_data[:scalar_count] = np.asarray(self.data, dtype=np.float64)
            cursor = scalar_count
        for data, size in zip(self._data_chunks, chunk_sizes):
            if size == 0:
                continue
            end = cursor + size
            out_data[cursor:end] = data
            cursor = end
        if cursor != total_count:
            return out_data[:cursor]
        return out_data[:total_count]

    def _refresh_fixed_pattern_values(self, values: np.ndarray) -> None:
        scalar_count = len(self.data)
        if scalar_count or not self._cached_slots_cover_pattern:
            values.fill(0.0)
        else:
            duplicate_dest_slots = self._cached_duplicate_dest_slots
            if duplicate_dest_slots is not None and duplicate_dest_slots.size:
                values[duplicate_dest_slots] = 0.0
        if scalar_count:
            scalar_values = np.asarray(self.data, dtype=np.float64)
            np.add.at(values, self._cached_slot_positions[:scalar_count], scalar_values)
        chunk_slices = self._cached_chunk_slices
        chunk_slot_plans = self._cached_chunk_slot_plans
        if chunk_slices is None or len(chunk_slices) != len(self._data_chunks):
            data = self._data_array()
            unique_mask = self._cached_unique_slot_mask
            if unique_mask is None or unique_mask.size != self._cached_slot_positions.size:
                np.add.at(values, self._cached_slot_positions, data)
                return
            if self._cached_has_unique_slots:
                values[self._cached_unique_slots] = data[self._cached_unique_data_positions]
            if self._cached_has_duplicate_slots:
                np.add.at(values, self._cached_duplicate_slots, data[self._cached_duplicate_data_positions])
            return
        if chunk_slot_plans is not None and len(chunk_slot_plans) == len(self._data_chunks):
            direct_chunk_slots = self._cached_direct_chunk_slots
            if direct_chunk_slots is not None and len(direct_chunk_slots) == len(self._data_chunks):
                for chunk, direct_slots, (unique_pos, unique_slots, duplicate_pos, duplicate_slots) in zip(
                    self._data_chunks,
                    direct_chunk_slots,
                    chunk_slot_plans,
                ):
                    if direct_slots is not None:
                        if direct_slots.size:
                            values[direct_slots] = chunk
                        continue
                    if unique_pos.size:
                        values[unique_slots] = chunk[unique_pos]
                    if duplicate_pos.size:
                        np.add.at(values, duplicate_slots, chunk[duplicate_pos])
                return
            for chunk, (unique_pos, unique_slots, duplicate_pos, duplicate_slots) in zip(
                self._data_chunks,
                chunk_slot_plans,
            ):
                if unique_pos.size:
                    values[unique_slots] = chunk[unique_pos]
                if duplicate_pos.size:
                    np.add.at(values, duplicate_slots, chunk[duplicate_pos])
            return
        for chunk, (start, end) in zip(self._data_chunks, chunk_slices):
            if start == end:
                continue
            np.add.at(values, self._cached_slot_positions[start:end], chunk)

    def __getitem__(self, key):
        row_key, col_key = key
        if isinstance(col_key, slice):
            if col_key != slice(None):
                raise NotImplementedError("SparseJacobianBuilder only supports full-column slices")
            return np.zeros(self.shape[1], dtype=np.float64)

        rows = np.asarray(row_key)
        cols = np.asarray(col_key)
        if rows.ndim == 0 and cols.ndim == 0:
            return 0.0
        broadcast_rows, _broadcast_cols = np.broadcast_arrays(rows, cols)
        return np.zeros(broadcast_rows.shape, dtype=np.float64)

    def __setitem__(self, key, value) -> None:
        row_key, col_key = key
        if isinstance(col_key, slice):
            if col_key != slice(None):
                raise NotImplementedError("SparseJacobianBuilder only supports full-column slices")
            values = np.asarray(value, dtype=np.float64)
            if values.ndim != 1 or values.size != self.shape[1]:
                raise ValueError("Full-row sparse assignment requires a vector with n_state entries")
            cols = np.nonzero(values)[0].astype(np.int32)
            if cols.size:
                self._append_arrays(
                    np.full(cols.size, int(row_key), dtype=np.int32),
                    cols,
                    values[cols].astype(np.float64, copy=False),
                )
            return

        if np.isscalar(row_key) and np.isscalar(col_key) and np.isscalar(value):
            col = int(col_key)
            if col >= 0:
                self.rows.append(int(row_key))
                self.cols.append(col)
                self.data.append(float(value))
            return

        rows = np.asarray(row_key)
        cols = np.asarray(col_key)
        values = np.asarray(value, dtype=np.float64)
        broadcast_rows, broadcast_cols, broadcast_values = np.broadcast_arrays(rows, cols, values)
        flat_rows = broadcast_rows.ravel().astype(np.int32)
        flat_cols = broadcast_cols.ravel().astype(np.int32)
        flat_values = broadcast_values.ravel().astype(np.float64)
        mask = flat_cols >= 0
        if np.any(mask):
            self._append_arrays(flat_rows[mask], flat_cols[mask], flat_values[mask])

    def add(self, row: int, col: int, value: float) -> None:
        if col >= 0:
            self.rows.append(int(row))
            self.cols.append(int(col))
            self.data.append(float(value))

    def add_many(self, rows: np.ndarray, cols: np.ndarray, values: np.ndarray, mask: np.ndarray = None) -> None:
        rows = _as_array_dtype(rows, np.int32)
        cols = _as_array_dtype(cols, np.int32)
        values = _as_array_dtype(values, np.float64)
        if mask is None:
            mask = cols >= 0
        else:
            mask = np.asarray(mask, dtype=bool) & (cols >= 0)
        if mask.any():
            if mask.all():
                if self._append_data_only_chunk(values):
                    return
                self._row_chunks.append(rows)
                self._col_chunks.append(cols)
                self._data_chunks.append(values)
                return
            masked_values = values[mask].astype(np.float64, copy=False)
            if self._append_data_only_chunk(masked_values):
                return
            self._row_chunks.append(rows[mask].astype(np.int32, copy=False))
            self._col_chunks.append(cols[mask].astype(np.int32, copy=False))
            self._data_chunks.append(masked_values)

    def to_csr(self):
        if self._assume_fixed_pattern and self._cached_slot_positions is not None and self._cached_pattern_indptr is not None and self._cached_pattern_indices is not None:
            if self._data_only_refresh_active and self._cached_chunk_slices is not None:
                expected_chunks = len(self._cached_chunk_slices)
                if self._data_only_chunk_cursor != expected_chunks:
                    raise RuntimeError(
                        "SparseJacobianBuilder fixed CSR data refresh pattern changed: "
                        f"expected {expected_chunks} chunks, got {self._data_only_chunk_cursor}"
                    )
                if self._data_only_direct_initialized and self._cached_csr_data is not None:
                    if self.rows:
                        n_cols = int(self.shape[1])
                        rows = np.asarray(self.rows, dtype=np.int64)
                        cols = np.asarray(self.cols, dtype=np.int64)
                        linear = rows * np.int64(n_cols) + cols
                        slot = np.searchsorted(self._cached_pattern_linear, linear)
                        if (
                            slot.size
                            and int(slot.max()) < self._cached_pattern_linear.size
                            and np.array_equal(self._cached_pattern_linear[slot], linear)
                        ):
                            np.add.at(self._cached_csr_data, slot, np.asarray(self.data, dtype=np.float64))
                        elif slot.size:
                            raise RuntimeError("SparseJacobianBuilder fixed CSR data refresh scalar pattern changed")
                    if self._cached_csr_matrix is None:
                        self._cached_csr_matrix = SP_CSR_MATRIX(
                            (self._cached_csr_data, self._cached_pattern_indices, self._cached_pattern_indptr),
                            shape=self.shape,
                            copy=False,
                        )
                    return self._cached_csr_matrix
            if self._cached_csr_data is None or self._cached_csr_data.size != self._cached_pattern_linear.size:
                self._cached_csr_data = np.zeros(self._cached_pattern_linear.size, dtype=np.float64)
                self._cached_csr_matrix = None
            values = self._cached_csr_data
            self._refresh_fixed_pattern_values(values)
            if self._cached_csr_matrix is None:
                self._cached_csr_matrix = SP_CSR_MATRIX(
                    (values, self._cached_pattern_indices, self._cached_pattern_indptr),
                    shape=self.shape,
                    copy=False,
                )
            return self._cached_csr_matrix
        rows, cols, data = self._coo_arrays()
        if self._cached_pattern_linear is not None and self._cached_pattern_indptr is not None and self._cached_pattern_indices is not None:
            n_cols = int(self.shape[1])
            linear = rows.astype(np.int64, copy=False) * n_cols + cols.astype(np.int64, copy=False)
            slot = np.searchsorted(self._cached_pattern_linear, linear)
            if slot.size and int(slot.max()) < self._cached_pattern_linear.size and np.array_equal(
                self._cached_pattern_linear[slot],
                linear,
            ):
                self._set_cached_slot_positions(slot)
                values = np.zeros(self._cached_pattern_linear.size, dtype=np.float64)
                np.add.at(values, slot, data)
                return SP_CSR_MATRIX(
                    (values, self._cached_pattern_indices, self._cached_pattern_indptr),
                    shape=self.shape,
                    copy=False,
                )
        csr = SP_COO_MATRIX((data, (rows, cols)), shape=self.shape).tocsr(copy=False)
        indptr = csr.indptr.astype(np.int32, copy=True)
        indices = csr.indices.astype(np.int32, copy=True)
        pattern_rows = np.repeat(np.arange(self.shape[0], dtype=np.int64), np.diff(indptr).astype(np.int64, copy=False))
        pattern_linear = pattern_rows * np.int64(self.shape[1]) + indices.astype(np.int64, copy=False)
        self._cached_pattern_linear = pattern_linear
        self._cached_pattern_indptr = indptr
        self._cached_pattern_indices = indices
        self._cached_csr_matrix = None
        if self._assume_fixed_pattern:
            n_cols = int(self.shape[1])
            linear = rows.astype(np.int64, copy=False) * n_cols + cols.astype(np.int64, copy=False)
            slot = np.searchsorted(pattern_linear, linear)
            if slot.size and int(slot.max()) < pattern_linear.size and np.array_equal(pattern_linear[slot], linear):
                self._set_cached_slot_positions(slot)
                self._cached_csr_data = csr.data
                self._cached_csr_matrix = csr
                self._data_only_refresh_enabled = True
        return csr


def is_sparse_matrix(matrix) -> bool:
    return bool(SP_ISSPARSE(matrix))


def sparse_structural_rank(matrix) -> Optional[int]:
    """Return sparse structural rank for sparse matrices."""
    if not is_sparse_matrix(matrix):
        return None
    return int(SP_STRUCTURAL_RANK(matrix))


def matrix_is_empty(matrix) -> bool:
    return matrix.shape[0] == 0 or matrix.shape[1] == 0


def full_normal_equation_from_lower(matrix):
    """Expand a lower-triangular normal-equation matrix into full symmetry."""
    if is_sparse_matrix(matrix):
        lower = matrix if getattr(matrix, "format", None) == "csc" else matrix.tocsc()
        full = lower + lower.T
        full.setdiag(lower.diagonal())
        return full.tocsc()
    lower = np.tril(np.asarray(matrix, dtype=np.float64))
    return lower + lower.T - np.diag(np.diag(lower))


def _normal_equation_structural_pattern(H, triangular: Optional[str] = None):
    """Return the structural pattern implied by H.T @ H, retaining zero entries."""
    if not is_sparse_matrix(H):
        return None
    triangular_mode = None if triangular is None else str(triangular).lower()
    if triangular_mode not in (None, "lower"):
        raise ValueError(f"Unsupported normal-equation triangular mode: {triangular!r}")
    H_csc = H if getattr(H, "format", None) == "csc" else H.tocsc()
    digest = hashlib.blake2b(
        H_csc.indptr.tobytes() + H_csc.indices.tobytes(),
        digest_size=16,
    ).digest()
    key = (H_csc.shape, int(H_csc.nnz), digest, triangular_mode)
    cached = _NORMAL_EQUATION_PATTERN_CACHE.get(key)
    if cached is not None:
        return cached
    H_pattern = H_csc.copy()
    H_pattern.data = np.ones(int(H_pattern.nnz), dtype=np.float64)
    pattern = (H_pattern.T @ H_pattern).tocsc()
    if triangular_mode == "lower":
        pattern = SP_TRIL(pattern, format="csc")
    pattern.data = np.zeros(int(pattern.nnz), dtype=np.float64)
    if len(_NORMAL_EQUATION_PATTERN_CACHE) > 16:
        _NORMAL_EQUATION_PATTERN_CACHE.clear()
    _NORMAL_EQUATION_PATTERN_CACHE[key] = pattern
    return pattern


def _expand_sparse_matrix_to_pattern(matrix, pattern):
    """Return matrix on the union pattern, keeping explicit structural zeros."""
    if pattern is None:
        return matrix
    matrix_csc = matrix if getattr(matrix, "format", None) == "csc" else matrix.tocsc()
    pattern_csc = pattern if getattr(pattern, "format", None) == "csc" else pattern.tocsc()
    if not matrix_csc.has_sorted_indices:
        matrix_csc.sort_indices()
    if not pattern_csc.has_sorted_indices:
        pattern_csc.sort_indices()
    if (
        matrix_csc.shape == pattern_csc.shape
        and int(matrix_csc.nnz) == int(pattern_csc.nnz)
        and np.array_equal(matrix_csc.indptr, pattern_csc.indptr)
        and np.array_equal(matrix_csc.indices, pattern_csc.indices)
    ):
        return matrix_csc

    pattern_key = id(pattern_csc)
    cache = _SPARSE_PATTERN_EXPANSION_CACHE.get(pattern_key)
    if (
        cache is not None
        and cache["shape"] == matrix_csc.shape
        and cache["nnz"] == int(matrix_csc.nnz)
        and np.array_equal(cache["indptr"], matrix_csc.indptr)
        and np.array_equal(cache["indices"], matrix_csc.indices)
    ):
        target_positions = cache["target_positions"]
    else:
        n_rows = int(pattern_csc.shape[0])
        pattern_cache_key = (id(pattern_csc), pattern_csc.shape, int(pattern_csc.nnz))
        pattern_linear = _SPARSE_PATTERN_LINEAR_INDEX_CACHE.get(pattern_cache_key)
        if pattern_linear is None:
            pattern_counts = np.diff(pattern_csc.indptr).astype(np.int64, copy=False)
            pattern_cols = np.repeat(np.arange(int(pattern_csc.shape[1]), dtype=np.int64), pattern_counts)
            pattern_linear = pattern_cols * n_rows + pattern_csc.indices.astype(np.int64, copy=False)
            if len(_SPARSE_PATTERN_LINEAR_INDEX_CACHE) > 16:
                _SPARSE_PATTERN_LINEAR_INDEX_CACHE.clear()
            _SPARSE_PATTERN_LINEAR_INDEX_CACHE[pattern_cache_key] = pattern_linear
        matrix_counts = np.diff(matrix_csc.indptr).astype(np.int64, copy=False)
        matrix_cols = np.repeat(np.arange(int(matrix_csc.shape[1]), dtype=np.int64), matrix_counts)
        matrix_linear = matrix_cols * n_rows + matrix_csc.indices.astype(np.int64, copy=False)
        target_positions = np.nonzero(np.isin(pattern_linear, matrix_linear, assume_unique=True))[0].astype(
            np.int64,
            copy=False,
        )
        _SPARSE_PATTERN_EXPANSION_CACHE[pattern_key] = {
            "shape": matrix_csc.shape,
            "nnz": int(matrix_csc.nnz),
            "indptr": matrix_csc.indptr.copy(),
            "indices": matrix_csc.indices.copy(),
            "target_positions": target_positions,
        }
    data = np.zeros(int(pattern_csc.nnz), dtype=np.float64)
    data[target_positions] = matrix_csc.data
    return SP_CSC_MATRIX((data, pattern_csc.indices, pattern_csc.indptr), shape=pattern_csc.shape, copy=False)


def _csr_row_index_for_data(matrix_csr) -> np.ndarray:
    """Return the row index for every CSR data slot, cached by row pointer pattern."""
    indptr = matrix_csr.indptr
    digest = hashlib.blake2b(indptr.tobytes(), digest_size=16).digest()
    key = (matrix_csr.shape[0], int(matrix_csr.nnz), digest)
    cached = _SPARSE_CSR_ROW_INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    counts = np.diff(indptr).astype(np.intp, copy=False)
    row_index = np.repeat(np.arange(matrix_csr.shape[0], dtype=np.intp), counts)
    if len(_SPARSE_CSR_ROW_INDEX_CACHE) > 16:
        _SPARSE_CSR_ROW_INDEX_CACHE.clear()
    _SPARSE_CSR_ROW_INDEX_CACHE[key] = row_index
    return row_index


def _row_weighted_sparse_matrix(matrix, weight: np.ndarray):
    """Return W*H with the same CSR structure, avoiding scipy's broadcast multiply path."""
    matrix_csr = matrix if getattr(matrix, "format", None) == "csr" else matrix.tocsr()
    weight = np.asarray(weight, dtype=np.float64)
    row_index = _csr_row_index_for_data(matrix_csr)
    weighted_data = np.empty(int(matrix_csr.nnz), dtype=np.float64)
    np.multiply(matrix_csr.data, weight[row_index], out=weighted_data)
    return SP_CSR_MATRIX(
        (weighted_data, matrix_csr.indices, matrix_csr.indptr),
        shape=matrix_csr.shape,
        copy=False,
    )


class NormalEquationAssemblyPlan:
    """Precomputed sparse assembly plan for repeated ``H.T W H`` builds.

    State-estimation iterations keep the active measurement layout fixed, so the
    Jacobian sparsity pattern is normally stable.  This plan turns the row-wise
    Jacobian pattern into raw normal-equation coordinate slots once, then each
    iteration only refreshes numeric values with vectorized reductions.
    """

    def __init__(
        self,
        *,
        shape: Tuple[int, int],
        h_indptr: np.ndarray,
        h_indices: np.ndarray,
        h_rows: np.ndarray,
        h_cols: np.ndarray,
        pair_left_pos: np.ndarray,
        pair_right_pos: np.ndarray,
        pair_rows: np.ndarray,
        pair_slot: np.ndarray,
        gain_indptr: np.ndarray,
        gain_indices: np.ndarray,
        pair_count: int,
        use_direct_assembly: bool,
    ):
        self.shape = tuple(int(item) for item in shape)
        self.h_indptr = np.asarray(h_indptr, dtype=np.int32)
        self.h_indices = np.asarray(h_indices, dtype=np.int32)
        self.h_rows = np.asarray(h_rows, dtype=np.int32)
        self.h_cols = np.asarray(h_cols, dtype=np.int32)
        self.pair_left_pos = np.asarray(pair_left_pos, dtype=np.int32)
        self.pair_right_pos = np.asarray(pair_right_pos, dtype=np.int32)
        self.pair_rows = np.asarray(pair_rows, dtype=np.int32)
        self.pair_slot = np.asarray(pair_slot, dtype=np.int32)
        self.gain_indptr = np.asarray(gain_indptr, dtype=np.int32)
        self.gain_indices = np.asarray(gain_indices, dtype=np.int32)
        self.gain_shape = (self.shape[1], self.shape[1])
        self.gain_nnz = int(self.gain_indices.size)
        self.pair_count = int(pair_count)
        self.use_direct_assembly = bool(use_direct_assembly)

    @classmethod
    def _row_pair_count(cls, H) -> int:
        H_csr = H if getattr(H, "format", None) == "csr" else H.tocsr()
        counts = np.diff(H_csr.indptr).astype(np.int64, copy=False)
        return int(np.dot(counts, counts))

    @classmethod
    def direct_assembly_is_reasonable(cls, H, max_pair_count: int = _MAX_DIRECT_NORMAL_ASSEMBLY_PAIRS) -> bool:
        if not is_sparse_matrix(H):
            return False
        return cls._row_pair_count(H) <= int(max_pair_count)

    @classmethod
    def from_jacobian(
        cls,
        H,
        *,
        max_direct_pair_count: int = _MAX_DIRECT_NORMAL_ASSEMBLY_PAIRS,
    ) -> "NormalEquationAssemblyPlan":
        if not is_sparse_matrix(H):
            raise TypeError("NormalEquationAssemblyPlan requires a sparse Jacobian")
        H_csr = H if getattr(H, "format", None) == "csr" else H.tocsr()
        if not H_csr.has_sorted_indices:
            H_csr = H_csr.copy()
            H_csr.sort_indices()
        indptr = H_csr.indptr.astype(np.int32, copy=True)
        indices = H_csr.indices.astype(np.int32, copy=True)
        counts = np.diff(indptr).astype(np.int32, copy=False)
        pair_count = int(np.dot(counts.astype(np.int64, copy=False), counts.astype(np.int64, copy=False)))
        h_rows = np.repeat(np.arange(H_csr.shape[0], dtype=np.int32), counts)
        h_cols = indices.copy()

        left_chunks = []
        right_chunks = []
        row_chunks = []
        for count in np.unique(counts):
            k = int(count)
            if k <= 0:
                continue
            rows = np.nonzero(counts == k)[0].astype(np.int32, copy=False)
            if rows.size == 0:
                continue
            positions = indptr[rows, None] + np.arange(k, dtype=np.int32)
            left = np.repeat(positions, k, axis=1).ravel()
            right = np.tile(positions, (1, k)).ravel()
            left_chunks.append(left.astype(np.int32, copy=False))
            right_chunks.append(right.astype(np.int32, copy=False))
            row_chunks.append(np.repeat(rows, k * k).astype(np.int32, copy=False))

        if left_chunks:
            pair_left_pos = np.concatenate(left_chunks).astype(np.int32, copy=False)
            pair_right_pos = np.concatenate(right_chunks).astype(np.int32, copy=False)
            pair_rows = np.concatenate(row_chunks).astype(np.int32, copy=False)
            n_state = int(H_csr.shape[1])
            gain_rows = indices[pair_left_pos].astype(np.int64, copy=False)
            gain_cols = indices[pair_right_pos].astype(np.int64, copy=False)
            raw_linear = gain_cols * np.int64(n_state) + gain_rows
            gain_linear, pair_slot = np.unique(raw_linear, return_inverse=True)
            gain_indices = (gain_linear % np.int64(n_state)).astype(np.int32, copy=False)
            gain_cols_unique = (gain_linear // np.int64(n_state)).astype(np.int64, copy=False)
            gain_indptr = np.zeros(n_state + 1, dtype=np.int32)
            np.add.at(gain_indptr, gain_cols_unique + 1, 1)
            np.cumsum(gain_indptr, out=gain_indptr)
            pair_slot = pair_slot.astype(np.int32, copy=False)
        else:
            pair_left_pos = np.array([], dtype=np.int32)
            pair_right_pos = np.array([], dtype=np.int32)
            pair_rows = np.array([], dtype=np.int32)
            pair_slot = np.array([], dtype=np.int32)
            gain_indices = np.array([], dtype=np.int32)
            gain_indptr = np.zeros(H_csr.shape[1] + 1, dtype=np.int32)

        return cls(
            shape=H_csr.shape,
            h_indptr=indptr,
            h_indices=indices,
            h_rows=h_rows,
            h_cols=h_cols,
            pair_left_pos=pair_left_pos,
            pair_right_pos=pair_right_pos,
            pair_rows=pair_rows,
            pair_slot=pair_slot,
            gain_indptr=gain_indptr,
            gain_indices=gain_indices,
            pair_count=pair_count,
            use_direct_assembly=pair_count <= int(max_direct_pair_count),
        )

    def matches(self, H) -> bool:
        if not is_sparse_matrix(H) or tuple(H.shape) != self.shape:
            return False
        H_csr = H if getattr(H, "format", None) == "csr" else H.tocsr()
        if not H_csr.has_sorted_indices:
            return False
        return (
            int(H_csr.nnz) == int(self.h_indices.size)
            and H_csr.indptr.shape == self.h_indptr.shape
            and H_csr.indices.shape == self.h_indices.shape
            and np.array_equal(H_csr.indptr, self.h_indptr)
            and np.array_equal(H_csr.indices, self.h_indices)
        )

    def assemble(
        self,
        H,
        residual: np.ndarray,
        weight: np.ndarray,
        *,
        uniform_weight: Optional[float] = None,
        weights_are_uniform: Optional[bool] = None,
        weighted_residual: Optional[np.ndarray] = None,
        dense_gain_limit: int = 1000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        H_csr = H if getattr(H, "format", None) == "csr" else H.tocsr()
        data = np.asarray(H_csr.data, dtype=np.float64)
        residual = np.asarray(residual, dtype=np.float64)
        weight = np.asarray(weight, dtype=np.float64)

        if weight.size == 0:
            row_weight = None
            weighted_residual_values = residual
            scale = 1.0
        elif weights_are_uniform is True or uniform_weight is not None:
            scale = float(weight[0] if uniform_weight is None else uniform_weight)
            row_weight = None
            weighted_residual_values = residual if scale == 1.0 else residual * scale
        else:
            if weights_are_uniform is None and weight.size:
                first_weight = float(weight[0])
                if np.all(weight == first_weight):
                    scale = first_weight
                    row_weight = None
                    weighted_residual_values = residual if scale == 1.0 else residual * scale
                else:
                    scale = 1.0
                    row_weight = weight
                    weighted_residual_values = weight * residual if weighted_residual is None else weighted_residual
            else:
                scale = 1.0
                row_weight = weight
                weighted_residual_values = weight * residual if weighted_residual is None else weighted_residual

        if self.pair_slot.size:
            gain_values = data[self.pair_left_pos] * data[self.pair_right_pos]
            if row_weight is not None:
                gain_values = gain_values * row_weight[self.pair_rows]
            elif scale != 1.0:
                gain_values = gain_values * scale
            gain_data = np.bincount(
                self.pair_slot,
                weights=gain_values,
                minlength=self.gain_nnz,
            ).astype(np.float64, copy=False)
        else:
            gain_data = np.array([], dtype=np.float64)

        rhs_weights = data * weighted_residual_values[self.h_rows] if data.size else np.array([], dtype=np.float64)
        rhs = np.bincount(self.h_cols, weights=rhs_weights, minlength=self.shape[1]).astype(np.float64, copy=False)
        gain = SP_CSC_MATRIX(
            (gain_data, self.gain_indices, self.gain_indptr),
            shape=self.gain_shape,
            copy=False,
        )
        if gain.shape[0] <= dense_gain_limit:
            return gain.toarray(), rhs
        return gain, rhs


class LowerNormalEquationCscPlan:
    """Solver-ready lower-CSC assembly plan for repeated ``H.T W H`` builds.

    The plan owns the CSC ``indptr``/``indices`` and a reusable ``data`` buffer.
    Iterations refresh only numeric values for the lower triangle; they do not
    build the full normal matrix, call ``tril()``, or expand into a cached
    structural pattern.
    """

    def __init__(
        self,
        *,
        shape: Tuple[int, int],
        h_indptr: np.ndarray,
        h_indices: np.ndarray,
        h_rows: np.ndarray,
        h_cols: np.ndarray,
        pair_left_pos: np.ndarray,
        pair_right_pos: np.ndarray,
        pair_rows: np.ndarray,
        pair_slot: np.ndarray,
        gain_indptr: np.ndarray,
        gain_indices: np.ndarray,
    ):
        self.shape = tuple(int(item) for item in shape)
        self.h_indptr = np.asarray(h_indptr, dtype=np.int32)
        self.h_indices = np.asarray(h_indices, dtype=np.int32)
        self.h_rows = np.asarray(h_rows, dtype=np.int32)
        self.h_cols = np.asarray(h_cols, dtype=np.int32)
        self.pair_left_pos = np.asarray(pair_left_pos, dtype=np.int32)
        self.pair_right_pos = np.asarray(pair_right_pos, dtype=np.int32)
        self.pair_rows = np.asarray(pair_rows, dtype=np.int32)
        self.pair_slot = np.asarray(pair_slot, dtype=np.int32)
        self.gain_indptr = np.asarray(gain_indptr, dtype=np.int32)
        self.gain_indices = np.asarray(gain_indices, dtype=np.int32)
        self.gain_shape = (self.shape[1], self.shape[1])
        self.gain_nnz = int(self.gain_indices.size)
        self.gain_data = np.zeros(self.gain_nnz, dtype=np.float64)
        self.rhs = np.zeros(self.shape[1], dtype=np.float64)
        self._pair_values = np.empty(int(self.pair_slot.size), dtype=np.float64)
        self._rhs_values = np.empty(int(self.h_cols.size), dtype=np.float64)
        self._fixed_pair_weight = None
        self._fixed_h_weight = None
        self.gain = SP_CSC_MATRIX(
            (self.gain_data, self.gain_indices, self.gain_indptr),
            shape=self.gain_shape,
            copy=False,
        )
        self.pair_count = int(self.pair_slot.size)

    @classmethod
    def from_jacobian(cls, H) -> "LowerNormalEquationCscPlan":
        if not is_sparse_matrix(H):
            raise TypeError("LowerNormalEquationCscPlan requires a sparse Jacobian")
        H_csr = H if getattr(H, "format", None) == "csr" else H.tocsr()
        if not H_csr.has_sorted_indices:
            H_csr = H_csr.copy()
            H_csr.sort_indices()
        indptr = H_csr.indptr.astype(np.int32, copy=True)
        indices = H_csr.indices.astype(np.int32, copy=True)
        counts = np.diff(indptr).astype(np.int32, copy=False)
        h_rows = np.repeat(np.arange(H_csr.shape[0], dtype=np.int32), counts)
        h_cols = indices.copy()

        pair_counts = (counts.astype(np.int64, copy=False) * (counts.astype(np.int64, copy=False) + 1)) // 2
        total_pairs = int(pair_counts.sum())
        if total_pairs:
            pair_left_pos = np.empty(total_pairs, dtype=np.int32)
            pair_right_pos = np.empty(total_pairs, dtype=np.int32)
            pair_rows = np.empty(total_pairs, dtype=np.int32)
            nonzero_rows = np.flatnonzero(counts).astype(np.int32, copy=False)
            width_order = np.argsort(counts[nonzero_rows], kind="stable")
            rows_by_width = nonzero_rows[width_order]
            widths_by_row = counts[rows_by_width]
            width_breaks = np.concatenate(
                (
                    np.array([0], dtype=np.int64),
                    np.nonzero(widths_by_row[1:] != widths_by_row[:-1])[0] + 1,
                    np.array([rows_by_width.size], dtype=np.int64),
                )
            )
            offset = 0
            for group_start, group_stop in zip(width_breaks[:-1], width_breaks[1:]):
                rows = rows_by_width[int(group_start) : int(group_stop)]
                k = int(counts[rows[0]])
                per_row_pairs = (k * (k + 1)) // 2
                out_stop = offset + int(rows.size) * per_row_pairs
                if k == 1:
                    source = indptr[rows]
                    pair_left_pos[offset:out_stop] = source
                    pair_right_pos[offset:out_stop] = source
                    pair_rows[offset:out_stop] = rows
                else:
                    lower_left, lower_right = np.tril_indices(k)
                    lower_left = lower_left.astype(np.int32, copy=False)
                    lower_right = lower_right.astype(np.int32, copy=False)
                    starts = indptr[rows]
                    pair_left_pos[offset:out_stop] = (starts[:, None] + lower_left[None, :]).ravel()
                    pair_right_pos[offset:out_stop] = (starts[:, None] + lower_right[None, :]).ravel()
                    pair_rows[offset:out_stop].reshape(rows.size, per_row_pairs)[:] = rows[:, None]
                offset = out_stop
            n_state = int(H_csr.shape[1])
            gain_rows = indices[pair_left_pos].astype(np.int64, copy=False)
            gain_cols = indices[pair_right_pos].astype(np.int64, copy=False)
            raw_linear = gain_cols * np.int64(n_state) + gain_rows
            gain_linear, pair_slot = np.unique(raw_linear, return_inverse=True)
            gain_indices = (gain_linear % np.int64(n_state)).astype(np.int32, copy=False)
            gain_cols_unique = (gain_linear // np.int64(n_state)).astype(np.int64, copy=False)
            gain_indptr = np.zeros(n_state + 1, dtype=np.int32)
            np.add.at(gain_indptr, gain_cols_unique + 1, 1)
            np.cumsum(gain_indptr, out=gain_indptr)
            pair_slot = pair_slot.astype(np.int32, copy=False)
        else:
            pair_left_pos = np.array([], dtype=np.int32)
            pair_right_pos = np.array([], dtype=np.int32)
            pair_rows = np.array([], dtype=np.int32)
            pair_slot = np.array([], dtype=np.int32)
            gain_indices = np.array([], dtype=np.int32)
            gain_indptr = np.zeros(H_csr.shape[1] + 1, dtype=np.int32)

        return cls(
            shape=H_csr.shape,
            h_indptr=indptr,
            h_indices=indices,
            h_rows=h_rows,
            h_cols=h_cols,
            pair_left_pos=pair_left_pos,
            pair_right_pos=pair_right_pos,
            pair_rows=pair_rows,
            pair_slot=pair_slot,
            gain_indptr=gain_indptr,
            gain_indices=gain_indices,
        )

    def prepare_fixed_weights(self, weight: np.ndarray) -> None:
        """Cache ``weight[pair_rows]`` for repeated fixed-weight assemblies."""
        weight = np.asarray(weight, dtype=np.float64)
        if self.pair_rows.size:
            self._fixed_pair_weight = weight[self.pair_rows].copy()
        else:
            self._fixed_pair_weight = np.array([], dtype=np.float64)
        if self.h_rows.size:
            self._fixed_h_weight = weight[self.h_rows].copy()
        else:
            self._fixed_h_weight = np.array([], dtype=np.float64)

    def clear_fixed_weights(self) -> None:
        self._fixed_pair_weight = None
        self._fixed_h_weight = None

    def matches(self, H) -> bool:
        if not is_sparse_matrix(H) or tuple(H.shape) != self.shape:
            return False
        H_csr = H if getattr(H, "format", None) == "csr" else H.tocsr()
        if not H_csr.has_sorted_indices:
            return False
        return (
            int(H_csr.nnz) == int(self.h_indices.size)
            and H_csr.indptr.shape == self.h_indptr.shape
            and H_csr.indices.shape == self.h_indices.shape
            and np.array_equal(H_csr.indptr, self.h_indptr)
            and np.array_equal(H_csr.indices, self.h_indices)
        )

    def assemble(
        self,
        H,
        residual: np.ndarray,
        weight: np.ndarray,
        *,
        uniform_weight: Optional[float] = None,
        weights_are_uniform: Optional[bool] = None,
        weighted_residual: Optional[np.ndarray] = None,
        dense_gain_limit: int = 1000,
        assume_fixed_weights: bool = False,
        copy_rhs: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        H_csr = H if getattr(H, "format", None) == "csr" else H.tocsr()
        data = np.asarray(H_csr.data, dtype=np.float64)
        residual = np.asarray(residual, dtype=np.float64)
        weight = np.asarray(weight, dtype=np.float64)

        if weight.size == 0:
            row_weight = None
            weighted_residual_values = residual
            scale = 1.0
        elif weights_are_uniform is True or uniform_weight is not None:
            scale = float(weight[0] if uniform_weight is None else uniform_weight)
            row_weight = None
            weighted_residual_values = residual if scale == 1.0 else residual * scale
        else:
            if weights_are_uniform is None and weight.size:
                first_weight = float(weight[0])
                if np.all(weight == first_weight):
                    scale = first_weight
                    row_weight = None
                    weighted_residual_values = residual if scale == 1.0 else residual * scale
                else:
                    scale = 1.0
                    row_weight = weight
                    weighted_residual_values = None if (assume_fixed_weights and weighted_residual is None) else (
                        weight * residual if weighted_residual is None else weighted_residual
                    )
            else:
                scale = 1.0
                row_weight = weight
                weighted_residual_values = None if (assume_fixed_weights and weighted_residual is None) else (
                    weight * residual if weighted_residual is None else weighted_residual
                )

        if self.pair_slot.size:
            np.multiply(data[self.pair_left_pos], data[self.pair_right_pos], out=self._pair_values)
            if row_weight is not None:
                if assume_fixed_weights:
                    if self._fixed_pair_weight is None or self._fixed_pair_weight.shape[0] != self.pair_rows.size:
                        self.prepare_fixed_weights(row_weight)
                    np.multiply(self._pair_values, self._fixed_pair_weight, out=self._pair_values)
                else:
                    np.multiply(self._pair_values, row_weight[self.pair_rows], out=self._pair_values)
            elif scale != 1.0:
                self._pair_values *= scale
            self.gain_data[:] = np.bincount(
                self.pair_slot,
                weights=self._pair_values,
                minlength=self.gain_nnz,
            )
        else:
            self.gain_data.fill(0.0)

        if data.size:
            if weighted_residual_values is None and row_weight is not None and assume_fixed_weights:
                if self._fixed_h_weight is None or self._fixed_h_weight.shape[0] != self.h_rows.size:
                    self.prepare_fixed_weights(row_weight)
                np.multiply(data, residual[self.h_rows], out=self._rhs_values)
                np.multiply(self._rhs_values, self._fixed_h_weight, out=self._rhs_values)
            else:
                np.multiply(data, weighted_residual_values[self.h_rows], out=self._rhs_values)
            self.rhs[:] = np.bincount(
                self.h_cols,
                weights=self._rhs_values,
                minlength=self.shape[1],
            )
        else:
            self.rhs.fill(0.0)
        rhs = self.rhs.copy() if copy_rhs else self.rhs
        if self.gain.shape[0] <= dense_gain_limit:
            return self.gain.toarray(), rhs
        return self.gain, rhs


class CholmodAAtNormalEquationPlan:
    """Prototype CHOLMOD ``A A.T`` plan for ``H.T W H`` normal equations.

    ``A`` is the weighted transpose of the measurement Jacobian:
    ``A = H.T * sqrt(W)``.  CHOLMOD's AAt factorization then solves the same
    system as ``H.T W H`` without explicitly assembling the lower normal CSC.
    The plan keeps the transposed CSC structure fixed and refreshes only data.
    """

    def __init__(
        self,
        *,
        shape: Tuple[int, int],
        h_indptr: np.ndarray,
        h_indices: np.ndarray,
        h_rows: np.ndarray,
        h_cols: np.ndarray,
    ):
        self.shape = tuple(int(item) for item in shape)
        self.h_indptr = np.asarray(h_indptr, dtype=np.int32)
        self.h_indices = np.asarray(h_indices, dtype=np.int32)
        self.h_rows = np.asarray(h_rows, dtype=np.int32)
        self.h_cols = np.asarray(h_cols, dtype=np.int32)
        self.a_indptr = self.h_indptr
        self.a_indices = self.h_indices
        self.a_shape = (self.shape[1], self.shape[0])
        self.a_data = np.zeros(int(self.h_indices.size), dtype=np.float64)
        self.rhs = np.zeros(self.shape[1], dtype=np.float64)
        self._rhs_values = np.empty(int(self.h_cols.size), dtype=np.float64)
        self._fixed_sqrt_weight = None
        self._fixed_h_weight = None
        self.A = SP_CSC_MATRIX(
            (self.a_data, self.a_indices, self.a_indptr),
            shape=self.a_shape,
            copy=False,
        )

    @classmethod
    def from_jacobian(cls, H) -> "CholmodAAtNormalEquationPlan":
        if not is_sparse_matrix(H):
            raise TypeError("CholmodAAtNormalEquationPlan requires a sparse Jacobian")
        H_csr = H if getattr(H, "format", None) == "csr" else H.tocsr()
        if not H_csr.has_sorted_indices:
            H_csr = H_csr.copy()
            H_csr.sort_indices()
        indptr = H_csr.indptr.astype(np.int32, copy=True)
        indices = H_csr.indices.astype(np.int32, copy=True)
        counts = np.diff(indptr).astype(np.int32, copy=False)
        h_rows = np.repeat(np.arange(H_csr.shape[0], dtype=np.int32), counts)
        h_cols = indices.copy()
        return cls(
            shape=H_csr.shape,
            h_indptr=indptr,
            h_indices=indices,
            h_rows=h_rows,
            h_cols=h_cols,
        )

    def prepare_fixed_weights(self, weight: np.ndarray) -> None:
        weight = np.asarray(weight, dtype=np.float64)
        if self.h_rows.size:
            self._fixed_sqrt_weight = np.sqrt(weight[self.h_rows]).astype(np.float64, copy=True)
            self._fixed_h_weight = weight[self.h_rows].copy()
        else:
            self._fixed_sqrt_weight = np.array([], dtype=np.float64)
            self._fixed_h_weight = np.array([], dtype=np.float64)

    def clear_fixed_weights(self) -> None:
        self._fixed_sqrt_weight = None
        self._fixed_h_weight = None

    def matches(self, H) -> bool:
        if not is_sparse_matrix(H) or tuple(H.shape) != self.shape:
            return False
        H_csr = H if getattr(H, "format", None) == "csr" else H.tocsr()
        if not H_csr.has_sorted_indices:
            return False
        return (
            int(H_csr.nnz) == int(self.h_indices.size)
            and H_csr.indptr.shape == self.h_indptr.shape
            and H_csr.indices.shape == self.h_indices.shape
            and np.array_equal(H_csr.indptr, self.h_indptr)
            and np.array_equal(H_csr.indices, self.h_indices)
        )

    def assemble(
        self,
        H,
        residual: np.ndarray,
        weight: np.ndarray,
        *,
        uniform_weight: Optional[float] = None,
        weights_are_uniform: Optional[bool] = None,
        weighted_residual: Optional[np.ndarray] = None,
        assume_fixed_weights: bool = False,
        copy_rhs: bool = False,
        assume_pattern_matches: bool = False,
    ):
        H_csr = H if getattr(H, "format", None) == "csr" else H.tocsr()
        if not H_csr.has_sorted_indices:
            H_csr = H_csr.copy()
            H_csr.sort_indices()
        if not assume_pattern_matches and not self.matches(H_csr):
            raise ValueError("Jacobian sparse pattern does not match the AAt plan")

        data = np.asarray(H_csr.data, dtype=np.float64)
        residual = np.asarray(residual, dtype=np.float64)
        weight = np.asarray(weight, dtype=np.float64)

        if weight.size == 0:
            row_weight = None
            sqrt_scale = 1.0
            weighted_residual_values = residual
        elif weights_are_uniform is True or uniform_weight is not None:
            scale = float(weight[0] if uniform_weight is None else uniform_weight)
            row_weight = None
            sqrt_scale = math.sqrt(scale)
            weighted_residual_values = residual if scale == 1.0 else residual * scale
        else:
            if weights_are_uniform is None and weight.size:
                first_weight = float(weight[0])
                if np.all(weight == first_weight):
                    row_weight = None
                    sqrt_scale = math.sqrt(first_weight)
                    weighted_residual_values = residual if first_weight == 1.0 else residual * first_weight
                else:
                    row_weight = weight
                    sqrt_scale = 1.0
                    weighted_residual_values = None if (assume_fixed_weights and weighted_residual is None) else (
                        weight * residual if weighted_residual is None else weighted_residual
                    )
            else:
                row_weight = weight
                sqrt_scale = 1.0
                weighted_residual_values = None if (assume_fixed_weights and weighted_residual is None) else (
                    weight * residual if weighted_residual is None else weighted_residual
                )

        if data.size:
            if row_weight is not None:
                if assume_fixed_weights:
                    if self._fixed_sqrt_weight is None or self._fixed_sqrt_weight.shape[0] != self.h_rows.size:
                        self.prepare_fixed_weights(row_weight)
                    np.multiply(data, self._fixed_sqrt_weight, out=self.a_data)
                else:
                    np.multiply(data, np.sqrt(row_weight[self.h_rows]), out=self.a_data)
            elif sqrt_scale != 1.0:
                np.multiply(data, sqrt_scale, out=self.a_data)
            else:
                self.a_data[:] = data

            if weighted_residual_values is None and row_weight is not None and assume_fixed_weights:
                if self._fixed_h_weight is None or self._fixed_h_weight.shape[0] != self.h_rows.size:
                    self.prepare_fixed_weights(row_weight)
                np.multiply(data, residual[self.h_rows], out=self._rhs_values)
                np.multiply(self._rhs_values, self._fixed_h_weight, out=self._rhs_values)
            else:
                np.multiply(data, weighted_residual_values[self.h_rows], out=self._rhs_values)
            self.rhs[:] = np.bincount(
                self.h_cols,
                weights=self._rhs_values,
                minlength=self.shape[1],
            )
        else:
            self.a_data.fill(0.0)
            self.rhs.fill(0.0)

        rhs = self.rhs.copy() if copy_rhs else self.rhs
        return self.A, rhs


class CholmodAAtNormalEquationSolver:
    """Prototype normal-equation solver backed by CHOLMOD ``A A.T`` factorization."""

    def __init__(self, assume_fixed_pattern: bool = False):
        self._cholmod_factor = None
        self._cholmod_pattern = None
        self._cholmod_disabled = False
        self.assume_fixed_pattern = bool(assume_fixed_pattern)
        self._fallback_solver = NormalEquationSolver(assume_fixed_pattern=assume_fixed_pattern)

    @staticmethod
    def _sparse_pattern(matrix) -> Tuple[Tuple[int, int], int, np.ndarray, np.ndarray]:
        csc = matrix if getattr(matrix, "format", None) == "csc" else matrix.tocsc()
        return (
            csc.shape,
            int(csc.nnz),
            csc.indptr.astype(np.int64, copy=True),
            csc.indices.astype(np.int64, copy=True),
        )

    @staticmethod
    def _same_sparse_pattern(pattern, matrix) -> bool:
        if pattern is None:
            return False
        csc = matrix if getattr(matrix, "format", None) == "csc" else matrix.tocsc()
        shape, nnz, indptr, indices = pattern
        return (
            shape == csc.shape
            and nnz == int(csc.nnz)
            and indptr.shape == csc.indptr.shape
            and indices.shape == csc.indices.shape
            and np.array_equal(indptr, csc.indptr)
            and np.array_equal(indices, csc.indices)
        )

    def _solve_sparse_cholmod_aat(self, A_csc, rhs: np.ndarray):
        if CHOLMOD_ANALYZE_AAT is not None:
            if self._cholmod_factor is None or (
                not self.assume_fixed_pattern and not self._same_sparse_pattern(self._cholmod_pattern, A_csc)
            ):
                self._cholmod_factor = CHOLMOD_ANALYZE_AAT(A_csc)
                if not self.assume_fixed_pattern:
                    self._cholmod_pattern = self._sparse_pattern(A_csc)
            self._cholmod_factor.cholesky_AAt_inplace(A_csc)
            return self._cholmod_factor(rhs), None
        if CHOLMOD_CHOLESKY_AAT is not None:
            factor = CHOLMOD_CHOLESKY_AAT(A_csc)
            return factor(rhs), None
        raise RuntimeError("CHOLMOD AAt is not available")

    def solve(
        self,
        A,
        rhs: np.ndarray,
        return_factor_diag: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not is_sparse_matrix(A):
            raise TypeError("CholmodAAtNormalEquationSolver requires a sparse weighted transpose matrix")
        A_csc = A if getattr(A, "format", None) == "csc" else A.tocsc()
        if not self._cholmod_disabled and (CHOLMOD_ANALYZE_AAT is not None or CHOLMOD_CHOLESKY_AAT is not None):
            try:
                return self._solve_sparse_cholmod_aat(A_csc, rhs)
            except Exception:
                self._cholmod_factor = None
                self._cholmod_pattern = None
                self._cholmod_disabled = True
        gain = A_csc @ A_csc.T
        return self._fallback_solver.solve(gain.tocsc(), rhs, return_factor_diag=return_factor_diag)

    def solve_from_plan(
        self,
        plan: CholmodAAtNormalEquationPlan,
        H,
        residual: np.ndarray,
        weight: np.ndarray,
        *,
        uniform_weight: Optional[float] = None,
        weights_are_uniform: Optional[bool] = None,
        weighted_residual: Optional[np.ndarray] = None,
        assume_fixed_weights: bool = False,
        return_factor_diag: bool = False,
        assume_pattern_matches: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        A, rhs = plan.assemble(
            H,
            residual,
            weight,
            uniform_weight=uniform_weight,
            weights_are_uniform=weights_are_uniform,
            weighted_residual=weighted_residual,
            assume_fixed_weights=assume_fixed_weights,
            copy_rhs=False,
            assume_pattern_matches=assume_pattern_matches,
        )
        return self.solve(A, rhs, return_factor_diag=return_factor_diag)


def measurement_leverage(H, gain_inv: np.ndarray) -> np.ndarray:
    """Compute diag(H gain_inv H.T) for dense or sparse measurement Jacobians."""
    projected = H @ gain_inv
    if is_sparse_matrix(H):
        return np.asarray(H.multiply(projected).sum(axis=1)).ravel()
    return np.sum(projected * H, axis=1)


def inverse_gain_for_bad_data(gain: np.ndarray, dense_state_limit: int = 5000):
    """Return gain inverse for bad-data leverage, or None when full inverse is too large.

    Normalized-residual bad-data detection needs diag(H G^-1 H.T). A full inverse is
    acceptable for the named examples, but multi-10k-state systems would require
    tens of GB if forced dense. For those large runs the caller can fall back to a
    residual-only score instead of failing after the state estimate has converged.
    """
    if is_sparse_matrix(gain):
        if gain.shape[0] > dense_state_limit:
            return None
        gain = gain.toarray()
    try:
        return np.linalg.inv(gain)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(gain)


def _rank_from_singular_values(singular_values: np.ndarray, shape: Tuple[int, int]) -> Tuple[int, float]:
    """Return numerical rank using NumPy's standard SVD tolerance rule."""
    tol = max(shape) * np.finfo(float).eps * (singular_values[0] if singular_values.size else 1.0)
    rank = int(np.sum(singular_values > tol))
    return rank, float(tol)


def observability_rank_details(
    H: np.ndarray,
    state_count: int,
    normal_matrix: np.ndarray = None,
    normal_factor_diag: np.ndarray = None,
    dense_svd_limit: int = 2000,
) -> Tuple[int, int, np.ndarray, List[Tuple[int, float]]]:
    """Fast rank analysis for tall state-estimation Jacobians.

    Well-conditioned observable systems are the common path; a Cholesky factor of
    a positive-definite normal matrix proves full column rank without computing a
    full spectral decomposition. `estimate()` passes its already assembled
    H.T @ W @ H here, which has the same rank test outcome when measurement
    weights are positive. Full SVD is kept as the conservative fallback for
    deficient or numerically marginal cases so weak-state reporting remains
    available.
    """
    state_count = int(state_count)
    measurement_count = int(H.shape[0])
    min_deficiency = max(0, state_count - measurement_count)
    full_column_rank_possible = measurement_count >= state_count
    gram = H.T @ H if normal_matrix is None else normal_matrix
    weak_states: List[Tuple[int, float]] = []

    if is_sparse_matrix(gram):
        diag_values = gram.diagonal()
    else:
        diag_values = np.diag(gram) if gram.size else np.array([], dtype=np.float64)
    scale = float(np.sqrt(np.max(diag_values))) if diag_values.size else 1.0
    tol = max(H.shape) * np.finfo(float).eps * max(scale, 1.0)
    if full_column_rank_possible and normal_factor_diag is not None:
        diag = np.asarray(normal_factor_diag, dtype=np.float64)
        if diag.size == state_count and float(np.min(diag)) > tol:
            return state_count, 0, np.array([], dtype=np.float64), weak_states
    elif full_column_rank_possible and is_sparse_matrix(gram) and CHOLMOD_CHOLESKY is not None:
        try:
            CHOLMOD_CHOLESKY(gram.tocsc())
            return state_count, 0, np.array([], dtype=np.float64), weak_states
        except Exception:
            pass
    if full_column_rank_possible and normal_factor_diag is None and is_sparse_matrix(gram):
        gram_csc = gram.tocsc()
        try:
            lu = SP_SPLU(gram_csc, diag_pivot_thresh=0.0, permc_spec="MMD_AT_PLUS_A")
            diag = np.abs(lu.U.diagonal())
            if diag.size == state_count and float(np.min(diag)) > tol:
                return state_count, 0, np.array([], dtype=np.float64), weak_states
        except Exception:
            pass
        try:
            lu = SP_SPLU(gram_csc)
            diag = np.abs(lu.U.diagonal())
            if diag.size == state_count and float(np.min(diag)) > tol:
                return state_count, 0, np.array([], dtype=np.float64), weak_states
        except Exception:
            pass
    elif full_column_rank_possible and normal_factor_diag is None:
        chol, info = DPOTRF(gram, lower=True, clean=False, overwrite_a=False)
        if info == 0:
            diag = np.diag(chol)
            if diag.size == state_count and float(np.min(diag)) > tol:
                return state_count, 0, np.array([], dtype=np.float64), weak_states

    if state_count > dense_svd_limit:
        # Dense SVD of multi-thousand-state sparse Jacobians can dominate the
        # whole SE run. If sparse/dense factorization cannot certify full rank,
        # return a conservative non-observable result using the normal diagonal
        # to identify the weakest state candidates.
        if is_sparse_matrix(gram):
            diag_values = np.asarray(gram.diagonal(), dtype=np.float64)
        else:
            diag_values = np.asarray(np.diag(gram), dtype=np.float64) if gram.size else np.array([], dtype=np.float64)
        if diag_values.size:
            weak_count = min(10, diag_values.size)
            weak_idx = np.argsort(diag_values)[:weak_count]
            weak_states = [(int(idx), float(diag_values[int(idx)])) for idx in weak_idx]
            near_zero = int(np.count_nonzero(diag_values <= max(tol * tol, 1e-24)))
            deficiency = max(1, near_zero, min_deficiency)
        else:
            deficiency = state_count
        return max(0, state_count - deficiency), deficiency, np.array([], dtype=np.float64), weak_states

    if is_sparse_matrix(gram):
        gram = gram.toarray()
    H_for_svd = H.toarray() if is_sparse_matrix(H) else H
    _, singular_values, vh = np.linalg.svd(H_for_svd, full_matrices=False)
    rank, _ = _rank_from_singular_values(singular_values, H.shape)
    deficiency = state_count - rank
    if deficiency > 0 and vh.size:
        null_vectors = vh[-deficiency:, :]
        scores = np.max(np.abs(null_vectors), axis=0)
        top = np.argsort(scores)[::-1][: min(10, scores.size)]
        weak_states = [(int(idx), float(scores[idx])) for idx in top]

    return rank, max(0, deficiency), singular_values, weak_states


def unanchored_angle_state_indices(
    H: np.ndarray,
    angle_cols: Sequence[int],
    tol: float = 1e-12,
) -> List[int]:
    """Find AC angle components whose measurement rows only define relative angles.

    Large sparse cases can have a tiny rank deficiency even though every node has
    local P/Q rows. In that situation the weak-state diagonal heuristic often
    points at arbitrary low-diagonal columns. This structural pass looks only at
    angle columns of H: rows with zero coefficient sum tie angles relatively,
    while rows with a nonzero sum provide an absolute anchor through a reference
    angle or a direct angle measurement.
    """
    angle_cols = [int(idx) for idx in angle_cols]
    if not angle_cols:
        return []
    angle_cols_array = np.asarray(angle_cols, dtype=np.int64)

    if is_sparse_matrix(H):
        H_csr = H if getattr(H, "format", None) == "csr" else H.tocsr()
        n = len(angle_cols)
        if np.array_equal(angle_cols_array, np.arange(n, dtype=np.int64)):
            mask = H_csr.indices < n
            if np.any(mask):
                row_nnz = np.diff(H_csr.indptr).astype(np.int64, copy=False)
                entry_rows = np.repeat(np.arange(H_csr.shape[0], dtype=np.int64), row_nnz)
                angle_rows = entry_rows[mask]
                angle_indices = H_csr.indices[mask].astype(np.int64, copy=False)
                angle_data = H_csr.data[mask]
                nnz_by_row = np.bincount(angle_rows, minlength=H_csr.shape[0])
                active_rows = np.flatnonzero(nnz_by_row).astype(np.int64, copy=False)
                degree = np.bincount(angle_indices, minlength=n)
                filtered_indptr = np.empty(H_csr.shape[0] + 1, dtype=np.int64)
                filtered_indptr[0] = 0
                np.cumsum(nnz_by_row, out=filtered_indptr[1:])
                positions = np.arange(angle_indices.size, dtype=np.int64)
                row_start_for_entry = filtered_indptr[angle_rows]
                nonfirst = positions > row_start_for_entry
                if np.any(nonfirst):
                    left = angle_indices[row_start_for_entry[nonfirst]]
                    right = angle_indices[positions[nonfirst]]
                    graph = SP_COO_MATRIX(
                        (
                            np.ones(left.size * 2, dtype=np.int8),
                            (np.concatenate((left, right)), np.concatenate((right, left))),
                        ),
                        shape=(n, n),
                    ).tocsr()
                else:
                    graph = SP_CSR_MATRIX((n, n), dtype=np.int8)
                n_components, labels = SP_CONNECTED_COMPONENTS(graph, directed=False, return_labels=True)
                anchored = np.zeros(int(n_components), dtype=bool)
                if active_rows.size:
                    starts = filtered_indptr[active_rows]
                    row_sums = np.add.reduceat(angle_data, starts)
                    anchor_rows = np.abs(row_sums) > tol
                    if np.any(anchor_rows):
                        first_cols = angle_indices[starts]
                        anchored[labels[first_cols[anchor_rows]]] = True
            else:
                degree = np.zeros(n, dtype=np.int64)
                n_components = n
                labels = np.arange(n, dtype=np.int32)
                anchored = np.zeros(n, dtype=bool)
            state_indices = []
            for component in np.flatnonzero(~anchored).tolist():
                local_cols = np.flatnonzero(labels == int(component))
                if local_cols.size == 0:
                    continue
                local_degree = degree[local_cols]
                best = local_cols[local_degree == int(np.max(local_degree))]
                representative = int(best[np.argmin(angle_cols_array[best])])
                state_indices.append((int(local_cols.size), int(angle_cols_array[representative])))
            state_indices.sort(key=lambda item: (-item[0], item[1]))
            return [idx for _size, idx in state_indices]

        sub = H[:, angle_cols].tocsr()
        nnz_by_row = np.diff(sub.indptr)
        active_rows = np.flatnonzero(nnz_by_row).astype(np.int64, copy=False)
        degree = np.bincount(sub.indices, minlength=len(angle_cols)) if sub.indices.size else np.zeros(len(angle_cols), dtype=np.int64)
        if sub.indices.size:
            positions = np.arange(sub.indices.size, dtype=np.int64)
            entry_rows = np.repeat(np.arange(sub.shape[0], dtype=np.int64), nnz_by_row)
            row_start_for_entry = sub.indptr[entry_rows]
            nonfirst = positions > row_start_for_entry
            if np.any(nonfirst):
                left = sub.indices[row_start_for_entry[nonfirst]]
                right = sub.indices[positions[nonfirst]]
                graph = SP_COO_MATRIX(
                    (
                        np.ones(left.size * 2, dtype=np.int8),
                        (np.concatenate((left, right)), np.concatenate((right, left))),
                    ),
                    shape=(n, n),
                ).tocsr()
            else:
                graph = SP_CSR_MATRIX((n, n), dtype=np.int8)
            n_components, labels = SP_CONNECTED_COMPONENTS(graph, directed=False, return_labels=True)
            anchored = np.zeros(int(n_components), dtype=bool)
            if active_rows.size:
                starts = sub.indptr[active_rows]
                row_sums = np.add.reduceat(sub.data, starts)
                anchor_rows = np.abs(row_sums) > tol
                if np.any(anchor_rows):
                    first_cols = sub.indices[starts]
                    anchored[labels[first_cols[anchor_rows]]] = True
        else:
            n_components = n
            labels = np.arange(n, dtype=np.int32)
            anchored = np.zeros(n, dtype=bool)
        state_indices = []
        for component in np.flatnonzero(~anchored).tolist():
            local_cols = np.flatnonzero(labels == int(component))
            if local_cols.size == 0:
                continue
            local_degree = degree[local_cols]
            best = local_cols[local_degree == int(np.max(local_degree))]
            representative = int(best[np.argmin(angle_cols_array[best])])
            state_indices.append((int(local_cols.size), int(angle_cols_array[representative])))
        state_indices.sort(key=lambda item: (-item[0], item[1]))
        return [idx for _size, idx in state_indices]
    else:
        matrix = np.asarray(H)
        sub = matrix[:, angle_cols]
        active_rows = None
        degree = np.count_nonzero(np.abs(sub) > tol, axis=0)

    n = len(angle_cols)
    parent = np.arange(n, dtype=np.int32)
    anchored = np.zeros(n, dtype=bool)

    def find(pos: int) -> int:
        while int(parent[pos]) != pos:
            parent[pos] = parent[int(parent[pos])]
            pos = int(parent[pos])
        return int(pos)

    def union(left: int, right: int) -> int:
        root_l = find(left)
        root_r = find(right)
        if root_l != root_r:
            parent[root_r] = root_l
            anchored[root_l] = anchored[root_l] or anchored[root_r]
        return root_l

    for row in range(sub.shape[0]):
        vals = sub[row, :]
        cols = np.flatnonzero(np.abs(vals) > tol)
        if cols.size == 0:
            continue
        root = int(cols[0])
        for col in cols[1:]:
            root = union(root, int(col))
        root = find(root)
        if abs(float(np.sum(vals[cols]))) > tol:
            anchored[root] = True

    components = {}
    for local_col in range(n):
        components.setdefault(find(local_col), []).append(local_col)

    state_indices = []
    for root, local_cols in components.items():
        root = find(root)
        if anchored[root]:
            continue
        representative = max(local_cols, key=lambda col: (int(degree[col]), -int(angle_cols[col])))
        state_indices.append((len(local_cols), int(angle_cols[representative])))
    state_indices.sort(key=lambda item: (-item[0], item[1]))
    return [idx for _, idx in state_indices]


class NormalEquationSolver:
    """Solve repeated normal equations, reusing CHOLMOD symbolic analysis when available."""

    def __init__(self, assume_fixed_pattern: bool = False):
        self._cholmod_factor = None
        self._cholmod_pattern = None
        self._cholmod_disabled = False
        self.assume_fixed_pattern = bool(assume_fixed_pattern)

    @staticmethod
    def _sparse_pattern(matrix) -> Tuple[Tuple[int, int], int, np.ndarray, np.ndarray]:
        return (
            matrix.shape,
            int(matrix.nnz),
            matrix.indptr.astype(np.int64, copy=True),
            matrix.indices.astype(np.int64, copy=True),
        )

    @staticmethod
    def _same_sparse_pattern(pattern, matrix) -> bool:
        if pattern is None:
            return False
        shape, nnz, indptr, indices = pattern
        return (
            shape == matrix.shape
            and nnz == int(matrix.nnz)
            and indptr.shape == matrix.indptr.shape
            and indices.shape == matrix.indices.shape
            and np.array_equal(indptr, matrix.indptr)
            and np.array_equal(indices, matrix.indices)
        )

    def _solve_sparse_cholmod(self, gain_csc, rhs: np.ndarray):
        if CHOLMOD_ANALYZE is not None:
            if self._cholmod_factor is None or (
                not self.assume_fixed_pattern and not self._same_sparse_pattern(self._cholmod_pattern, gain_csc)
            ):
                self._cholmod_factor = CHOLMOD_ANALYZE(gain_csc)
                if not self.assume_fixed_pattern:
                    self._cholmod_pattern = self._sparse_pattern(gain_csc)
            self._cholmod_factor.cholesky_inplace(gain_csc)
            return self._cholmod_factor(rhs), None
        if CHOLMOD_CHOLESKY is not None:
            factor = CHOLMOD_CHOLESKY(gain_csc)
            return factor(rhs), None
        raise RuntimeError("CHOLMOD is not available")

    def solve(
        self,
        gain: np.ndarray,
        rhs: np.ndarray,
        return_factor_diag: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Solve normal equations, preferring sparse SPD Cholesky when available."""
        if is_sparse_matrix(gain):
            gain_csc = gain if getattr(gain, "format", None) == "csc" else gain.tocsc()
            if not self._cholmod_disabled and (CHOLMOD_ANALYZE is not None or CHOLMOD_CHOLESKY is not None):
                try:
                    return self._solve_sparse_cholmod(gain_csc, rhs)
                except Exception:
                    self._cholmod_factor = None
                    self._cholmod_pattern = None
                    self._cholmod_disabled = True
            return _solve_normal_equations_no_cholmod(gain_csc, rhs, return_factor_diag)
        return _solve_dense_normal_equations(gain, rhs, return_factor_diag)


def _solve_normal_equations_no_cholmod(
    gain_csc,
    rhs: np.ndarray,
    return_factor_diag: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        lu = SP_SPLU(gain_csc, diag_pivot_thresh=0.0, permc_spec="MMD_AT_PLUS_A")
        factor_diag = np.abs(lu.U.diagonal()) if return_factor_diag else None
        return lu.solve(rhs), factor_diag
    except Exception:
        pass
    try:
        lu = SP_SPLU(gain_csc)
        factor_diag = np.abs(lu.U.diagonal()) if return_factor_diag else None
        return lu.solve(rhs), factor_diag
    except Exception:
        pass
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SP_MATRIX_RANK_WARNING)
            dx = SP_SPSOLVE(gain_csc, rhs)
        return dx, None
    except Exception:
        pass
    return _solve_dense_normal_equations(gain_csc.toarray(), rhs, return_factor_diag)


def _solve_dense_normal_equations(
    gain: np.ndarray,
    rhs: np.ndarray,
    return_factor_diag: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        factor, dx, info = DPOSV(gain, rhs.copy(), lower=True, overwrite_a=False, overwrite_b=True)
        if info == 0:
            factor_diag = np.diag(factor).copy() if return_factor_diag else None
            return dx, factor_diag
    except Exception:
        pass
    try:
        factor = CHO_FACTOR(gain, lower=True, check_finite=False)
        dx = CHO_SOLVE(factor, rhs, check_finite=False)
        factor_diag = np.diag(factor[0]).copy() if return_factor_diag else None
        return dx, factor_diag
    except Exception:
        pass
    try:
        return np.linalg.solve(gain, rhs), None
    except np.linalg.LinAlgError:
        return np.linalg.pinv(gain) @ rhs, None


def solve_normal_equations_with_factor(
    gain: np.ndarray,
    rhs: np.ndarray,
    return_factor_diag: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve normal equations and return the Cholesky diagonal when available."""
    return NormalEquationSolver().solve(gain, rhs, return_factor_diag=return_factor_diag)


def build_normal_equations(
    H: np.ndarray,
    residual: np.ndarray,
    weight: np.ndarray,
    dense_gain_limit: int = 1000,
    uniform_weight: Optional[float] = None,
    weights_are_uniform: Optional[bool] = None,
    weighted_residual: Optional[np.ndarray] = None,
    normal_pattern=None,
    assume_normal_pattern_matches: bool = False,
    normal_assembly_plan: Optional[NormalEquationAssemblyPlan] = None,
    triangular: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build WLS normal equations while avoiding WH allocation for uniform weights."""
    triangular_mode = None if triangular is None else str(triangular).lower()
    if triangular_mode not in (None, "lower"):
        raise ValueError(f"Unsupported normal-equation triangular mode: {triangular!r}")
    if is_sparse_matrix(H):
        if (
            normal_assembly_plan is not None
            and normal_assembly_plan.use_direct_assembly
            and normal_assembly_plan.matches(H)
        ):
            return normal_assembly_plan.assemble(
                H,
                residual,
                weight,
                uniform_weight=uniform_weight,
                weights_are_uniform=weights_are_uniform,
                weighted_residual=weighted_residual,
                dense_gain_limit=dense_gain_limit,
            )
        if weight.size == 0:
            gain = H.T @ H
            rhs = H.T @ residual
        elif weights_are_uniform is True or uniform_weight is not None:
            if uniform_weight is None:
                uniform_weight = float(weight[0])
            gain = H.T @ H
            rhs = H.T @ residual
            if uniform_weight != 1.0:
                gain = uniform_weight * gain
                rhs = uniform_weight * rhs
        elif weights_are_uniform is False:
            weighted_residual = weight * residual if weighted_residual is None else weighted_residual
            weighted_H = _row_weighted_sparse_matrix(H, weight)
            gain = H.T @ weighted_H
            rhs = H.T @ weighted_residual
        else:
            first_weight = float(weight[0])
            if np.all(weight == first_weight):
                gain = H.T @ H
                rhs = H.T @ residual
                if first_weight != 1.0:
                    gain = first_weight * gain
                    rhs = first_weight * rhs
            else:
                weighted_residual = weight * residual if weighted_residual is None else weighted_residual
                weighted_H = _row_weighted_sparse_matrix(H, weight)
                gain = H.T @ weighted_H
                rhs = H.T @ weighted_residual
        if is_sparse_matrix(gain):
            if triangular_mode == "lower":
                gain = SP_TRIL(gain, format="csc")
                if normal_pattern is not None:
                    normal_pattern = SP_TRIL(normal_pattern, format="csc")
            if assume_normal_pattern_matches:
                gain = gain.tocsc() if getattr(gain, "format", None) != "csc" else gain
            else:
                if normal_pattern is None:
                    normal_pattern = _normal_equation_structural_pattern(H, triangular=triangular_mode)
                gain = _expand_sparse_matrix_to_pattern(gain, normal_pattern)
            gain = gain.toarray() if gain.shape[0] <= dense_gain_limit else gain.tocsc()
        return gain, np.asarray(rhs, dtype=np.float64).ravel()

    if weight.size == 0:
        gain = H.T @ H
        rhs = H.T @ residual
        if triangular_mode == "lower":
            gain = np.tril(gain)
        return gain, rhs

    if weights_are_uniform is True or uniform_weight is not None:
        if uniform_weight is None:
            uniform_weight = float(weight[0])
        gain = H.T @ H
        rhs = H.T @ residual
        if uniform_weight != 1.0:
            gain = uniform_weight * gain
            rhs = uniform_weight * rhs
        if triangular_mode == "lower":
            gain = np.tril(gain)
        return gain, rhs

    if weights_are_uniform is False:
        weighted_residual = weight * residual if weighted_residual is None else weighted_residual
        WH = weight[:, None] * H
        gain = H.T @ WH
        rhs = H.T @ weighted_residual
        if triangular_mode == "lower":
            gain = np.tril(gain)
        return gain, rhs

    first_weight = float(weight[0])
    if np.all(weight == first_weight):
        gain = H.T @ H
        rhs = H.T @ residual
        if first_weight != 1.0:
            gain = first_weight * gain
            rhs = first_weight * rhs
        if triangular_mode == "lower":
            gain = np.tril(gain)
        return gain, rhs

    WH = weight[:, None] * H
    gain = H.T @ WH
    weighted_residual = weight * residual if weighted_residual is None else weighted_residual
    rhs = H.T @ weighted_residual
    if triangular_mode == "lower":
        gain = np.tril(gain)
    return gain, rhs
