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


ANGLE_MEASUREMENT_TYPES = frozenset(("ANGLE", "THETA", "ANGLE_DIFF", "THETA_DIFF"))
_NORMAL_EQUATION_PATTERN_CACHE = {}
_SPARSE_PATTERN_EXPANSION_CACHE = {}
_SPARSE_PATTERN_LINEAR_INDEX_CACHE = {}

def targeted_redundancy_count(state_count: int, ratio: float) -> int:
    """Return the configured pseudo-measurement redundancy target as a row count."""
    return max(0, int(math.ceil(max(0.0, float(ratio)) * max(0, int(state_count)))))


def observability_weak_direction(
    H,
    state_labels: Sequence[str],
    weak_states: Sequence[Tuple[str, float]] = (),
    dense_svd_limit: int = 2000,
) -> np.ndarray:
    """Return a normalized state-space direction that is currently weakest."""
    state_count = len(state_labels)
    if state_count <= 0:
        return np.array([], dtype=np.float64)

    direction = np.zeros(state_count, dtype=np.float64)
    if weak_states:
        label_to_pos = {label: pos for pos, label in enumerate(state_labels)}
        for label, score in weak_states:
            pos = label_to_pos.get(label)
            if pos is not None:
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
        self._cached_has_unique_slots = False
        self._cached_has_duplicate_slots = False
        self._cached_csr_data: Optional[np.ndarray] = None
        self._data_buffer: Optional[np.ndarray] = None
        self._assume_fixed_pattern = False
        self.reset()

    def _set_cached_slot_positions(self, slot: np.ndarray) -> None:
        self._cached_slot_positions = slot
        self._cached_chunk_slices = self._current_chunk_slices()
        if slot.size == 0:
            self._cached_unique_slot_mask = np.array([], dtype=bool)
            self._cached_duplicate_slot_mask = np.array([], dtype=bool)
            self._cached_unique_slots = np.array([], dtype=np.int64)
            self._cached_unique_data_positions = np.array([], dtype=np.int64)
            self._cached_duplicate_slots = np.array([], dtype=np.int64)
            self._cached_duplicate_data_positions = np.array([], dtype=np.int64)
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
            self._row_chunks.append(rows)
            self._col_chunks.append(cols)
            self._data_chunks.append(values)
            return
        self._row_chunks.append(rows[mask].astype(np.int32, copy=False))
        self._col_chunks.append(cols[mask].astype(np.int32, copy=False))
        self._data_chunks.append(values[mask].astype(np.float64, copy=False))

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
        values.fill(0.0)
        scalar_count = len(self.data)
        if scalar_count:
            scalar_values = np.asarray(self.data, dtype=np.float64)
            np.add.at(values, self._cached_slot_positions[:scalar_count], scalar_values)
        chunk_slices = self._cached_chunk_slices
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
                self._row_chunks.append(rows)
                self._col_chunks.append(cols)
                self._data_chunks.append(values)
                return
            self._row_chunks.append(rows[mask].astype(np.int32, copy=False))
            self._col_chunks.append(cols[mask].astype(np.int32, copy=False))
            self._data_chunks.append(values[mask].astype(np.float64, copy=False))

    def to_csr(self):
        if self._assume_fixed_pattern and self._cached_slot_positions is not None and self._cached_pattern_indptr is not None and self._cached_pattern_indices is not None:
            if self._cached_csr_data is None or self._cached_csr_data.size != self._cached_pattern_linear.size:
                self._cached_csr_data = np.zeros(self._cached_pattern_linear.size, dtype=np.float64)
            values = self._cached_csr_data
            self._refresh_fixed_pattern_values(values)
            return SP_CSR_MATRIX(
                (values, self._cached_pattern_indices, self._cached_pattern_indptr),
                shape=self.shape,
                copy=False,
            )
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
        if self._assume_fixed_pattern:
            n_cols = int(self.shape[1])
            linear = rows.astype(np.int64, copy=False) * n_cols + cols.astype(np.int64, copy=False)
            slot = np.searchsorted(pattern_linear, linear)
            if slot.size and int(slot.max()) < pattern_linear.size and np.array_equal(pattern_linear[slot], linear):
                self._set_cached_slot_positions(slot)
                self._cached_csr_data = csr.data
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


def _normal_equation_structural_pattern(H):
    """Return the structural pattern implied by H.T @ H, retaining zero entries."""
    if not is_sparse_matrix(H):
        return None
    H_csc = H if getattr(H, "format", None) == "csc" else H.tocsc()
    digest = hashlib.blake2b(
        H_csc.indptr.tobytes() + H_csc.indices.tobytes(),
        digest_size=16,
    ).digest()
    key = (H_csc.shape, int(H_csc.nnz), digest)
    cached = _NORMAL_EQUATION_PATTERN_CACHE.get(key)
    if cached is not None:
        return cached
    H_pattern = H_csc.copy()
    H_pattern.data = np.ones(int(H_pattern.nnz), dtype=np.float64)
    pattern = (H_pattern.T @ H_pattern).tocsc()
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
    state_labels: Sequence[str],
    normal_matrix: np.ndarray = None,
    normal_factor_diag: np.ndarray = None,
    dense_svd_limit: int = 2000,
) -> Tuple[int, int, np.ndarray, List[Tuple[str, float]]]:
    """Fast rank analysis for tall state-estimation Jacobians.

    Well-conditioned observable systems are the common path; a Cholesky factor of
    a positive-definite normal matrix proves full column rank without computing a
    full spectral decomposition. `estimate()` passes its already assembled
    H.T @ W @ H here, which has the same rank test outcome when measurement
    weights are positive. Full SVD is kept as the conservative fallback for
    deficient or numerically marginal cases so weak-state reporting remains
    available.
    """
    state_count = len(state_labels)
    measurement_count = int(H.shape[0])
    min_deficiency = max(0, state_count - measurement_count)
    full_column_rank_possible = measurement_count >= state_count
    gram = H.T @ H if normal_matrix is None else normal_matrix
    weak_states: List[Tuple[str, float]] = []

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
            weak_states = [(state_labels[int(idx)], float(diag_values[int(idx)])) for idx in weak_idx]
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
        weak_states = [(state_labels[idx], float(scores[idx])) for idx in top]

    return rank, max(0, deficiency), singular_values, weak_states


def unanchored_angle_state_labels(
    H: np.ndarray,
    state_labels: Sequence[str],
    angle_prefix: str,
    tol: float = 1e-12,
) -> List[str]:
    """Find AC angle components whose measurement rows only define relative angles.

    Large sparse cases can have a tiny rank deficiency even though every node has
    local P/Q rows. In that situation the weak-state diagonal heuristic often
    points at arbitrary low-diagonal columns. This structural pass looks only at
    angle columns of H: rows with zero coefficient sum tie angles relatively,
    while rows with a nonzero sum provide an absolute anchor through a reference
    angle or a direct angle measurement.
    """
    angle_cols = [idx for idx, label in enumerate(state_labels) if label.startswith(angle_prefix)]
    if not angle_cols:
        return []

    if is_sparse_matrix(H):
        sub = H[:, angle_cols].tocsr()
        degree = np.bincount(sub.indices, minlength=len(angle_cols)) if sub.indices.size else np.zeros(len(angle_cols), dtype=np.int64)
    else:
        matrix = np.asarray(H)
        sub = matrix[:, angle_cols]
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

    if is_sparse_matrix(sub):
        for row in range(sub.shape[0]):
            start = int(sub.indptr[row])
            end = int(sub.indptr[row + 1])
            if start == end:
                continue
            cols = sub.indices[start:end]
            vals = sub.data[start:end]
            root = int(cols[0])
            for col in cols[1:]:
                root = union(root, int(col))
            root = find(root)
            if abs(float(np.sum(vals))) > tol:
                anchored[root] = True
    else:
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

    labels = []
    for root, local_cols in components.items():
        root = find(root)
        if anchored[root]:
            continue
        representative = max(local_cols, key=lambda col: (int(degree[col]), -int(angle_cols[col])))
        labels.append((len(local_cols), state_labels[int(angle_cols[representative])]))
    labels.sort(key=lambda item: (-item[0], item[1]))
    return [label for _, label in labels]


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
) -> Tuple[np.ndarray, np.ndarray]:
    """Build WLS normal equations while avoiding WH allocation for uniform weights."""
    if is_sparse_matrix(H):
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
            weighted_H = H.multiply(weight[:, None])
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
                weighted_H = H.multiply(weight[:, None])
                gain = H.T @ weighted_H
                rhs = H.T @ weighted_residual
        if is_sparse_matrix(gain):
            if assume_normal_pattern_matches:
                gain = gain.tocsc() if getattr(gain, "format", None) != "csc" else gain
            else:
                if normal_pattern is None:
                    normal_pattern = _normal_equation_structural_pattern(H)
                gain = _expand_sparse_matrix_to_pattern(gain, normal_pattern)
            gain = gain.toarray() if gain.shape[0] <= dense_gain_limit else gain.tocsc()
        return gain, np.asarray(rhs, dtype=np.float64).ravel()

    if weight.size == 0:
        gain = H.T @ H
        rhs = H.T @ residual
        return gain, rhs

    if weights_are_uniform is True or uniform_weight is not None:
        if uniform_weight is None:
            uniform_weight = float(weight[0])
        gain = H.T @ H
        rhs = H.T @ residual
        if uniform_weight != 1.0:
            gain = uniform_weight * gain
            rhs = uniform_weight * rhs
        return gain, rhs

    if weights_are_uniform is False:
        weighted_residual = weight * residual if weighted_residual is None else weighted_residual
        WH = weight[:, None] * H
        gain = H.T @ WH
        rhs = H.T @ weighted_residual
        return gain, rhs

    first_weight = float(weight[0])
    if np.all(weight == first_weight):
        gain = H.T @ H
        rhs = H.T @ residual
        if first_weight != 1.0:
            gain = first_weight * gain
            rhs = first_weight * rhs
        return gain, rhs

    WH = weight[:, None] * H
    gain = H.T @ WH
    weighted_residual = weight * residual if weighted_residual is None else weighted_residual
    rhs = H.T @ weighted_residual
    return gain, rhs
