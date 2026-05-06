import hashlib
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.sparse import coo_matrix as SP_COO_MATRIX
    from scipy.sparse import csc_matrix as SP_CSC_MATRIX
    from scipy.sparse import issparse as SP_ISSPARSE
except Exception:
    SP_COO_MATRIX = None
    SP_CSC_MATRIX = None
    SP_ISSPARSE = None

try:
    from scipy.sparse.csgraph import structural_rank as SP_STRUCTURAL_RANK
except Exception:
    SP_STRUCTURAL_RANK = None

try:
    from scipy.sparse.linalg import splu as SP_SPLU
    from scipy.sparse.linalg import spsolve as SP_SPSOLVE
except Exception:
    SP_SPLU = None
    SP_SPSOLVE = None

try:
    from sksparse.cholmod import cholesky as CHOLMOD_CHOLESKY
    from sksparse.cholmod import analyze as CHOLMOD_ANALYZE
except Exception:
    CHOLMOD_CHOLESKY = None
    CHOLMOD_ANALYZE = None

try:
    from scipy.linalg.lapack import dposv as DPOSV
    from scipy.linalg.lapack import dpotrf as DPOTRF
except Exception:
    DPOSV = None
    DPOTRF = None

try:
    from scipy.linalg import cho_factor as CHO_FACTOR
    from scipy.linalg import cho_solve as CHO_SOLVE
except Exception:
    CHO_FACTOR = None
    CHO_SOLVE = None


ANGLE_MEASUREMENT_TYPES = frozenset(("ANGLE", "THETA", "ANGLE_DIFF", "THETA_DIFF"))
_NORMAL_EQUATION_PATTERN_CACHE = {}
_SPARSE_PATTERN_EXPANSION_CACHE = {}


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
        self.rows: List[int] = []
        self.cols: List[int] = []
        self.data: List[float] = []
        self._row_chunks: List[np.ndarray] = []
        self._col_chunks: List[np.ndarray] = []
        self._data_chunks: List[np.ndarray] = []

    def _append_arrays(self, rows: np.ndarray, cols: np.ndarray, values: np.ndarray) -> None:
        """Keep vectorized writes as NumPy chunks until final COO/CSR construction."""
        if rows.size == 0:
            return
        rows = np.asarray(rows, dtype=np.int32)
        cols = np.asarray(cols, dtype=np.int32)
        values = np.asarray(values, dtype=np.float64)
        mask = cols >= 0
        if not np.any(mask):
            return
        self._row_chunks.append(rows[mask].astype(np.int32, copy=False))
        self._col_chunks.append(cols[mask].astype(np.int32, copy=False))
        self._data_chunks.append(values[mask].astype(np.float64, copy=False))

    def _coo_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        row_parts = []
        col_parts = []
        data_parts = []
        if self.rows:
            row_parts.append(np.asarray(self.rows, dtype=np.int32))
            col_parts.append(np.asarray(self.cols, dtype=np.int32))
            data_parts.append(np.asarray(self.data, dtype=np.float64))
        row_parts.extend(self._row_chunks)
        col_parts.extend(self._col_chunks)
        data_parts.extend(self._data_chunks)
        if not row_parts:
            return (
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
                np.array([], dtype=np.float64),
            )
        if len(row_parts) == 1:
            return row_parts[0], col_parts[0], data_parts[0]
        return np.concatenate(row_parts), np.concatenate(col_parts), np.concatenate(data_parts)

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
        broadcast_rows, broadcast_cols = np.broadcast_arrays(rows, cols)
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
        rows = np.asarray(rows, dtype=np.int32)
        cols = np.asarray(cols, dtype=np.int32)
        values = np.asarray(values, dtype=np.float64)
        if mask is None:
            mask = cols >= 0
        else:
            mask = np.asarray(mask, dtype=bool) & (cols >= 0)
        if np.any(mask):
            self._row_chunks.append(rows[mask].astype(np.int32, copy=False))
            self._col_chunks.append(cols[mask].astype(np.int32, copy=False))
            self._data_chunks.append(values[mask].astype(np.float64, copy=False))

    def to_csr(self):
        rows, cols, data = self._coo_arrays()
        if SP_COO_MATRIX is None:
            dense = np.zeros(self.shape, dtype=np.float64)
            if rows.size:
                np.add.at(dense, (rows, cols), data)
            return dense
        return SP_COO_MATRIX((data, (rows, cols)), shape=self.shape).tocsr()


def is_sparse_matrix(matrix) -> bool:
    return bool(SP_ISSPARSE is not None and SP_ISSPARSE(matrix))


def sparse_structural_rank(matrix) -> Optional[int]:
    """Return sparse structural rank when SciPy provides the graph matcher."""
    if SP_STRUCTURAL_RANK is None or not is_sparse_matrix(matrix):
        return None
    return int(SP_STRUCTURAL_RANK(matrix))


def matrix_is_empty(matrix) -> bool:
    return matrix.shape[0] == 0 or matrix.shape[1] == 0


def _normal_equation_structural_pattern(H):
    """Return the structural pattern implied by H.T @ H, retaining zero entries."""
    if SP_COO_MATRIX is None or not is_sparse_matrix(H):
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
    if pattern is None or SP_COO_MATRIX is None or SP_CSC_MATRIX is None:
        return matrix
    matrix_csc = matrix if getattr(matrix, "format", None) == "csc" else matrix.tocsc()
    pattern_csc = pattern if getattr(pattern, "format", None) == "csc" else pattern.tocsc()
    if (
        matrix_csc.shape == pattern_csc.shape
        and int(matrix_csc.nnz) == int(pattern_csc.nnz)
        and np.array_equal(matrix_csc.indptr, pattern_csc.indptr)
        and np.array_equal(matrix_csc.indices, pattern_csc.indices)
    ):
        return matrix_csc

    matrix_digest = hashlib.blake2b(
        matrix_csc.indptr.tobytes() + matrix_csc.indices.tobytes(),
        digest_size=16,
    ).digest()
    key = (
        id(pattern_csc),
        int(matrix_csc.nnz),
        matrix_digest,
    )
    target_positions = _SPARSE_PATTERN_EXPANSION_CACHE.get(key)
    if target_positions is None:
        positions = np.empty(int(matrix_csc.nnz), dtype=np.int64)
        write_pos = 0
        for col in range(int(matrix_csc.shape[1])):
            matrix_start = int(matrix_csc.indptr[col])
            matrix_end = int(matrix_csc.indptr[col + 1])
            if matrix_start == matrix_end:
                continue
            pattern_start = int(pattern_csc.indptr[col])
            pattern_end = int(pattern_csc.indptr[col + 1])
            pattern_indices = pattern_csc.indices[pattern_start:pattern_end]
            matrix_indices = matrix_csc.indices[matrix_start:matrix_end]
            local_positions = np.searchsorted(pattern_indices, matrix_indices)
            positions[write_pos : write_pos + matrix_indices.size] = pattern_start + local_positions
            write_pos += matrix_indices.size
        target_positions = positions
        if len(_SPARSE_PATTERN_EXPANSION_CACHE) > 32:
            _SPARSE_PATTERN_EXPANSION_CACHE.clear()
        _SPARSE_PATTERN_EXPANSION_CACHE[key] = target_positions

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
    gram = H.T @ H if normal_matrix is None else normal_matrix
    weak_states: List[Tuple[str, float]] = []

    if is_sparse_matrix(gram):
        diag_values = gram.diagonal()
    else:
        diag_values = np.diag(gram) if gram.size else np.array([], dtype=np.float64)
    scale = float(np.sqrt(np.max(diag_values))) if diag_values.size else 1.0
    tol = max(H.shape) * np.finfo(float).eps * max(scale, 1.0)
    if normal_factor_diag is not None:
        diag = np.asarray(normal_factor_diag, dtype=np.float64)
        if diag.size == state_count and float(np.min(diag)) > tol:
            return state_count, 0, np.array([], dtype=np.float64), weak_states
    elif is_sparse_matrix(gram) and SP_SPLU is not None:
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
    elif DPOTRF is not None:
        chol, info = DPOTRF(gram, lower=True, clean=False, overwrite_a=False)
        if info == 0:
            diag = np.diag(chol)
            if diag.size == state_count and float(np.min(diag)) > tol:
                return state_count, 0, np.array([], dtype=np.float64), weak_states
    else:
        try:
            chol = np.linalg.cholesky(gram)
            diag = np.diag(chol)
            if diag.size == state_count and float(np.min(diag)) > tol:
                return state_count, 0, np.array([], dtype=np.float64), weak_states
        except np.linalg.LinAlgError:
            pass

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
            deficiency = max(1, near_zero)
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


def solve_normal_equations(gain: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve a WLS normal equation, preferring Cholesky for positive definite gain."""
    dx, _ = solve_normal_equations_with_factor(gain, rhs)
    return dx


class NormalEquationSolver:
    """Solve repeated normal equations, reusing CHOLMOD symbolic analysis when available."""

    def __init__(self):
        self._cholmod_factor = None
        self._cholmod_pattern = None
        self._cholmod_disabled = False

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
            if not self._same_sparse_pattern(self._cholmod_pattern, gain_csc):
                self._cholmod_factor = CHOLMOD_ANALYZE(gain_csc)
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
    if SP_SPLU is not None:
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
    if SP_SPSOLVE is not None:
        try:
            return SP_SPSOLVE(gain_csc, rhs), None
        except Exception:
            pass
    return _solve_dense_normal_equations(gain_csc.toarray(), rhs, return_factor_diag)


def _solve_dense_normal_equations(
    gain: np.ndarray,
    rhs: np.ndarray,
    return_factor_diag: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    if DPOSV is not None:
        try:
            factor, dx, info = DPOSV(gain, rhs.copy(), lower=True, overwrite_a=False, overwrite_b=True)
            if info == 0:
                factor_diag = np.diag(factor).copy() if return_factor_diag else None
                return dx, factor_diag
        except Exception:
            pass
    if CHO_FACTOR is not None and CHO_SOLVE is not None:
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
            gain = _expand_sparse_matrix_to_pattern(gain, _normal_equation_structural_pattern(H))
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
