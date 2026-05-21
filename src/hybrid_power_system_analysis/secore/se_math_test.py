from pathlib import Path
import argparse

import numpy as np
import scipy.sparse as sp

try:
    from sksparse.cholmod import analyze as CHOLMOD_ANALYZE
    from sksparse.cholmod import cholesky as CHOLMOD_CHOLESKY
except Exception:
    CHOLMOD_ANALYZE = None
    CHOLMOD_CHOLESKY = None


class SEMathTest:
    """Sparse matrix dump readers and CHOLMOD helpers for SE math tests."""

    def _read_sparse_triplet_dump(self, directory, filename):
        """Read a 1-based sparse triplet dump written by AC SE matrix debug output."""
        path = Path(directory) / filename
        if not path.exists():
            raise FileNotFoundError(path)

        shape = None
        nnz = None
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("# shape"):
                    parts = stripped.split()
                    if len(parts) < 4:
                        raise ValueError(f"Invalid shape header in {path}: {stripped!r}")
                    shape = (int(parts[2]), int(parts[3]))
                    if len(parts) >= 6 and parts[4] == "nnz":
                        nnz = int(parts[5])
                    break
        if shape is None:
            raise ValueError(f"Missing '# shape <rows> <cols> nnz <n>' header in {path}")

        empty_i = np.array([], dtype=np.int64)
        empty_x = np.array([], dtype=np.float64)
        if nnz == 0:
            return shape, empty_i, empty_i, empty_x

        data = np.loadtxt(path, comments="#", dtype=np.float64, ndmin=2)
        if data.size == 0:
            return shape, empty_i, empty_i, empty_x
        if data.shape[1] < 3:
            raise ValueError(f"Expected triplet rows 'row col value' in {path}")

        rows = data[:, 0].astype(np.int64, copy=False) - 1
        cols = data[:, 1].astype(np.int64, copy=False) - 1
        values = data[:, 2].astype(np.float64, copy=False)
        if rows.size and (
            int(rows.min()) < 0
            or int(cols.min()) < 0
            or int(rows.max()) >= shape[0]
            or int(cols.max()) >= shape[1]
        ):
            raise ValueError(f"Triplet index out of bounds for shape {shape} in {path}")
        return shape, rows, cols, values

    def read_j_matrix(self, directory, filename):
        """Read a dumped Jacobian file such as ``j1.txt`` and return a CSC matrix."""
        shape, rows, cols, values = self._read_sparse_triplet_dump(directory, filename)
        return sp.csc_matrix((values, (rows, cols)), shape=shape)

    def read_d_vector(self, directory, filename):
        """Read a dumped diagonal D file such as ``d1.txt`` and return its diagonal vector."""
        shape, rows, cols, values = self._read_sparse_triplet_dump(directory, filename)
        if shape[0] != shape[1]:
            raise ValueError(f"D dump must be square, got shape {shape}")
        if rows.size and not np.array_equal(rows, cols):
            raise ValueError("D dump contains non-diagonal entries")
        diag = np.zeros(shape[0], dtype=np.float64)
        if rows.size:
            diag[rows] = values
        return diag

    def read_h_matrix(self, directory, filename):
        """Read a dumped information matrix file such as ``h1.txt`` and return a CSC matrix."""
        shape, rows, cols, values = self._read_sparse_triplet_dump(directory, filename)
        return sp.csc_matrix((values, (rows, cols)), shape=shape)

    def read_file(
        self,
        directory,
        *,
        iterations=13,
        start_index=1,
        j_template="j{index}.txt",
        d_template="d{index}.txt",
        h_template="h{index}.txt",
    ):
        """Read all J/D/H dump files into memory before numerical processing.

        ``h_list`` keeps the same length as ``j_list`` and ``d_list``. If an H
        dump is not present for an iteration, the corresponding entry is
        ``None`` because the solver paths rebuild/update H from J and D.
        """
        base_dir = Path(directory)
        j_list = []
        d_list = []
        h_list = []
        for offset in range(int(iterations)):
            index = int(start_index) + offset
            j_name = j_template.format(index=index)
            d_name = d_template.format(index=index)
            h_name = h_template.format(index=index) if h_template is not None else None
            j_list.append(self.read_j_matrix(base_dir, j_name))
            d_list.append(self.read_d_vector(base_dir, d_name))
            if h_name is not None and (base_dir / h_name).exists():
                h_list.append(self.read_h_matrix(base_dir, h_name))
            else:
                h_list.append(None)
        return j_list, d_list, h_list

    def load_matrix_lists(
        self,
        directory,
        *,
        iterations=13,
        start_index=1,
        j_template="j{index}.txt",
        d_template="d{index}.txt",
        h_template="h{index}.txt",
    ):
        """Compatibility wrapper for ``read_file``."""
        return self.read_file(
            directory,
            iterations=iterations,
            start_index=start_index,
            j_template=j_template,
            d_template=d_template,
            h_template=h_template,
        )

    def _matrix_lists_or_read(
        self,
        directory,
        matrix_lists,
        *,
        iterations=13,
        start_index=1,
        j_template="j{index}.txt",
        d_template="d{index}.txt",
        h_template="h{index}.txt",
    ):
        if matrix_lists is not None:
            if len(matrix_lists) != 3:
                raise ValueError("matrix_lists must be a (j_list, d_list, h_list) tuple")
            return matrix_lists
        if directory is None:
            raise ValueError("directory is required when matrix_lists is not provided")
        return self.read_file(
            directory,
            iterations=iterations,
            start_index=start_index,
            j_template=j_template,
            d_template=d_template,
            h_template=h_template,
        )

    def update_csc_matrix_from_file(self, matrix, directory, filename):
        """Parse a sparse triplet dump and update a preallocated CSC matrix in place.

        The matrix shape and sparsity pattern must already contain every triplet in
        the file. Entries not present in the file are reset to zero.
        """
        if not sp.isspmatrix_csc(matrix):
            raise TypeError("update_csc_matrix_from_file requires a scipy.sparse.csc_matrix")
        shape, rows, cols, values = self._read_sparse_triplet_dump(directory, filename)
        if tuple(matrix.shape) != tuple(shape):
            raise ValueError(f"CSC matrix shape {matrix.shape} does not match dump shape {shape}")
        if not matrix.has_sorted_indices:
            matrix.sort_indices()

        matrix.data.fill(0.0)
        if values.size == 0:
            return matrix

        n_rows = int(matrix.shape[0])
        pattern_counts = np.diff(matrix.indptr).astype(np.int64, copy=False)
        pattern_cols = np.repeat(np.arange(int(matrix.shape[1]), dtype=np.int64), pattern_counts)
        pattern_linear = pattern_cols * np.int64(n_rows) + matrix.indices.astype(np.int64, copy=False)
        file_linear = cols.astype(np.int64, copy=False) * np.int64(n_rows) + rows.astype(np.int64, copy=False)
        target_pos = np.searchsorted(pattern_linear, file_linear)
        valid = target_pos < pattern_linear.size
        if valid.size:
            valid[valid] &= pattern_linear[target_pos[valid]] == file_linear[valid]
        if not np.all(valid):
            first = int(np.flatnonzero(~valid)[0])
            raise ValueError(
                f"Dump entry row={int(rows[first]) + 1}, col={int(cols[first]) + 1} "
                "is not present in the preallocated CSC pattern"
            )

        if np.unique(target_pos).size == target_pos.size:
            matrix.data[target_pos] = values
        else:
            np.add.at(matrix.data, target_pos, values)
        return matrix

    def _as_square_csc_for_cholmod(self, matrix):
        matrix_csc = matrix if sp.isspmatrix_csc(matrix) else sp.csc_matrix(matrix)
        if matrix_csc.shape[0] != matrix_csc.shape[1]:
            raise ValueError(f"CHOLMOD requires a square matrix, got shape {matrix_csc.shape}")
        if not matrix_csc.has_sorted_indices:
            matrix_csc.sort_indices()
        return matrix_csc

    def cholmod_analyze(self, matrix):
        """Run CHOLMOD symbolic analysis once for a reusable symmetric pattern.

        CHOLMOD consumes the lower triangle of a symmetric sparse matrix, so the
        H matrices produced by ``get_H_sparsity`` and ``symbolic_AtBA_lower`` can
        be passed directly without materializing the upper triangle.
        """
        if CHOLMOD_ANALYZE is None:
            raise RuntimeError("sksparse.cholmod analyze is not available")
        return CHOLMOD_ANALYZE(self._as_square_csc_for_cholmod(matrix))

    def cholmod_factorize_inplace(self, factor, matrix):
        """Refresh numeric CHOLMOD factorization on a lower-triangle H matrix."""
        factor.cholesky_inplace(self._as_square_csc_for_cholmod(matrix))
        return factor

    def cholmod_factorize(self, matrix):
        """Factorize a symmetric matrix once with CHOLMOD.

        ``matrix`` may store only the lower triangle.
        """
        if CHOLMOD_CHOLESKY is None:
            raise RuntimeError("sksparse.cholmod is not available")
        return CHOLMOD_CHOLESKY(self._as_square_csc_for_cholmod(matrix))

    def cholmod_solve(self, factor, rhs):
        """Solve ``A x = b`` using a reusable CHOLMOD factor."""
        rhs_array = np.asarray(rhs, dtype=np.float64)
        return np.asarray(factor(rhs_array), dtype=np.float64)

    def solve_cholmod(self, matrix, rhs):
        """Factorize ``A`` with CHOLMOD and solve ``A x = b`` in one call."""
        return self.cholmod_solve(self.cholmod_factorize(matrix), rhs)

    def solve(self, factor, rhs):
        """Solve ``A x = b`` with an already factorized CHOLMOD factor."""
        return self.cholmod_solve(factor, rhs)

    def get_H_sparsity(self, matrix):
        matrix_csc = matrix if sp.isspmatrix_csc(matrix) else sp.csc_matrix(matrix)
        matrix_pattern = matrix_csc.copy()
        matrix_pattern.data = np.ones_like(matrix_pattern.data)
        h_full = matrix_pattern.T @ matrix_pattern
        h_pattern = sp.tril(h_full, format="csc")
        h_pattern.data = np.ones_like(h_pattern.data)
        h_pattern.sort_indices()
        return h_pattern

    def update_H_data(self, h_matrix, matrix, d_vector):
        """Refresh ``h_matrix.data`` for ``tril(matrix.T @ diag(d_vector) @ matrix)``."""
        if not sp.isspmatrix_csc(h_matrix):
            raise TypeError("update_H_data requires a scipy.sparse.csc_matrix H pattern")
        matrix_csc = matrix if sp.isspmatrix_csc(matrix) else sp.csc_matrix(matrix)
        d_array = np.asarray(d_vector, dtype=np.float64)
        if d_array.ndim != 1 or d_array.shape[0] != matrix_csc.shape[0]:
            raise ValueError(
                f"D vector length {d_array.shape} does not match J rows {matrix_csc.shape[0]}"
            )
        if h_matrix.shape != (matrix_csc.shape[1], matrix_csc.shape[1]):
            raise ValueError(
                f"H shape {h_matrix.shape} does not match J column count {matrix_csc.shape[1]}"
            )
        if not h_matrix.has_sorted_indices:
            h_matrix.sort_indices()

        weighted_matrix = matrix_csc.multiply(d_array[:, None])
        refreshed = sp.tril(matrix_csc.T @ weighted_matrix, format="csc")
        refreshed.sort_indices()

        h_matrix.data.fill(0.0)
        if refreshed.nnz == 0:
            return h_matrix

        n_rows = int(h_matrix.shape[0])
        pattern_counts = np.diff(h_matrix.indptr).astype(np.int64, copy=False)
        pattern_cols = np.repeat(np.arange(int(h_matrix.shape[1]), dtype=np.int64), pattern_counts)
        pattern_linear = pattern_cols * np.int64(n_rows) + h_matrix.indices.astype(np.int64, copy=False)
        refreshed_counts = np.diff(refreshed.indptr).astype(np.int64, copy=False)
        refreshed_cols = np.repeat(np.arange(int(refreshed.shape[1]), dtype=np.int64), refreshed_counts)
        refreshed_linear = refreshed_cols * np.int64(n_rows) + refreshed.indices.astype(np.int64, copy=False)
        target_pos = np.searchsorted(pattern_linear, refreshed_linear)
        valid = target_pos < pattern_linear.size
        if valid.size:
            valid[valid] &= pattern_linear[target_pos[valid]] == refreshed_linear[valid]
        if not np.all(valid):
            first = int(np.flatnonzero(~valid)[0])
            raise ValueError(
                f"Refreshed H entry row={int(refreshed.indices[first])}, col={int(refreshed_cols[first])} "
                "is not present in the preallocated H pattern"
            )
        h_matrix.data[target_pos] = refreshed.data
        return h_matrix

    def update_H_data_simple(self, A, B_diag, H):
        """Refresh H with a simple column-intersection loop for ``A.T @ diag(B) @ A``."""
        if not sp.isspmatrix_csc(A):
            A = sp.csc_matrix(A)
        if not sp.isspmatrix_csc(H):
            raise TypeError("update_H_data_simple requires a scipy.sparse.csc_matrix H pattern")
        B_diag = np.asarray(B_diag, dtype=np.float64)
        if B_diag.ndim != 1 or B_diag.shape[0] != A.shape[0]:
            raise ValueError(f"B_diag length {B_diag.shape} does not match A rows {A.shape[0]}")
        if H.shape != (A.shape[1], A.shape[1]):
            raise ValueError(f"H shape {H.shape} does not match A column count {A.shape[1]}")
        if not A.has_sorted_indices:
            A.sort_indices()
        if not H.has_sorted_indices:
            H.sort_indices()

        H.data.fill(0.0)
        ptr = A.indptr
        indices = A.indices
        data = A.data
        h_ptr = H.indptr
        h_indices = H.indices
        h_data = H.data

        for j in range(A.shape[1]):
            js = h_ptr[j]
            je = h_ptr[j + 1]
            if js == je:
                continue
            rows = h_indices[js:je]
            Aj_data = data[ptr[j]:ptr[j + 1]]
            Aj_idx = indices[ptr[j]:ptr[j + 1]]
            for idx, i in enumerate(rows):
                Ai_data = data[ptr[i]:ptr[i + 1]]
                Ai_idx = indices[ptr[i]:ptr[i + 1]]
                mask = np.isin(Ai_idx, Aj_idx, assume_unique=True)
                common = Ai_idx[mask]
                val = np.sum(
                    Ai_data[mask]
                    * B_diag[common]
                    * Aj_data[np.isin(Aj_idx, common, assume_unique=True)]
                )
                h_data[js + idx] = val
        return H

    def symbolic_H_data_simple_pattern_plan(self, A):
        """Build lower H pattern and its planned data-refresh contributions in one pass."""
        if not sp.isspmatrix_csc(A):
            A = sp.csc_matrix(A)
        if not A.has_sorted_indices:
            A.sort_indices()

        m, n = A.shape
        col_of = np.repeat(np.arange(n, dtype=np.int64), np.diff(A.indptr).astype(np.int64, copy=False))
        row_of = A.indices.astype(np.int64, copy=False)
        data_pos = np.arange(A.data.size, dtype=np.int64)
        order = np.lexsort((col_of, row_of))
        sorted_cols = col_of[order]
        sorted_data_pos = data_pos[order]
        row_counts = np.bincount(row_of, minlength=m).astype(np.int64, copy=False)
        row_start = np.empty(m + 1, dtype=np.int64)
        row_start[0] = 0
        np.cumsum(row_counts, out=row_start[1:])
        pair_counts = (row_counts * (row_counts + 1)) // 2
        total_pairs = int(pair_counts.sum())

        if total_pairs == 0:
            h_matrix = sp.csc_matrix((n, n), dtype=A.dtype)
            return h_matrix, {
                "entry_ptr": np.zeros(1, dtype=np.int64),
                "K": np.zeros(0, dtype=np.int64),
                "Pi": np.zeros(0, dtype=np.int64),
                "Pj": np.zeros(0, dtype=np.int64),
                "A_shape": tuple(int(item) for item in A.shape),
                "H_shape": tuple(int(item) for item in h_matrix.shape),
                "H_nnz": int(h_matrix.nnz),
                "all_entries_nonempty": True,
            }

        row_i_all = np.empty(total_pairs, dtype=np.int64)
        col_j_all = np.empty(total_pairs, dtype=np.int64)
        k_all = np.empty(total_pairs, dtype=np.int64)
        pi_all = np.empty(total_pairs, dtype=np.int64)
        pj_all = np.empty(total_pairs, dtype=np.int64)
        tril_cache = {}
        offset = 0
        nonzero_rows = np.flatnonzero(row_counts)
        width_order = np.argsort(row_counts[nonzero_rows], kind="stable")
        rows_by_width = nonzero_rows[width_order]
        widths_by_row = row_counts[rows_by_width]
        width_breaks = np.concatenate(
            (np.array([0], dtype=np.int64), np.nonzero(widths_by_row[1:] != widths_by_row[:-1])[0] + 1)
        )
        width_breaks = np.concatenate((width_breaks, np.array([rows_by_width.size], dtype=np.int64)))
        for group_start, group_stop in zip(width_breaks[:-1], width_breaks[1:]):
            rows_for_width = rows_by_width[int(group_start) : int(group_stop)]
            width = int(row_counts[rows_for_width[0]])
            per_row_pairs = (width * (width + 1)) // 2
            out_stop = offset + int(rows_for_width.size) * per_row_pairs
            if width == 1:
                source = row_start[rows_for_width]
                row_i_all[offset:out_stop] = sorted_cols[source]
                col_j_all[offset:out_stop] = sorted_cols[source]
                k_all[offset:out_stop] = rows_for_width
                pi_all[offset:out_stop] = sorted_data_pos[source]
                pj_all[offset:out_stop] = sorted_data_pos[source]
            else:
                left, right = tril_cache.get(width, (None, None))
                if left is None:
                    left, right = np.tril_indices(width)
                    left = left.astype(np.int64, copy=False)
                    right = right.astype(np.int64, copy=False)
                    tril_cache[width] = (left, right)
                starts_for_width = row_start[rows_for_width]
                source_left = (starts_for_width[:, None] + left[None, :]).ravel()
                source_right = (starts_for_width[:, None] + right[None, :]).ravel()
                row_i_all[offset:out_stop] = sorted_cols[source_left]
                col_j_all[offset:out_stop] = sorted_cols[source_right]
                k_all[offset:out_stop].reshape(rows_for_width.size, per_row_pairs)[:] = rows_for_width[:, None]
                pi_all[offset:out_stop] = sorted_data_pos[source_left]
                pj_all[offset:out_stop] = sorted_data_pos[source_right]
            offset = out_stop

        order_h = np.lexsort((row_i_all, col_j_all))
        row_i_all = row_i_all[order_h]
        col_j_all = col_j_all[order_h]
        k_all = k_all[order_h]
        pi_all = pi_all[order_h]
        pj_all = pj_all[order_h]

        is_new = np.empty(row_i_all.size, dtype=bool)
        is_new[0] = True
        is_new[1:] = (row_i_all[1:] != row_i_all[:-1]) | (col_j_all[1:] != col_j_all[:-1])
        starts = np.nonzero(is_new)[0]
        entry_ptr = np.empty(starts.size + 1, dtype=np.int64)
        entry_ptr[:-1] = starts
        entry_ptr[-1] = row_i_all.size

        indices = row_i_all[starts]
        cols_unique = col_j_all[starts]
        indptr = np.zeros(n + 1, dtype=np.int64)
        np.add.at(indptr, cols_unique + 1, 1)
        np.cumsum(indptr, out=indptr)
        h_matrix = sp.csc_matrix(
            (np.zeros(indices.size, dtype=A.dtype), indices, indptr),
            shape=(n, n),
        )
        return h_matrix, {
            "entry_ptr": entry_ptr,
            "K": k_all,
            "Pi": pi_all,
            "Pj": pj_all,
            "A_shape": tuple(int(item) for item in A.shape),
            "H_shape": tuple(int(item) for item in h_matrix.shape),
            "H_nnz": int(h_matrix.nnz),
            "all_entries_nonempty": True,
        }

    def symbolic_H_data_simple_plan(self, A, H):
        """Precompute contribution indexes for refreshing a fixed H pattern.

        The plan maps each contribution in ``A.T @ diag(B) @ A`` to a position
        in ``H.data``.  It is built once for fixed A/H sparsity and reused by
        ``update_H_data_simple_planned`` for different diagonal B values.
        """
        if not sp.isspmatrix_csc(A):
            A = sp.csc_matrix(A)
        if not sp.isspmatrix_csc(H):
            raise TypeError("symbolic_H_data_simple_plan requires a scipy.sparse.csc_matrix H pattern")
        if H.shape != (A.shape[1], A.shape[1]):
            raise ValueError(f"H shape {H.shape} does not match A column count {A.shape[1]}")
        if not A.has_sorted_indices:
            A.sort_indices()
        if not H.has_sorted_indices:
            H.sort_indices()

        m, n = A.shape
        h_counts = np.diff(H.indptr).astype(np.int64, copy=False)
        h_cols = np.repeat(np.arange(n, dtype=np.int64), h_counts)
        h_linear = h_cols * np.int64(n) + H.indices.astype(np.int64, copy=False)

        col_of = np.repeat(np.arange(n, dtype=np.int64), np.diff(A.indptr).astype(np.int64, copy=False))
        row_of = A.indices.astype(np.int64, copy=False)
        data_pos = np.arange(A.data.size, dtype=np.int64)
        order = np.lexsort((col_of, row_of))
        sorted_rows = row_of[order]
        sorted_cols = col_of[order]
        sorted_data_pos = data_pos[order]
        row_start = np.searchsorted(sorted_rows, np.arange(m + 1, dtype=np.int64))

        slot_parts = []
        k_parts = []
        pi_parts = []
        pj_parts = []
        for row in range(m):
            start, end = int(row_start[row]), int(row_start[row + 1])
            width = end - start
            if width == 0:
                continue
            cols_row = sorted_cols[start:end]
            data_pos_row = sorted_data_pos[start:end]
            left, right = np.divmod(np.arange(width * width, dtype=np.int64), width)
            row_i = cols_row[left]
            col_j = cols_row[right]
            lower_mask = row_i >= col_j
            if not np.any(lower_mask):
                continue
            row_i = row_i[lower_mask]
            col_j = col_j[lower_mask]
            linear = col_j * np.int64(n) + row_i
            slot = np.searchsorted(h_linear, linear)
            valid = slot < h_linear.size
            if valid.size:
                valid[valid] &= h_linear[slot[valid]] == linear[valid]
            if not np.all(valid):
                first = int(np.flatnonzero(~valid)[0])
                raise ValueError(
                    f"H pattern is missing structural entry row={int(row_i[first])}, col={int(col_j[first])}"
                )
            slot_parts.append(slot.astype(np.int64, copy=False))
            k_parts.append(np.full(row_i.size, row, dtype=np.int64))
            pi_parts.append(data_pos_row[left][lower_mask].astype(np.int64, copy=False))
            pj_parts.append(data_pos_row[right][lower_mask].astype(np.int64, copy=False))

        if not slot_parts:
            return {
                "entry_ptr": np.zeros(H.nnz + 1, dtype=np.int64),
                "K": np.zeros(0, dtype=np.int64),
                "Pi": np.zeros(0, dtype=np.int64),
                "Pj": np.zeros(0, dtype=np.int64),
                "A_shape": tuple(int(item) for item in A.shape),
                "H_shape": tuple(int(item) for item in H.shape),
                "H_nnz": int(H.nnz),
                "all_entries_nonempty": False,
            }

        slots = np.concatenate(slot_parts)
        k_all = np.concatenate(k_parts)
        pi_all = np.concatenate(pi_parts)
        pj_all = np.concatenate(pj_parts)
        order = np.argsort(slots, kind="stable")
        slots = slots[order]
        k_all = k_all[order]
        pi_all = pi_all[order]
        pj_all = pj_all[order]

        counts = np.bincount(slots, minlength=int(H.nnz)).astype(np.int64, copy=False)
        entry_ptr = np.empty(int(H.nnz) + 1, dtype=np.int64)
        entry_ptr[0] = 0
        np.cumsum(counts, out=entry_ptr[1:])
        return {
            "entry_ptr": entry_ptr,
            "K": k_all,
            "Pi": pi_all,
            "Pj": pj_all,
            "A_shape": tuple(int(item) for item in A.shape),
            "H_shape": tuple(int(item) for item in H.shape),
            "H_nnz": int(H.nnz),
            "all_entries_nonempty": bool(np.all(counts > 0)),
        }

    def update_H_data_simple_planned(self, A, B_diag, H, contrib):
        """Refresh H using a precomputed ``symbolic_H_data_simple_plan``."""
        if not sp.isspmatrix_csc(A):
            A = sp.csc_matrix(A)
        if not sp.isspmatrix_csc(H):
            raise TypeError("update_H_data_simple_planned requires a scipy.sparse.csc_matrix H pattern")
        B_diag = np.asarray(B_diag, dtype=np.float64)
        if B_diag.ndim != 1 or B_diag.shape[0] != A.shape[0]:
            raise ValueError(f"B_diag length {B_diag.shape} does not match A rows {A.shape[0]}")
        if tuple(A.shape) != tuple(contrib.get("A_shape", A.shape)):
            raise ValueError(f"A shape {A.shape} does not match planned shape {contrib.get('A_shape')}")
        if tuple(H.shape) != tuple(contrib.get("H_shape", H.shape)) or int(H.nnz) != int(contrib.get("H_nnz", H.nnz)):
            raise ValueError("H shape or sparsity size does not match planned H pattern")

        entry_ptr = contrib["entry_ptr"]
        starts = entry_ptr[:-1]
        if contrib["K"].size == 0:
            H.data.fill(0.0)
            return H
        values = A.data[contrib["Pi"]] * B_diag[contrib["K"]] * A.data[contrib["Pj"]]
        if contrib.get("all_entries_nonempty", False) and starts.size == H.data.size:
            H.data[:] = np.add.reduceat(values, starts)
            return H
        ends = entry_ptr[1:]
        nonempty = ends > starts
        H.data.fill(0.0)
        H.data[nonempty] = np.add.reduceat(values, starts[nonempty])
        return H

    def update_H_data_simple_planned_weighted(self, A, H, contrib, weight_by_contrib):
        """Refresh H with ``B_diag[contrib["K"]]`` already materialized."""
        if not sp.isspmatrix_csc(A):
            A = sp.csc_matrix(A)
        if not sp.isspmatrix_csc(H):
            raise TypeError("update_H_data_simple_planned_weighted requires a scipy.sparse.csc_matrix H pattern")
        weight_by_contrib = np.asarray(weight_by_contrib, dtype=np.float64)
        if tuple(A.shape) != tuple(contrib.get("A_shape", A.shape)):
            raise ValueError(f"A shape {A.shape} does not match planned shape {contrib.get('A_shape')}")
        if tuple(H.shape) != tuple(contrib.get("H_shape", H.shape)) or int(H.nnz) != int(contrib.get("H_nnz", H.nnz)):
            raise ValueError("H shape or sparsity size does not match planned H pattern")
        if weight_by_contrib.ndim != 1 or weight_by_contrib.shape[0] != contrib["K"].size:
            raise ValueError("weight_by_contrib length does not match planned contribution count")

        entry_ptr = contrib["entry_ptr"]
        starts = entry_ptr[:-1]
        if contrib["K"].size == 0:
            H.data.fill(0.0)
            return H
        values = A.data[contrib["Pi"]] * weight_by_contrib * A.data[contrib["Pj"]]
        if contrib.get("all_entries_nonempty", False) and starts.size == H.data.size:
            H.data[:] = np.add.reduceat(values, starts)
            return H
        ends = entry_ptr[1:]
        nonempty = ends > starts
        H.data.fill(0.0)
        H.data[nonempty] = np.add.reduceat(values, starts[nonempty])
        return H

    def symbolic_AtBA_lower(self, matrix):
        """Build lower-triangular sparsity and contribution lookup for ``A.T @ B @ A``."""
        matrix = matrix.tocsc()
        m, n = matrix.shape

        col_of = np.repeat(np.arange(n, dtype=np.int64), np.diff(matrix.indptr))
        row_of = matrix.indices.astype(np.int64, copy=False)

        order = np.lexsort((col_of, row_of))
        sorted_rows = row_of[order]
        sorted_cols = col_of[order]
        sorted_data_pos = order
        row_start = np.searchsorted(sorted_rows, np.arange(m + 1))

        row_i_parts = []
        col_j_parts = []
        row_k_parts = []
        data_i_parts = []
        data_j_parts = []
        for row in range(m):
            start, end = int(row_start[row]), int(row_start[row + 1])
            width = end - start
            if width == 0:
                continue
            cols_row = sorted_cols[start:end]
            data_pos_row = sorted_data_pos[start:end]
            left, right = np.divmod(np.arange(width * width), width)
            row_i = cols_row[left]
            col_j = cols_row[right]
            lower_mask = row_i >= col_j
            row_i_parts.append(row_i[lower_mask])
            col_j_parts.append(col_j[lower_mask])
            row_k_parts.append(np.full(int(lower_mask.sum()), row, dtype=np.int64))
            data_i_parts.append(data_pos_row[left][lower_mask])
            data_j_parts.append(data_pos_row[right][lower_mask])

        if not row_i_parts:
            h_matrix = sp.csc_matrix((n, n), dtype=matrix.dtype)
            return h_matrix, {
                "entry_ptr": np.zeros(1, dtype=np.int64),
                "K": np.zeros(0, dtype=np.int64),
                "Pi": np.zeros(0, dtype=np.int64),
                "Pj": np.zeros(0, dtype=np.int64),
            }

        row_i_all = np.concatenate(row_i_parts)
        col_j_all = np.concatenate(col_j_parts)
        row_k_all = np.concatenate(row_k_parts)
        data_i_all = np.concatenate(data_i_parts)
        data_j_all = np.concatenate(data_j_parts)

        order_h = np.lexsort((row_i_all, col_j_all))
        row_i_all = row_i_all[order_h]
        col_j_all = col_j_all[order_h]
        row_k_all = row_k_all[order_h]
        data_i_all = data_i_all[order_h]
        data_j_all = data_j_all[order_h]

        is_new = np.empty(row_i_all.size, dtype=bool)
        is_new[0] = True
        is_new[1:] = (row_i_all[1:] != row_i_all[:-1]) | (col_j_all[1:] != col_j_all[:-1])
        starts = np.nonzero(is_new)[0]
        entry_ptr = np.empty(starts.size + 1, dtype=np.int64)
        entry_ptr[:-1] = starts
        entry_ptr[-1] = row_i_all.size

        indices = row_i_all[starts]
        cols_unique = col_j_all[starts]
        indptr = np.zeros(n + 1, dtype=np.int64)
        np.add.at(indptr, cols_unique + 1, 1)
        np.cumsum(indptr, out=indptr)

        h_matrix = sp.csc_matrix(
            (np.zeros(indices.size, dtype=matrix.dtype), indices, indptr),
            shape=(n, n),
        )
        return h_matrix, {
            "entry_ptr": entry_ptr,
            "K": row_k_all,
            "Pi": data_i_all,
            "Pj": data_j_all,
        }

    def update_AtBA_lower(self, h_matrix, matrix_data, b_diag, contrib):
        """Refresh lower-triangular ``A.T @ B @ A`` data with fixed sparsity."""
        values = (
            matrix_data[contrib["Pi"]]
            * b_diag[contrib["K"]]
            * matrix_data[contrib["Pj"]]
        )
        h_matrix.data[:] = np.add.reduceat(values, contrib["entry_ptr"][:-1])
        return h_matrix

    def _rhs_for_iteration(self, rhs, iteration_index, h_matrix):
        if callable(rhs):
            rhs_value = rhs(iteration_index, h_matrix)
        elif rhs is None:
            rhs_value = np.ones(h_matrix.shape[0], dtype=np.float64)
        else:
            rhs_value = rhs
        rhs_array = np.asarray(rhs_value, dtype=np.float64)
        if rhs_array.ndim != 1 or rhs_array.shape[0] != h_matrix.shape[0]:
            raise ValueError(
                f"RHS vector length {rhs_array.shape} does not match H shape {h_matrix.shape}"
            )
        return rhs_array

    def _check_same_jacobian_pattern(self, matrix, shape, indptr, indices, iteration_index):
        if tuple(matrix.shape) != tuple(shape):
            raise ValueError(
                f"Jacobian shape changed at iteration {iteration_index}: {matrix.shape} != {shape}"
            )
        if not np.array_equal(matrix.indptr, indptr) or not np.array_equal(matrix.indices, indices):
            raise ValueError(f"Jacobian sparsity pattern changed at iteration {iteration_index}")

    def _validate_loaded_matrix_pair(self, j_matrix, d_vector, iteration_index):
        if d_vector.shape[0] != j_matrix.shape[0]:
            raise ValueError(
                f"D vector length {d_vector.shape[0]} does not match J rows {j_matrix.shape[0]} "
                f"at iteration {iteration_index}"
            )

    def main(
        self,
        directory=None,
        *,
        iterations=13,
        rhs=None,
        start_index=1,
        j_template="j{index}.txt",
        d_template="d{index}.txt",
        h_template="h{index}.txt",
        matrix_lists=None,
    ):
        """Read dumped J/D files repeatedly and solve ``(J.T D J) x = b``.

        The first iteration builds the lower-triangular H sparsity structure and
        runs CHOLMOD symbolic analysis. Every iteration then refreshes H numeric
        data, updates the numeric factorization in place, and solves with the
        reusable factor.
        """
        h_matrix = None
        contrib = None
        factor = None
        jacobian_shape = None
        jacobian_indptr = None
        jacobian_indices = None
        results = []
        j_list, d_list, _h_list = self._matrix_lists_or_read(
            directory,
            matrix_lists,
            iterations=iterations,
            start_index=start_index,
            j_template=j_template,
            d_template=d_template,
            h_template=h_template,
        )

        for offset, (j_matrix, d_vector) in enumerate(zip(j_list, d_list)):
            index = int(start_index) + offset
            self._validate_loaded_matrix_pair(j_matrix, d_vector, index)

            if h_matrix is None:
                h_matrix, contrib = self.symbolic_AtBA_lower(j_matrix)
                factor = self.cholmod_analyze(h_matrix)
                jacobian_shape = tuple(j_matrix.shape)
                jacobian_indptr = j_matrix.indptr.copy()
                jacobian_indices = j_matrix.indices.copy()
            else:
                self._check_same_jacobian_pattern(
                    j_matrix,
                    jacobian_shape,
                    jacobian_indptr,
                    jacobian_indices,
                    index,
                )

            self.update_AtBA_lower(h_matrix, j_matrix.data, d_vector, contrib)
            self.cholmod_factorize_inplace(factor, h_matrix)
            rhs_array = self._rhs_for_iteration(rhs, index, h_matrix)
            results.append(self.solve(factor, rhs_array))

        return results

    def main2(
        self,
        directory=None,
        *,
        iterations=13,
        rhs=None,
        start_index=1,
        j_template="j{index}.txt",
        d_template="d{index}.txt",
        h_template="h{index}.txt",
        matrix_lists=None,
    ):
        """Solve repeated ``J.T D J`` systems using ``get_H_sparsity/update_H_data``."""
        h_matrix = None
        factor = None
        jacobian_shape = None
        jacobian_indptr = None
        jacobian_indices = None
        results = []
        j_list, d_list, _h_list = self._matrix_lists_or_read(
            directory,
            matrix_lists,
            iterations=iterations,
            start_index=start_index,
            j_template=j_template,
            d_template=d_template,
            h_template=h_template,
        )

        for offset, (j_matrix, d_vector) in enumerate(zip(j_list, d_list)):
            index = int(start_index) + offset
            self._validate_loaded_matrix_pair(j_matrix, d_vector, index)

            if h_matrix is None:
                h_matrix = self.get_H_sparsity(j_matrix)
                factor = self.cholmod_analyze(h_matrix)
                jacobian_shape = tuple(j_matrix.shape)
                jacobian_indptr = j_matrix.indptr.copy()
                jacobian_indices = j_matrix.indices.copy()
            else:
                self._check_same_jacobian_pattern(
                    j_matrix,
                    jacobian_shape,
                    jacobian_indptr,
                    jacobian_indices,
                    index,
                )

            self.update_H_data(h_matrix, j_matrix, d_vector)
            self.cholmod_factorize_inplace(factor, h_matrix)
            rhs_array = self._rhs_for_iteration(rhs, index, h_matrix)
            results.append(self.solve(factor, rhs_array))

        return results

    def main3(
        self,
        directory=None,
        *,
        iterations=13,
        rhs=None,
        start_index=1,
        j_template="j{index}.txt",
        d_template="d{index}.txt",
        h_template="h{index}.txt",
        matrix_lists=None,
    ):
        """Solve repeated systems like ``main2`` using ``update_H_data_simple``."""
        h_matrix = None
        factor = None
        jacobian_shape = None
        jacobian_indptr = None
        jacobian_indices = None
        results = []
        j_list, d_list, _h_list = self._matrix_lists_or_read(
            directory,
            matrix_lists,
            iterations=iterations,
            start_index=start_index,
            j_template=j_template,
            d_template=d_template,
            h_template=h_template,
        )

        for offset, (j_matrix, d_vector) in enumerate(zip(j_list, d_list)):
            index = int(start_index) + offset
            self._validate_loaded_matrix_pair(j_matrix, d_vector, index)

            if h_matrix is None:
                h_matrix = self.get_H_sparsity(j_matrix)
                factor = self.cholmod_analyze(h_matrix)
                jacobian_shape = tuple(j_matrix.shape)
                jacobian_indptr = j_matrix.indptr.copy()
                jacobian_indices = j_matrix.indices.copy()
            else:
                self._check_same_jacobian_pattern(
                    j_matrix,
                    jacobian_shape,
                    jacobian_indptr,
                    jacobian_indices,
                    index,
                )

            self.update_H_data_simple(j_matrix, d_vector, h_matrix)
            self.cholmod_factorize_inplace(factor, h_matrix)
            rhs_array = self._rhs_for_iteration(rhs, index, h_matrix)
            results.append(self.solve(factor, rhs_array))

        return results

    def main4(
        self,
        directory=None,
        *,
        iterations=13,
        rhs=None,
        start_index=1,
        j_template="j{index}.txt",
        d_template="d{index}.txt",
        h_template="h{index}.txt",
        matrix_lists=None,
    ):
        """Solve repeated systems using a planned H data refresh."""
        h_matrix = None
        contrib = None
        factor = None
        jacobian_shape = None
        jacobian_indptr = None
        jacobian_indices = None
        results = []
        j_list, d_list, _h_list = self._matrix_lists_or_read(
            directory,
            matrix_lists,
            iterations=iterations,
            start_index=start_index,
            j_template=j_template,
            d_template=d_template,
            h_template=h_template,
        )

        for offset, (j_matrix, d_vector) in enumerate(zip(j_list, d_list)):
            index = int(start_index) + offset
            self._validate_loaded_matrix_pair(j_matrix, d_vector, index)

            if h_matrix is None:
                h_matrix, contrib = self.symbolic_H_data_simple_pattern_plan(j_matrix)
                factor = self.cholmod_analyze(h_matrix)
                jacobian_shape = tuple(j_matrix.shape)
                jacobian_indptr = j_matrix.indptr.copy()
                jacobian_indices = j_matrix.indices.copy()
            else:
                self._check_same_jacobian_pattern(
                    j_matrix,
                    jacobian_shape,
                    jacobian_indptr,
                    jacobian_indices,
                    index,
                )

            self.update_H_data_simple_planned(j_matrix, d_vector, h_matrix, contrib)
            self.cholmod_factorize_inplace(factor, h_matrix)
            rhs_array = self._rhs_for_iteration(rhs, index, h_matrix)
            results.append(self.solve(factor, rhs_array))

        return results

    def main5_prepare(self, j_matrix, d_vector=None):
        """Build reusable ``main5`` symbolic data, CHOLMOD analysis, and optional D cache."""
        if d_vector is not None:
            self._validate_loaded_matrix_pair(j_matrix, np.asarray(d_vector, dtype=np.float64), 1)
        h_matrix, contrib = self.symbolic_H_data_simple_pattern_plan(j_matrix)
        prepared = {
            "h_matrix": h_matrix,
            "contrib": contrib,
            "factor": self.cholmod_analyze(h_matrix),
            "jacobian_shape": tuple(j_matrix.shape),
            "jacobian_indptr": j_matrix.indptr.copy(),
            "jacobian_indices": j_matrix.indices.copy(),
            "d_vector": None,
            "weight_by_contrib": None,
        }
        if d_vector is not None:
            d_array = np.asarray(d_vector, dtype=np.float64)
            prepared["d_vector"] = d_array.copy()
            prepared["weight_by_contrib"] = d_array[contrib["K"]].copy()
        return prepared

    def main5(
        self,
        directory=None,
        *,
        iterations=13,
        rhs=None,
        start_index=1,
        j_template="j{index}.txt",
        d_template="d{index}.txt",
        h_template="h{index}.txt",
        matrix_lists=None,
        prepared=None,
    ):
        """Solve like ``main4`` with reusable prepare data and cached D-by-contribution weights."""
        j_list, d_list, _h_list = self._matrix_lists_or_read(
            directory,
            matrix_lists,
            iterations=iterations,
            start_index=start_index,
            j_template=j_template,
            d_template=d_template,
            h_template=h_template,
        )
        if not j_list:
            return []

        if prepared is None:
            prepared = self.main5_prepare(j_list[0], d_list[0])

        h_matrix = prepared["h_matrix"]
        contrib = prepared["contrib"]
        factor = prepared["factor"]
        jacobian_shape = prepared["jacobian_shape"]
        jacobian_indptr = prepared["jacobian_indptr"]
        jacobian_indices = prepared["jacobian_indices"]
        d_reference = prepared.get("d_vector")
        weight_by_contrib = prepared.get("weight_by_contrib")
        use_cached_weight = (
            d_reference is not None
            and weight_by_contrib is not None
            and all(np.array_equal(np.asarray(item, dtype=np.float64), d_reference) for item in d_list)
        )

        results = []
        for offset, (j_matrix, d_vector) in enumerate(zip(j_list, d_list)):
            index = int(start_index) + offset
            self._validate_loaded_matrix_pair(j_matrix, d_vector, index)
            self._check_same_jacobian_pattern(
                j_matrix,
                jacobian_shape,
                jacobian_indptr,
                jacobian_indices,
                index,
            )

            if use_cached_weight:
                self.update_H_data_simple_planned_weighted(j_matrix, h_matrix, contrib, weight_by_contrib)
            else:
                self.update_H_data_simple_planned(j_matrix, d_vector, h_matrix, contrib)
            self.cholmod_factorize_inplace(factor, h_matrix)
            rhs_array = self._rhs_for_iteration(rhs, index, h_matrix)
            results.append(self.solve(factor, rhs_array))

        return results


def _parse_rhs_arg(rhs_arg):
    if rhs_arg is None or rhs_arg == "":
        return None
    path = Path(rhs_arg)
    if path.exists():
        return np.loadtxt(path, dtype=np.float64)
    return np.fromstring(rhs_arg, sep=",", dtype=np.float64)


def _run_cli(argv=None):
    parser = argparse.ArgumentParser(description="Run repeated CHOLMOD solves from SE matrix dumps.")
    parser.add_argument("directory", help="Directory containing j*.txt and d*.txt sparse dumps.")
    parser.add_argument("--iterations", type=int, default=13)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--j-template", default="j{index}.txt")
    parser.add_argument("--d-template", default="d{index}.txt")
    parser.add_argument("--rhs", default=None, help="Comma-separated RHS values or a text file path.")
    args = parser.parse_args(argv)

    results = SEMathTest().main(
        args.directory,
        iterations=args.iterations,
        rhs=_parse_rhs_arg(args.rhs),
        start_index=args.start_index,
        j_template=args.j_template,
        d_template=args.d_template,
    )
    for offset, result in enumerate(results, start=args.start_index):
        print(f"iter={offset} size={result.size} max_abs={float(np.max(np.abs(result))):.17e}")


if __name__ == "__main__":
    _run_cli()
