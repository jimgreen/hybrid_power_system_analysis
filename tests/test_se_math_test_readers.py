from pathlib import Path

import numpy as np
from scipy.sparse import csc_matrix, isspmatrix_csc


def _write_triplet(path: Path, rows: int, cols: int, label: str, entries) -> None:
    lines = [
        "# sparse_triplet row col value",
        f"# shape {rows} {cols} nnz {len(entries)}",
        f"# matrix {label}",
    ]
    lines.extend(f"{row} {col} {value:.17e}" for row, col, value in entries)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_se_math_test_reads_matrix_dump_triplets(tmp_path):
    from secore.se_math_test import SEMathTest

    math_test = SEMathTest()

    _write_triplet(
        tmp_path / "j1.txt",
        3,
        4,
        "jacobian",
        [(1, 2, 1.5), (3, 4, -2.0), (2, 1, 0.25)],
    )
    _write_triplet(
        tmp_path / "d1.txt",
        3,
        3,
        "weight_diagonal",
        [(1, 1, 10.0), (2, 2, 20.0), (3, 3, 30.0)],
    )
    _write_triplet(
        tmp_path / "h1.txt",
        4,
        4,
        "information_matrix",
        [(1, 1, 5.0), (2, 1, -1.0), (1, 2, -1.0), (4, 4, 7.5)],
    )

    j = math_test.read_j_matrix(tmp_path, "j1.txt")
    d = math_test.read_d_vector(tmp_path, "d1.txt")
    h = math_test.read_h_matrix(tmp_path, "h1.txt")

    assert isspmatrix_csc(j)
    assert isspmatrix_csc(h)
    assert j.shape == (3, 4)
    assert h.shape == (4, 4)
    np.testing.assert_allclose(
        j.toarray(),
        np.array(
            [
                [0.0, 1.5, 0.0, 0.0],
                [0.25, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -2.0],
            ]
        ),
    )
    np.testing.assert_allclose(d, np.array([10.0, 20.0, 30.0]))
    np.testing.assert_allclose(
        h.toarray(),
        np.array(
            [
                [5.0, -1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 7.5],
            ]
        ),
    )


def test_update_csc_matrix_from_triplet_file_reuses_allocated_structure(tmp_path):
    from secore.se_math_test import SEMathTest

    math_test = SEMathTest()

    _write_triplet(
        tmp_path / "h1.txt",
        4,
        4,
        "information_matrix",
        [(1, 1, 5.0), (2, 1, -1.0), (1, 2, -1.0), (4, 4, 7.5)],
    )
    matrix = csc_matrix(
        (
            np.zeros(4, dtype=np.float64),
            (
                np.array([0, 1, 0, 3], dtype=np.int64),
                np.array([0, 0, 1, 3], dtype=np.int64),
            ),
        ),
        shape=(4, 4),
    )
    matrix.sort_indices()
    old_data = matrix.data
    old_indices = matrix.indices
    old_indptr = matrix.indptr

    returned = math_test.update_csc_matrix_from_file(matrix, tmp_path, "h1.txt")

    assert returned is matrix
    assert matrix.data is old_data
    assert matrix.indices is old_indices
    assert matrix.indptr is old_indptr
    np.testing.assert_allclose(
        matrix.toarray(),
        np.array(
            [
                [5.0, -1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 7.5],
            ]
        ),
    )


def test_solve_cholmod_uses_cholmod_factorization(monkeypatch):
    import secore.se_math_test as se_math_test

    calls = []

    class FakeFactor:
        def __init__(self, matrix):
            self.matrix = matrix

        def __call__(self, rhs):
            return np.linalg.solve(self.matrix.toarray(), rhs)

    def fake_cholesky(matrix):
        calls.append(matrix)
        return FakeFactor(matrix)

    monkeypatch.setattr(se_math_test, "CHOLMOD_CHOLESKY", fake_cholesky)
    math_test = se_math_test.SEMathTest()
    matrix = csc_matrix(np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64))
    rhs = np.array([1.0, 2.0], dtype=np.float64)

    x = math_test.solve_cholmod(matrix, rhs)

    assert len(calls) == 1
    assert isspmatrix_csc(calls[0])
    np.testing.assert_allclose(x, np.linalg.solve(matrix.toarray(), rhs))


def test_cholmod_factorize_once_solves_multiple_rhs(monkeypatch):
    import secore.se_math_test as se_math_test

    factorize_calls = []
    solve_calls = []

    class FakeFactor:
        def __init__(self, matrix):
            self.matrix = matrix

        def __call__(self, rhs):
            solve_calls.append(np.asarray(rhs).copy())
            return np.linalg.solve(self.matrix.toarray(), rhs)

    def fake_cholesky(matrix):
        factorize_calls.append(matrix)
        return FakeFactor(matrix)

    monkeypatch.setattr(se_math_test, "CHOLMOD_CHOLESKY", fake_cholesky)
    math_test = se_math_test.SEMathTest()
    matrix = csc_matrix(np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64))
    rhs1 = np.array([1.0, 2.0], dtype=np.float64)
    rhs2 = np.array([3.0, 4.0], dtype=np.float64)

    factor = math_test.cholmod_factorize(matrix)
    x1 = math_test.cholmod_solve(factor, rhs1)
    x2 = math_test.cholmod_solve(factor, rhs2)

    assert len(factorize_calls) == 1
    assert len(solve_calls) == 2
    assert isspmatrix_csc(factorize_calls[0])
    np.testing.assert_allclose(x1, np.linalg.solve(matrix.toarray(), rhs1))
    np.testing.assert_allclose(x2, np.linalg.solve(matrix.toarray(), rhs2))


def test_cholmod_analyze_once_refactorizes_numeric_values(monkeypatch):
    import secore.se_math_test as se_math_test

    analyze_calls = []
    numeric_calls = []
    solve_calls = []

    class FakeAnalyzedFactor:
        def __init__(self, matrix):
            self.matrix = matrix

        def cholesky_inplace(self, matrix):
            numeric_calls.append(matrix)
            self.matrix = matrix

        def __call__(self, rhs):
            solve_calls.append(np.asarray(rhs).copy())
            return np.linalg.solve(self.matrix.toarray(), rhs)

    def fake_analyze(matrix):
        analyze_calls.append(matrix)
        return FakeAnalyzedFactor(matrix)

    monkeypatch.setattr(se_math_test, "CHOLMOD_ANALYZE", fake_analyze)
    math_test = se_math_test.SEMathTest()
    matrix1 = csc_matrix(np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64))
    matrix2 = csc_matrix(np.array([[5.0, 1.0], [1.0, 6.0]], dtype=np.float64))
    rhs1 = np.array([1.0, 2.0], dtype=np.float64)
    rhs2 = np.array([3.0, 4.0], dtype=np.float64)

    factor = math_test.cholmod_analyze(matrix1)
    math_test.cholmod_factorize_inplace(factor, matrix1)
    x1 = math_test.cholmod_solve(factor, rhs1)
    math_test.cholmod_factorize_inplace(factor, matrix2)
    x2 = math_test.cholmod_solve(factor, rhs2)

    assert len(analyze_calls) == 1
    assert len(numeric_calls) == 2
    assert len(solve_calls) == 2
    assert isspmatrix_csc(analyze_calls[0])
    assert all(isspmatrix_csc(matrix) for matrix in numeric_calls)
    np.testing.assert_allclose(x1, np.linalg.solve(matrix1.toarray(), rhs1))
    np.testing.assert_allclose(x2, np.linalg.solve(matrix2.toarray(), rhs2))


def test_solve_uses_precomputed_factor():
    from secore.se_math_test import SEMathTest

    solve_calls = []

    class FakeFactor:
        def __call__(self, rhs):
            solve_calls.append(np.asarray(rhs).copy())
            return np.array([rhs[0] + rhs[1], rhs[1] - rhs[0]], dtype=np.float64)

    math_test = SEMathTest()
    rhs = np.array([2.0, 5.0], dtype=np.float64)

    result = math_test.solve(FakeFactor(), rhs)

    assert len(solve_calls) == 1
    np.testing.assert_allclose(solve_calls[0], rhs)
    np.testing.assert_allclose(result, np.array([7.0, 3.0]))


def test_main_reads_13_matrix_pairs_and_reuses_symbolic_analysis(tmp_path):
    from secore.se_math_test import SEMathTest

    for index in range(1, 14):
        _write_triplet(
            tmp_path / f"j{index}.txt",
            3,
            2,
            "jacobian",
            [(1, 1, 1.0 + index), (2, 2, 2.0 + index), (3, 1, 3.0 + index)],
        )
        _write_triplet(
            tmp_path / f"d{index}.txt",
            3,
            3,
            "weight_diagonal",
            [(1, 1, 1.0), (2, 2, 2.0), (3, 3, 3.0)],
        )

    class FakeFactor:
        def __init__(self):
            self.matrix = None

        def cholesky_inplace(self, matrix):
            self.matrix = matrix.copy()

        def __call__(self, rhs):
            return np.asarray(rhs, dtype=np.float64) + self.matrix.diagonal()

    class RecordingSEMathTest(SEMathTest):
        def __init__(self):
            super().__init__()
            self.symbolic_calls = []
            self.analyze_calls = []
            self.update_calls = []
            self.factorize_inplace_calls = []
            self.solve_calls = []

        def symbolic_AtBA_lower(self, matrix):
            self.symbolic_calls.append(matrix.copy())
            h_matrix = csc_matrix(np.eye(matrix.shape[1], dtype=np.float64))
            return h_matrix, {"token": "contrib"}

        def update_AtBA_lower(self, h_matrix, matrix_data, b_diag, contrib):
            self.update_calls.append((matrix_data.copy(), b_diag.copy(), contrib))
            h_matrix.data[:] = len(self.update_calls)
            return h_matrix

        def cholmod_analyze(self, matrix):
            self.analyze_calls.append(matrix.copy())
            return FakeFactor()

        def cholmod_factorize_inplace(self, factor, matrix):
            self.factorize_inplace_calls.append(matrix.copy())
            return super().cholmod_factorize_inplace(factor, matrix)

        def solve(self, factor, rhs):
            self.solve_calls.append(np.asarray(rhs).copy())
            return super().solve(factor, rhs)

    math_test = RecordingSEMathTest()
    rhs = np.array([10.0, 20.0], dtype=np.float64)

    results = math_test.main(tmp_path, rhs=rhs)

    assert len(results) == 13
    assert len(math_test.symbolic_calls) == 1
    assert len(math_test.analyze_calls) == 1
    assert len(math_test.update_calls) == 13
    assert len(math_test.factorize_inplace_calls) == 13
    assert len(math_test.solve_calls) == 13
    np.testing.assert_allclose(results[0], np.array([11.0, 21.0]))
    np.testing.assert_allclose(results[-1], np.array([23.0, 33.0]))


def test_load_matrix_lists_reads_j_d_h_files_once(tmp_path):
    from secore.se_math_test import SEMathTest

    for index in range(1, 4):
        _write_triplet(
            tmp_path / f"j{index}.txt",
            3,
            2,
            "jacobian",
            [(1, 1, index), (2, 2, index + 1), (3, 1, index + 2)],
        )
        _write_triplet(
            tmp_path / f"d{index}.txt",
            3,
            3,
            "weight_diagonal",
            [(1, 1, index), (2, 2, index + 1), (3, 3, index + 2)],
        )
        _write_triplet(
            tmp_path / f"h{index}.txt",
            2,
            2,
            "information_matrix",
            [(1, 1, index), (2, 1, index + 1), (2, 2, index + 2)],
        )

    math_test = SEMathTest()

    j_list, d_list, h_list = math_test.load_matrix_lists(tmp_path, iterations=3)

    assert len(j_list) == 3
    assert len(d_list) == 3
    assert len(h_list) == 3
    assert all(isspmatrix_csc(matrix) for matrix in j_list)
    assert all(isspmatrix_csc(matrix) for matrix in h_list)
    np.testing.assert_allclose(d_list[0], np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(h_list[-1].toarray(), np.array([[3.0, 0.0], [4.0, 5.0]]))


def test_main_variants_accept_preloaded_read_file_data(tmp_path):
    from secore.se_math_test import SEMathTest

    for index in range(1, 4):
        _write_triplet(
            tmp_path / f"j{index}.txt",
            3,
            2,
            "jacobian",
            [(1, 1, 1.0 + index), (2, 2, 2.0 + index), (3, 1, 3.0 + index)],
        )
        _write_triplet(
            tmp_path / f"d{index}.txt",
            3,
            3,
            "weight_diagonal",
            [(1, 1, 1.0), (2, 2, 2.0), (3, 3, 3.0)],
        )

    class FakeFactor:
        def __init__(self):
            self.matrix = None

        def cholesky_inplace(self, matrix):
            self.matrix = matrix.copy()

        def __call__(self, rhs):
            return np.asarray(rhs, dtype=np.float64) + self.matrix.diagonal()

    class PreloadedOnlySEMathTest(SEMathTest):
        def __init__(self):
            super().__init__()
            self.update_calls = []

        def read_file(self, *args, **kwargs):
            raise AssertionError("main variants should use preloaded matrix_lists")

        def symbolic_AtBA_lower(self, matrix):
            return csc_matrix(np.eye(matrix.shape[1], dtype=np.float64)), {"token": "main"}

        def update_AtBA_lower(self, h_matrix, matrix_data, b_diag, contrib):
            self.update_calls.append(("main", contrib))
            h_matrix.data[:] = len(self.update_calls)
            return h_matrix

        def get_H_sparsity(self, matrix):
            return csc_matrix(np.eye(matrix.shape[1], dtype=np.float64))

        def update_H_data(self, h_matrix, matrix, d_vector):
            self.update_calls.append(("main2", matrix.shape))
            h_matrix.data[:] = len(self.update_calls)
            return h_matrix

        def symbolic_H_data_simple_pattern_plan(self, matrix):
            return csc_matrix(np.eye(matrix.shape[1], dtype=np.float64)), {"token": "main4"}

        def update_H_data_simple_planned(self, matrix, d_vector, h_matrix, contrib):
            self.update_calls.append(("main4", contrib))
            h_matrix.data[:] = len(self.update_calls)
            return h_matrix

        def main5_prepare(self, matrix, d_vector=None):
            return {
                "h_matrix": csc_matrix(np.eye(matrix.shape[1], dtype=np.float64)),
                "contrib": {"token": "main5"},
                "factor": FakeFactor(),
                "jacobian_shape": tuple(matrix.shape),
                "jacobian_indptr": matrix.indptr.copy(),
                "jacobian_indices": matrix.indices.copy(),
                "d_vector": np.asarray(d_vector, dtype=np.float64).copy(),
                "weight_by_contrib": np.ones(matrix.shape[1], dtype=np.float64),
            }

        def update_H_data_simple_planned_weighted(self, matrix, h_matrix, contrib, weight_by_contrib):
            self.update_calls.append(("main5", contrib))
            h_matrix.data[:] = len(self.update_calls)
            return h_matrix

        def cholmod_analyze(self, matrix):
            return FakeFactor()

    matrix_lists = SEMathTest().read_file(tmp_path, iterations=3)
    rhs = np.array([10.0, 20.0], dtype=np.float64)

    for method_name in ("main", "main2", "main4", "main5"):
        math_test = PreloadedOnlySEMathTest()
        results = getattr(math_test, method_name)(matrix_lists=matrix_lists, iterations=3, rhs=rhs)

        assert len(results) == 3
        assert len(math_test.update_calls) == 3
        np.testing.assert_allclose(results[0], np.array([11.0, 21.0]))


def test_main_loads_files_once_then_computes_from_memory(tmp_path):
    from secore.se_math_test import SEMathTest

    for index in range(1, 4):
        _write_triplet(
            tmp_path / f"j{index}.txt",
            3,
            2,
            "jacobian",
            [(1, 1, 1.0 + index), (2, 2, 2.0 + index), (3, 1, 3.0 + index)],
        )
        _write_triplet(
            tmp_path / f"d{index}.txt",
            3,
            3,
            "weight_diagonal",
            [(1, 1, 1.0), (2, 2, 2.0), (3, 3, 3.0)],
        )

    class FakeFactor:
        def __init__(self):
            self.matrix = None

        def cholesky_inplace(self, matrix):
            self.matrix = matrix.copy()

        def __call__(self, rhs):
            return np.asarray(rhs, dtype=np.float64) + self.matrix.diagonal()

    class RecordingSEMathTest(SEMathTest):
        def __init__(self):
            super().__init__()
            self.events = []

        def read_j_matrix(self, directory, filename):
            self.events.append(("read_j", filename))
            return super().read_j_matrix(directory, filename)

        def read_d_vector(self, directory, filename):
            self.events.append(("read_d", filename))
            return super().read_d_vector(directory, filename)

        def symbolic_AtBA_lower(self, matrix):
            self.events.append(("symbolic", matrix.shape))
            return csc_matrix(np.eye(matrix.shape[1], dtype=np.float64)), {"token": "contrib"}

        def update_AtBA_lower(self, h_matrix, matrix_data, b_diag, contrib):
            self.events.append(("update", len(self.events)))
            h_matrix.data[:] = 1.0
            return h_matrix

        def cholmod_analyze(self, matrix):
            self.events.append(("analyze", matrix.shape))
            return FakeFactor()

    math_test = RecordingSEMathTest()
    results = math_test.main(tmp_path, iterations=3, rhs=np.array([1.0, 2.0]))
    first_compute_event = next(
        pos for pos, event in enumerate(math_test.events) if event[0] == "symbolic"
    )

    assert len(results) == 3
    assert [event[0] for event in math_test.events[:first_compute_event]] == [
        "read_j",
        "read_d",
        "read_j",
        "read_d",
        "read_j",
        "read_d",
    ]
    assert not any(event[0].startswith("read_") for event in math_test.events[first_compute_event:])


def test_update_H_data_refreshes_get_H_sparsity_pattern_in_place():
    from secore.se_math_test import SEMathTest

    math_test = SEMathTest()
    matrix = csc_matrix(
        np.array(
            [
                [1.0, 0.0],
                [0.0, 2.0],
                [3.0, 4.0],
            ],
            dtype=np.float64,
        )
    )
    d_vector = np.array([5.0, 6.0, 7.0], dtype=np.float64)
    h_matrix = math_test.get_H_sparsity(matrix)
    old_data = h_matrix.data
    expected = np.tril(matrix.T.toarray() @ np.diag(d_vector) @ matrix.toarray())

    returned = math_test.update_H_data(h_matrix, matrix, d_vector)

    assert returned is h_matrix
    assert h_matrix.data is old_data
    np.testing.assert_allclose(h_matrix.toarray(), expected)


def test_update_H_data_simple_refreshes_get_H_sparsity_pattern_in_place():
    from secore.se_math_test import SEMathTest

    math_test = SEMathTest()
    matrix = csc_matrix(
        np.array(
            [
                [1.0, 0.0],
                [0.0, 2.0],
                [3.0, 4.0],
            ],
            dtype=np.float64,
        )
    )
    d_vector = np.array([5.0, 6.0, 7.0], dtype=np.float64)
    h_matrix = math_test.get_H_sparsity(matrix)
    old_data = h_matrix.data
    expected = np.tril(matrix.T.toarray() @ np.diag(d_vector) @ matrix.toarray())

    returned = math_test.update_H_data_simple(matrix, d_vector, h_matrix)

    assert returned is h_matrix
    assert h_matrix.data is old_data
    np.testing.assert_allclose(h_matrix.toarray(), expected)


def test_planned_H_data_update_reuses_symbolic_intersection_plan():
    from secore.se_math_test import SEMathTest

    math_test = SEMathTest()
    matrix = csc_matrix(
        np.array(
            [
                [1.0, 0.0, 2.0],
                [0.0, 3.0, 4.0],
                [5.0, 6.0, 0.0],
            ],
            dtype=np.float64,
        )
    )
    d_vector_1 = np.array([2.0, 3.0, 4.0], dtype=np.float64)
    d_vector_2 = np.array([5.0, 6.0, 7.0], dtype=np.float64)
    h_matrix = math_test.get_H_sparsity(matrix)
    old_data = h_matrix.data

    contrib = math_test.symbolic_H_data_simple_plan(matrix, h_matrix)
    returned_1 = math_test.update_H_data_simple_planned(matrix, d_vector_1, h_matrix, contrib)
    expected_1 = np.tril(matrix.T.toarray() @ np.diag(d_vector_1) @ matrix.toarray())
    actual_1 = h_matrix.toarray().copy()
    returned_2 = math_test.update_H_data_simple_planned(matrix, d_vector_2, h_matrix, contrib)
    expected_2 = np.tril(matrix.T.toarray() @ np.diag(d_vector_2) @ matrix.toarray())

    assert returned_1 is h_matrix
    assert returned_2 is h_matrix
    assert h_matrix.data is old_data
    assert set(contrib) >= {"entry_ptr", "K", "Pi", "Pj"}
    assert contrib["entry_ptr"].size == h_matrix.nnz + 1
    assert contrib["K"].size == contrib["Pi"].size == contrib["Pj"].size
    np.testing.assert_allclose(actual_1, expected_1)
    np.testing.assert_allclose(h_matrix.toarray(), expected_2)


def test_symbolic_H_data_simple_pattern_plan_builds_h_and_reusable_plan():
    from secore.se_math_test import SEMathTest

    math_test = SEMathTest()
    matrix = csc_matrix(
        np.array(
            [
                [1.0, 0.0, 2.0],
                [0.0, 3.0, 4.0],
                [5.0, 6.0, 0.0],
            ],
            dtype=np.float64,
        )
    )
    d_vector = np.array([2.0, 3.0, 4.0], dtype=np.float64)

    h_matrix, contrib = math_test.symbolic_H_data_simple_pattern_plan(matrix)
    returned = math_test.update_H_data_simple_planned(matrix, d_vector, h_matrix, contrib)
    expected = np.tril(matrix.T.toarray() @ np.diag(d_vector) @ matrix.toarray())

    assert returned is h_matrix
    assert h_matrix.shape == (matrix.shape[1], matrix.shape[1])
    assert contrib["entry_ptr"].size == h_matrix.nnz + 1
    np.testing.assert_allclose(h_matrix.toarray(), expected)


def test_get_H_sparsity_keeps_structural_entries_when_unweighted_values_cancel():
    from secore.se_math_test import SEMathTest

    math_test = SEMathTest()
    matrix = csc_matrix(np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.float64))
    d_vector = np.array([1.0, 2.0], dtype=np.float64)
    h_matrix = math_test.get_H_sparsity(matrix)
    expected = np.tril(matrix.T.toarray() @ np.diag(d_vector) @ matrix.toarray())

    math_test.update_H_data(h_matrix, matrix, d_vector)

    np.testing.assert_allclose(h_matrix.toarray(), expected)


def test_cholmod_solves_lower_triangular_stored_h_as_symmetric_system():
    import pytest
    import secore.se_math_test as se_math_test

    if se_math_test.CHOLMOD_ANALYZE is None:
        pytest.skip("sksparse.cholmod is not available")

    math_test = se_math_test.SEMathTest()
    matrix = csc_matrix(
        np.array(
            [
                [1.0, 0.0],
                [0.0, 2.0],
                [3.0, 4.0],
            ],
            dtype=np.float64,
        )
    )
    d_vector = np.array([5.0, 6.0, 7.0], dtype=np.float64)
    rhs = np.array([1.0, 2.0], dtype=np.float64)
    h_lower = math_test.get_H_sparsity(matrix)
    math_test.update_H_data(h_lower, matrix, d_vector)
    h_full = matrix.T.toarray() @ np.diag(d_vector) @ matrix.toarray()

    assert h_lower.nnz < np.count_nonzero(h_full)
    np.testing.assert_allclose(np.tril(h_full), h_lower.toarray())

    factor = math_test.cholmod_analyze(h_lower)
    math_test.cholmod_factorize_inplace(factor, h_lower)
    result = math_test.solve(factor, rhs)

    np.testing.assert_allclose(result, np.linalg.solve(h_full, rhs), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(h_full @ result, rhs, rtol=1e-12, atol=1e-12)


def test_main2_reads_13_matrix_pairs_and_reuses_h_sparsity_analysis(tmp_path):
    from secore.se_math_test import SEMathTest

    for index in range(1, 14):
        _write_triplet(
            tmp_path / f"j{index}.txt",
            3,
            2,
            "jacobian",
            [(1, 1, 1.0 + index), (2, 2, 2.0 + index), (3, 1, 3.0 + index)],
        )
        _write_triplet(
            tmp_path / f"d{index}.txt",
            3,
            3,
            "weight_diagonal",
            [(1, 1, 1.0), (2, 2, 2.0), (3, 3, 3.0)],
        )

    class FakeFactor:
        def __init__(self):
            self.matrix = None

        def cholesky_inplace(self, matrix):
            self.matrix = matrix.copy()

        def __call__(self, rhs):
            return np.asarray(rhs, dtype=np.float64) + self.matrix.diagonal()

    class RecordingSEMathTest(SEMathTest):
        def __init__(self):
            super().__init__()
            self.get_sparsity_calls = []
            self.analyze_calls = []
            self.update_h_calls = []
            self.factorize_inplace_calls = []
            self.solve_calls = []

        def get_H_sparsity(self, matrix):
            self.get_sparsity_calls.append(matrix.copy())
            return csc_matrix(np.eye(matrix.shape[1], dtype=np.float64))

        def update_H_data(self, h_matrix, matrix, d_vector):
            self.update_h_calls.append((matrix.data.copy(), d_vector.copy()))
            h_matrix.data[:] = len(self.update_h_calls)
            return h_matrix

        def cholmod_analyze(self, matrix):
            self.analyze_calls.append(matrix.copy())
            return FakeFactor()

        def cholmod_factorize_inplace(self, factor, matrix):
            self.factorize_inplace_calls.append(matrix.copy())
            return super().cholmod_factorize_inplace(factor, matrix)

        def solve(self, factor, rhs):
            self.solve_calls.append(np.asarray(rhs).copy())
            return super().solve(factor, rhs)

    math_test = RecordingSEMathTest()
    rhs = np.array([10.0, 20.0], dtype=np.float64)

    results = math_test.main2(tmp_path, rhs=rhs)

    assert len(results) == 13
    assert len(math_test.get_sparsity_calls) == 1
    assert len(math_test.analyze_calls) == 1
    assert len(math_test.update_h_calls) == 13
    assert len(math_test.factorize_inplace_calls) == 13
    assert len(math_test.solve_calls) == 13
    np.testing.assert_allclose(results[0], np.array([11.0, 21.0]))
    np.testing.assert_allclose(results[-1], np.array([23.0, 33.0]))


def test_main2_loads_files_once_then_computes_from_memory(tmp_path):
    from secore.se_math_test import SEMathTest

    for index in range(1, 4):
        _write_triplet(
            tmp_path / f"j{index}.txt",
            3,
            2,
            "jacobian",
            [(1, 1, 1.0 + index), (2, 2, 2.0 + index), (3, 1, 3.0 + index)],
        )
        _write_triplet(
            tmp_path / f"d{index}.txt",
            3,
            3,
            "weight_diagonal",
            [(1, 1, 1.0), (2, 2, 2.0), (3, 3, 3.0)],
        )

    class FakeFactor:
        def __init__(self):
            self.matrix = None

        def cholesky_inplace(self, matrix):
            self.matrix = matrix.copy()

        def __call__(self, rhs):
            return np.asarray(rhs, dtype=np.float64) + self.matrix.diagonal()

    class RecordingSEMathTest(SEMathTest):
        def __init__(self):
            super().__init__()
            self.events = []

        def read_j_matrix(self, directory, filename):
            self.events.append(("read_j", filename))
            return super().read_j_matrix(directory, filename)

        def read_d_vector(self, directory, filename):
            self.events.append(("read_d", filename))
            return super().read_d_vector(directory, filename)

        def get_H_sparsity(self, matrix):
            self.events.append(("sparsity", matrix.shape))
            return csc_matrix(np.eye(matrix.shape[1], dtype=np.float64))

        def update_H_data(self, h_matrix, matrix, d_vector):
            self.events.append(("update_h", matrix.shape))
            h_matrix.data[:] = 1.0
            return h_matrix

        def cholmod_analyze(self, matrix):
            self.events.append(("analyze", matrix.shape))
            return FakeFactor()

    math_test = RecordingSEMathTest()
    results = math_test.main2(tmp_path, iterations=3, rhs=np.array([1.0, 2.0]))
    first_compute_event = next(
        pos for pos, event in enumerate(math_test.events) if event[0] == "sparsity"
    )

    assert len(results) == 3
    assert [event[0] for event in math_test.events[:first_compute_event]] == [
        "read_j",
        "read_d",
        "read_j",
        "read_d",
        "read_j",
        "read_d",
    ]
    assert not any(event[0].startswith("read_") for event in math_test.events[first_compute_event:])


def test_main3_reads_13_matrix_pairs_and_uses_simple_h_update(tmp_path):
    from secore.se_math_test import SEMathTest

    for index in range(1, 14):
        _write_triplet(
            tmp_path / f"j{index}.txt",
            3,
            2,
            "jacobian",
            [(1, 1, 1.0 + index), (2, 2, 2.0 + index), (3, 1, 3.0 + index)],
        )
        _write_triplet(
            tmp_path / f"d{index}.txt",
            3,
            3,
            "weight_diagonal",
            [(1, 1, 1.0), (2, 2, 2.0), (3, 3, 3.0)],
        )

    class FakeFactor:
        def __init__(self):
            self.matrix = None

        def cholesky_inplace(self, matrix):
            self.matrix = matrix.copy()

        def __call__(self, rhs):
            return np.asarray(rhs, dtype=np.float64) + self.matrix.diagonal()

    class RecordingSEMathTest(SEMathTest):
        def __init__(self):
            super().__init__()
            self.get_sparsity_calls = []
            self.simple_update_calls = []
            self.regular_update_calls = []
            self.factorize_inplace_calls = []
            self.solve_calls = []

        def get_H_sparsity(self, matrix):
            self.get_sparsity_calls.append(matrix.copy())
            return csc_matrix(np.eye(matrix.shape[1], dtype=np.float64))

        def update_H_data_simple(self, matrix, d_vector, h_matrix):
            self.simple_update_calls.append((matrix.data.copy(), d_vector.copy()))
            h_matrix.data[:] = len(self.simple_update_calls)
            return h_matrix

        def update_H_data(self, h_matrix, matrix, d_vector):
            self.regular_update_calls.append((matrix.data.copy(), d_vector.copy()))
            return h_matrix

        def cholmod_analyze(self, matrix):
            return FakeFactor()

        def cholmod_factorize_inplace(self, factor, matrix):
            self.factorize_inplace_calls.append(matrix.copy())
            return super().cholmod_factorize_inplace(factor, matrix)

        def solve(self, factor, rhs):
            self.solve_calls.append(np.asarray(rhs).copy())
            return super().solve(factor, rhs)

    math_test = RecordingSEMathTest()
    rhs = np.array([10.0, 20.0], dtype=np.float64)

    results = math_test.main3(tmp_path, rhs=rhs)

    assert len(results) == 13
    assert len(math_test.get_sparsity_calls) == 1
    assert len(math_test.simple_update_calls) == 13
    assert not math_test.regular_update_calls
    assert len(math_test.factorize_inplace_calls) == 13
    assert len(math_test.solve_calls) == 13
    np.testing.assert_allclose(results[0], np.array([11.0, 21.0]))
    np.testing.assert_allclose(results[-1], np.array([23.0, 33.0]))


def test_main4_reads_13_matrix_pairs_and_uses_planned_h_update(tmp_path):
    from secore.se_math_test import SEMathTest

    for index in range(1, 14):
        _write_triplet(
            tmp_path / f"j{index}.txt",
            3,
            2,
            "jacobian",
            [(1, 1, 1.0 + index), (2, 2, 2.0 + index), (3, 1, 3.0 + index)],
        )
        _write_triplet(
            tmp_path / f"d{index}.txt",
            3,
            3,
            "weight_diagonal",
            [(1, 1, 1.0), (2, 2, 2.0), (3, 3, 3.0)],
        )

    class FakeFactor:
        def __init__(self):
            self.matrix = None

        def cholesky_inplace(self, matrix):
            self.matrix = matrix.copy()

        def __call__(self, rhs):
            return np.asarray(rhs, dtype=np.float64) + self.matrix.diagonal()

    class RecordingSEMathTest(SEMathTest):
        def __init__(self):
            super().__init__()
            self.get_sparsity_calls = []
            self.plan_calls = []
            self.combined_plan_calls = []
            self.planned_update_calls = []
            self.simple_update_calls = []
            self.regular_update_calls = []
            self.factorize_inplace_calls = []
            self.solve_calls = []

        def get_H_sparsity(self, matrix):
            self.get_sparsity_calls.append(matrix.copy())
            return csc_matrix(np.eye(matrix.shape[1], dtype=np.float64))

        def symbolic_H_data_simple_pattern_plan(self, matrix):
            self.combined_plan_calls.append(matrix.copy())
            return csc_matrix(np.eye(matrix.shape[1], dtype=np.float64)), {"token": "combined"}

        def symbolic_H_data_simple_plan(self, matrix, h_matrix):
            self.plan_calls.append((matrix.copy(), h_matrix.copy()))
            return {"token": "planned"}

        def update_H_data_simple_planned(self, matrix, d_vector, h_matrix, contrib):
            self.planned_update_calls.append((matrix.data.copy(), d_vector.copy(), contrib))
            h_matrix.data[:] = len(self.planned_update_calls)
            return h_matrix

        def update_H_data_simple(self, matrix, d_vector, h_matrix):
            self.simple_update_calls.append((matrix.data.copy(), d_vector.copy()))
            return h_matrix

        def update_H_data(self, h_matrix, matrix, d_vector):
            self.regular_update_calls.append((matrix.data.copy(), d_vector.copy()))
            return h_matrix

        def cholmod_analyze(self, matrix):
            return FakeFactor()

        def cholmod_factorize_inplace(self, factor, matrix):
            self.factorize_inplace_calls.append(matrix.copy())
            return super().cholmod_factorize_inplace(factor, matrix)

        def solve(self, factor, rhs):
            self.solve_calls.append(np.asarray(rhs).copy())
            return super().solve(factor, rhs)

    math_test = RecordingSEMathTest()
    rhs = np.array([10.0, 20.0], dtype=np.float64)

    results = math_test.main4(tmp_path, rhs=rhs)

    assert len(results) == 13
    assert not math_test.get_sparsity_calls
    assert not math_test.plan_calls
    assert len(math_test.combined_plan_calls) == 1
    assert len(math_test.planned_update_calls) == 13
    assert not math_test.simple_update_calls
    assert not math_test.regular_update_calls
    assert len(math_test.factorize_inplace_calls) == 13
    assert len(math_test.solve_calls) == 13
    np.testing.assert_allclose(results[0], np.array([11.0, 21.0]))
    np.testing.assert_allclose(results[-1], np.array([23.0, 33.0]))


def test_main5_reuses_prepared_plan_and_cached_weight_update(tmp_path):
    from secore.se_math_test import SEMathTest

    for index in range(1, 4):
        _write_triplet(
            tmp_path / f"j{index}.txt",
            3,
            2,
            "jacobian",
            [(1, 1, 1.0 + index), (2, 2, 2.0 + index), (3, 1, 3.0 + index)],
        )
        _write_triplet(
            tmp_path / f"d{index}.txt",
            3,
            3,
            "weight_diagonal",
            [(1, 1, 1.0), (2, 2, 2.0), (3, 3, 3.0)],
        )

    class FakeFactor:
        def __init__(self):
            self.matrix = None

        def cholesky_inplace(self, matrix):
            self.matrix = matrix.copy()

        def __call__(self, rhs):
            return np.asarray(rhs, dtype=np.float64) + self.matrix.diagonal()

    class RecordingSEMathTest(SEMathTest):
        def __init__(self):
            super().__init__()
            self.combined_plan_calls = []
            self.analyze_calls = []
            self.weighted_update_calls = []
            self.regular_update_calls = []

        def symbolic_H_data_simple_pattern_plan(self, matrix):
            self.combined_plan_calls.append(matrix.copy())
            contrib = {
                "K": np.array([0, 1], dtype=np.int64),
                "A_shape": tuple(matrix.shape),
                "H_shape": (matrix.shape[1], matrix.shape[1]),
                "H_nnz": matrix.shape[1],
            }
            return csc_matrix(np.eye(matrix.shape[1], dtype=np.float64)), contrib

        def cholmod_analyze(self, matrix):
            self.analyze_calls.append(matrix.copy())
            return FakeFactor()

        def update_H_data_simple_planned_weighted(self, matrix, h_matrix, contrib, weight_by_contrib):
            self.weighted_update_calls.append(weight_by_contrib.copy())
            h_matrix.data[:] = len(self.weighted_update_calls)
            return h_matrix

        def update_H_data_simple_planned(self, matrix, d_vector, h_matrix, contrib):
            self.regular_update_calls.append(d_vector.copy())
            return h_matrix

    matrix_lists = SEMathTest().read_file(tmp_path, iterations=3)
    math_test = RecordingSEMathTest()
    prepared = math_test.main5_prepare(matrix_lists[0][0], matrix_lists[1][0])
    results = math_test.main5(matrix_lists=matrix_lists, iterations=3, rhs=np.array([10.0, 20.0]), prepared=prepared)

    assert len(results) == 3
    assert len(math_test.combined_plan_calls) == 1
    assert len(math_test.analyze_calls) == 1
    assert len(math_test.weighted_update_calls) == 3
    assert not math_test.regular_update_calls
    np.testing.assert_allclose(prepared["weight_by_contrib"], np.array([1.0, 2.0]))
    np.testing.assert_allclose(results[0], np.array([11.0, 21.0]))


def test_se_math_test_public_operations_are_class_methods():
    import secore.se_math_test as se_math_test

    public_function_names = [
        "read_j_matrix",
        "read_d_vector",
        "read_h_matrix",
        "read_file",
        "load_matrix_lists",
        "update_csc_matrix_from_file",
        "cholmod_analyze",
        "cholmod_factorize_inplace",
        "cholmod_factorize",
        "cholmod_solve",
        "solve_cholmod",
        "get_H_sparsity",
        "update_H_data",
        "update_H_data_simple",
        "symbolic_H_data_simple_pattern_plan",
        "symbolic_H_data_simple_plan",
        "update_H_data_simple_planned",
        "update_H_data_simple_planned_weighted",
        "symbolic_AtBA_lower",
        "update_AtBA_lower",
        "solve",
        "main",
        "main2",
        "main3",
        "main4",
        "main5_prepare",
        "main5",
    ]

    math_test = se_math_test.SEMathTest()
    for name in public_function_names:
        assert hasattr(math_test, name)
        assert not hasattr(se_math_test, name)
