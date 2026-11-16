"""Shared parser numerical and batch-boundary contracts for all SE callers."""
import numpy as np
import pytest
import model.meas_array_model as parser


def compare_numeric(actual, expected):
    for key, value in expected.items():
        if isinstance(value, np.ndarray) and key not in ('name', 'device_type', 'device_name', 'meas_type'):
            np.testing.assert_array_equal(actual[key], value, err_msg=key)
            assert actual[key].dtype == value.dtype, key
    assert actual['device_name_id_by_name'] == expected['device_name_id_by_name']
    assert actual['rows_by_device_type_code'].keys() == expected['rows_by_device_type_code'].keys()
    for code, rows in expected['rows_by_device_type_code'].items():
        np.testing.assert_array_equal(actual['rows_by_device_type_code'][code], rows)


@pytest.mark.parametrize('count', [0, 1, 3, 4, 6, 7, 16385])
@pytest.mark.parametrize('matrix', [False, True])
@pytest.mark.parametrize('status', [False, True])
def test_batched_numeric_parser_matches_string_parser(tmp_path, monkeypatch, count, matrix, status):
    monkeypatch.setattr(parser, '_MEAS_PARSE_CHUNK_SIZE', 3 if count < 10 else 16384, raising=False)
    file = tmp_path/'case.meas'
    header = '@ idx name dev_type dev_name meas_type weight valid value' + (' status' if status else '')
    rows = []
    values = ['-0.0', '-1.2345678901234567e-12', '1.7976931348623157e308', '5e-324', '110.0']
    for i in range(count):
        kind, meas = [('ACNode', 'v'), ('DCLoad', 'P_LOAD'), ('Unknown', 'mystery')][i % 3]
        row = f'# {i + 100000} m{i} {kind} 节点_{i % 5} {meas} {1.25 if i % 2 else 1.0} {i % 2} {values[i % len(values)]}'
        if status and i % 3:
            row += ' ' + ['0', '2', '3'][i % 3] + ' extra'
        rows.append(row)
    file.write_text('<Measurement>\n' + header + '\n' + '\n'.join(rows) + '\n</Measurement>\n', encoding='utf8')
    actual = parser._build_meas_ppc_from_measurement_file(file, include_strings=False, include_matrix=matrix)
    expected = parser._build_meas_ppc_from_measurement_file(file, include_strings=True, include_matrix=matrix)
    compare_numeric(actual, expected)
    if not matrix:
        assert actual['meas'] is None


@pytest.mark.parametrize('field', ['idx', 'weight', 'value'])
@pytest.mark.parametrize('bad_row', [0, 3, 6])
def test_malformed_numeric_token_raises_runtime_error_in_every_batch(tmp_path, monkeypatch, field, bad_row):
    monkeypatch.setattr(parser, '_MEAS_PARSE_CHUNK_SIZE', 3, raising=False)
    file = tmp_path/'broken.meas'
    fields = ['idx', 'name', 'dev_type', 'dev_name', 'meas_type', 'weight', 'valid', 'value']
    rows = []
    for i in range(7):
        row = [str(i), 'm', 'ACNode', 'n', 'V', '1.0', '1', '1.23']
        if i == bad_row:
            row[fields.index(field)] = 'not-a-number'
        rows.append('# ' + ' '.join(row))
    file.write_text('<Measurement>\n@ ' + ' '.join(fields) + '\n' + '\n'.join(rows) + '\n</Measurement>\n')
    with pytest.raises(RuntimeError, match='Malformed Measurement row'):
        parser._build_meas_ppc_from_measurement_file_array_only(file)


def test_nonfinite_and_signed_zero_values_match_reference(tmp_path, monkeypatch):
    monkeypatch.setattr(parser, '_MEAS_PARSE_CHUNK_SIZE', 2, raising=False)
    file = tmp_path/'special.meas'
    tokens = ['nan', 'inf', '-inf', '-0.0', '+0.0', '1_000.5']
    file.write_text('<Measurement>\n@ idx name dev_type dev_name meas_type weight valid value\n' +
                    '\n'.join(f'# {i} m ACNode n V 1 1 {value}' for i, value in enumerate(tokens)) +
                    '\n</Measurement>\n')
    actual = parser._build_meas_ppc_from_measurement_file_array_only(file, include_matrix=False)
    expected = np.array([float(value) for value in tokens])
    np.testing.assert_array_equal(actual['value_array'], expected)
    np.testing.assert_array_equal(np.signbit(actual['value_array']), np.signbit(expected))


def test_underestimated_capacity_grows_at_batch_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr(parser, '_MEAS_PARSE_CHUNK_SIZE', 3, raising=False)
    monkeypatch.setattr(parser, '_initial_measurement_capacity', lambda path: 2)
    file = tmp_path/'grow.meas'
    file.write_text('<Measurement>\n@ idx name dev_type dev_name meas_type weight valid value\n' +
                    '\n'.join(f'# {i} m ACNode n V 1 1 {i}.25' for i in range(20)) +
                    '\n</Measurement>\n')
    actual = parser._build_meas_ppc_from_measurement_file_array_only(file, include_matrix=False)
    np.testing.assert_array_equal(actual['idx_array'], np.arange(20))
    np.testing.assert_array_equal(actual['value_array'], np.arange(20) + 0.25)
