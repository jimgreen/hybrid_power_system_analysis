import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _write_tight_calmt(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "<ACNode>",
                "@ idx name v_max v_min",
                "# 0 nd_0 0.5 0.4",
                "</ACNode>",
                "",
                "<DCNode>",
                "@ idx name v_max v_min",
                "# 0 nd_1 50 40",
                "</DCNode>",
                "",
                "<ACBranch>",
                "@ idx name i_max p_max",
                "# 0 line_0_1 0.000001 0.000001",
                "</ACBranch>",
                "",
                "<DCBranch>",
                "@ idx name i_max p_max",
                "# 0 line_0_1 0.000001 0.000001",
                "</DCBranch>",
                "",
            ]
        ),
        encoding="utf8",
    )


def test_hybrid_ca_writes_risk_result_efile(tmp_path):
    from ca.hybrid_ca import run_hybrid_ca
    from efile_read import EBook

    n0_file = tmp_path / "hybrid_net_40.ca.n0rslt"
    n1_file = tmp_path / "hybrid_net_40.ca.n1rslt"
    info_file = tmp_path / "ca_calc_info.e"
    calmt_file = tmp_path / "hybrid_net_40.calmt"
    _write_tight_calmt(calmt_file)

    n0_risks, n1_risks = run_hybrid_ca(
        ROOT / "data" / "hybrid" / "hybrid_net_40.e",
        calmt_file,
        n0_file,
        n1_file,
        info_file,
        scan_n1=False,
    )

    assert n0_risks
    assert n1_risks == []
    assert n0_file.exists()
    assert n1_file.exists()
    book = EBook(n0_file)
    table = book.data["ca_risk_result"]
    assert table.header_list == ["risk_source", "risk_dist", "risk_type", "risk_value", "risk_info"]
    assert any(row["risk_source"] == "BASE" and row["risk_type"] == "越限" for row in table.data)
    assert EBook(n1_file).data["ca_risk_result"].data == []

    info = EBook(info_file).data["ca_calc_info"]
    assert info.header_list == [
        "calc_start_time",
        "calc_elapsed_seconds",
        "n0_risk_count",
        "n1_scan_count",
        "n1_risk_count",
        "risk_level",
    ]
    assert len(info.data) == 1
    assert info.data[0]["n0_risk_count"] == str(len(n0_risks))
    assert info.data[0]["n1_scan_count"] == "0"
    assert info.data[0]["risk_level"] == "中"


def test_hybrid_ca_n1_scan_includes_ac_units_and_loads():
    from ca.hybrid_ca import _contingency_rows

    contingencies = set(_contingency_rows(ROOT / "data" / "hybrid" / "qinling.e"))

    assert ("ACGenerator", "ACUnit", "wt01_10kw") in contingencies
    assert ("ACLoad", "ACLoad", "load_ac_1") in contingencies
    assert ("DCGenerator", "DCUnit", "pv01_vsrc") in contingencies


def test_ca_efile_output_columns_are_aligned(tmp_path):
    from ca.hybrid_ca import CARisk, append_calc_info, write_risk_result

    risk_file = tmp_path / "risk.e"
    write_risk_result(
        [
            CARisk("BASE", "ACNode.n1.v", "越限", 1.0, "v=1<v_min=270"),
            CARisk("N-1:ACUnit.long_name", "ACBranch.branch.p", "负荷损失", 123.456, "loss=123.456"),
        ],
        risk_file,
    )
    assert _data_column_starts(risk_file) == _header_column_starts(risk_file)

    info_file = tmp_path / "info.e"
    append_calc_info(
        info_file,
        calc_start_time="2026-05-08T22:00:00",
        calc_elapsed_seconds=1.25,
        n0_risk_count=2,
        n1_scan_count=91,
        n1_risk_count=11713,
        risk_level="高",
    )
    assert _data_column_starts(info_file) == _header_column_starts(info_file)


def _header_column_starts(path: Path) -> list[int]:
    header = next(line for line in path.read_text(encoding="utf8").splitlines() if line.startswith("@"))
    return _token_starts(header, 2)


def _data_column_starts(path: Path) -> list[int]:
    row = next(line for line in path.read_text(encoding="utf8").splitlines() if line.startswith("#"))
    return _token_starts(row, 2)


def _token_starts(line: str, offset: int) -> list[int]:
    starts = []
    in_token = False
    for idx, char in enumerate(line[offset:], start=offset):
        if char.isspace():
            in_token = False
        elif not in_token:
            starts.append(idx)
            in_token = True
    return starts
