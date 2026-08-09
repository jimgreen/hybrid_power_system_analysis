from pathlib import Path

import numpy as np
import pytest

from efile_read import EBook
from lfcore.ac_lf import ACPowerFlowCalc, load_ac_ppc_from_e_file
from lfcore.dc_lf import DCPowerFlowCalc, load_dc_ppc_from_e_file
from lfcore.hybrid_lf import HybridPowerFlowCalc, _read_lf_network_from_file
from secore.ac_se import ACStateEstimator
from secore.dc_se import DCStateEstimator
from secore.hybrid_se import HybridStateEstimator


ROOT = Path(__file__).resolve().parents[1]
CASES = (
    (
        "ac",
        ROOT / "data" / "model" / "ac" / "ac_net_30.e",
        ROOT / "data" / "meas" / "ac" / "ac_net_30.meas",
    ),
    (
        "dc",
        ROOT / "data" / "model" / "dc" / "dc_net_30.e",
        ROOT / "data" / "meas" / "dc" / "dc_net_30.meas",
    ),
    (
        "hybrid",
        ROOT / "data" / "model" / "hybrid" / "hybrid_net_40.e",
        ROOT / "data" / "meas" / "hybrid" / "hybrid_net_40.meas",
    ),
)


def _renamed_case(model_path: Path, meas_path: Path, target_dir: Path) -> tuple[Path, Path]:
    model_book = EBook(model_path)
    renamed_devices = {}
    for table_name, block in model_book.data.items():
        for row in block.data:
            if "idx" not in row or "name" not in row:
                continue
            old_name = str(row["name"])
            new_name = f"device_{table_name}_{int(float(row['idx']))}"
            renamed_devices[(table_name, old_name)] = new_name
            row["name"] = new_name

    for block in model_book.data.values():
        for row in block.data:
            key = (str(row.get("dev_type", "")), str(row.get("dev_name", "")))
            if key in renamed_devices:
                row["dev_name"] = renamed_devices[key]

    renamed_model = target_dir / model_path.name
    model_book.apply_to_file(renamed_model)

    meas_book = EBook(meas_path)
    for row in meas_book.data["Measurement"].data:
        key = (str(row["dev_type"]), str(row["dev_name"]))
        assert key in renamed_devices
        row["dev_name"] = renamed_devices[key]
        row["name"] = f"measurement_{int(float(row['idx']))}"

    renamed_meas = target_dir / meas_path.name
    meas_book.apply_to_file(renamed_meas)
    return renamed_model, renamed_meas


def _run_lf(kind: str, model_path: Path) -> np.ndarray:
    if kind == "ac":
        calc = ACPowerFlowCalc(
            load_ac_ppc_from_e_file(model_path),
            linear_solver="scipy",
            result_mode="array",
            verbose=False,
        )
    elif kind == "dc":
        calc = DCPowerFlowCalc(
            load_dc_ppc_from_e_file(model_path),
            linear_solver="scipy",
            result_mode="array",
            verbose=False,
        )
    else:
        calc = HybridPowerFlowCalc(
            _read_lf_network_from_file(model_path),
            linear_solver="scipy",
            result_mode="array",
            verbose=False,
        )

    assert calc.run() == 0
    assert calc.converged
    return calc.x.copy()


def _run_se(kind: str, model_path: Path, meas_path: Path):
    estimator_class = {
        "ac": ACStateEstimator,
        "dc": DCStateEstimator,
        "hybrid": HybridStateEstimator,
    }[kind]
    estimator = estimator_class(
        e_file=model_path,
        meas_file=meas_path,
        flat_start=True,
        max_iter=50,
    )
    estimator.run(
        result_mode="array",
        skip_bad_data=True,
        verbose=False,
        final_diagnostics=False,
    )
    result = estimator.estimate_result
    assert result is not None
    assert result.converged
    return result.x.copy(), result.objective


@pytest.mark.parametrize(("kind", "model_path", "meas_path"), CASES)
def test_lf_and_se_do_not_infer_behavior_from_device_names(
    tmp_path: Path,
    kind: str,
    model_path: Path,
    meas_path: Path,
):
    renamed_model, renamed_meas = _renamed_case(model_path, meas_path, tmp_path)

    np.testing.assert_allclose(
        _run_lf(kind, renamed_model),
        _run_lf(kind, model_path),
        rtol=0.0,
        atol=1e-11,
    )
    renamed_x, renamed_objective = _run_se(kind, renamed_model, renamed_meas)
    original_x, original_objective = _run_se(kind, model_path, meas_path)
    np.testing.assert_allclose(renamed_x, original_x, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(
        renamed_objective,
        original_objective,
        rtol=0.0,
        atol=1e-10,
    )
