from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional

from efile_read import EBook


def _find_project_root() -> Path:
    for path in Path(__file__).resolve().parents:
        if (path / "pyproject.toml").exists():
            return path
    return Path(__file__).resolve().parent


ROOT_DIR = _find_project_root()
DEFAULT_LF_PARAMETER_FILE = ROOT_DIR / "lf.para"
DEFAULT_SE_PARAMETER_FILE = ROOT_DIR / "se.para"


def _to_bool(value: Any) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean parameter value: {value}")


def _load_key_value_block(path: Path, block_name: str) -> Dict[str, str]:
    data = EBook(path).to_dict()
    if block_name not in data:
        raise RuntimeError(f"{path} does not contain a <{block_name}> block")
    values = {}
    for row in data[block_name]["data"]:
        values[str(row["name"])] = str(row["value"])
    return values


def _require(values: Dict[str, str], key: str) -> str:
    if key not in values:
        raise RuntimeError(f"Missing required algorithm parameter: {key}")
    return values[key]


def _override_value(current, value):
    return current if value is None else value


@dataclass(frozen=True)
class PowerFlowParameters:
    tol: float
    max_iter: int
    min_voltage: float
    divergence_threshold: float

    def with_overrides(
        self,
        tol: Optional[float] = None,
        max_iter: Optional[int] = None,
        min_voltage: Optional[float] = None,
        divergence_threshold: Optional[float] = None,
    ) -> "PowerFlowParameters":
        return replace(
            self,
            tol=float(_override_value(self.tol, tol)),
            max_iter=int(_override_value(self.max_iter, max_iter)),
            min_voltage=float(_override_value(self.min_voltage, min_voltage)),
            divergence_threshold=float(_override_value(self.divergence_threshold, divergence_threshold)),
        )


@dataclass(frozen=True)
class StateEstimationParameters:
    tol: float
    max_iter: int
    diff_step: float
    flat_start: bool
    bad_threshold: float
    max_remove: int
    pseudo_measurement_weight: float
    targeted_pseudo_measurement_max: int
    targeted_pseudo_measurement_redundancy_ratio: float
    targeted_pseudo_measurement_step: int
    voltage_floor: float
    min_current_voltage: float
    power_flow_tol: float
    power_flow_max_iter: int
    power_flow_min_voltage: float

    def with_overrides(
        self,
        tol: Optional[float] = None,
        max_iter: Optional[int] = None,
        diff_step: Optional[float] = None,
        flat_start: Optional[bool] = None,
        bad_threshold: Optional[float] = None,
        max_remove: Optional[int] = None,
        targeted_pseudo_measurement_max: Optional[int] = None,
        targeted_pseudo_measurement_redundancy_ratio: Optional[float] = None,
        targeted_pseudo_measurement_step: Optional[int] = None,
    ) -> "StateEstimationParameters":
        return replace(
            self,
            tol=float(_override_value(self.tol, tol)),
            max_iter=int(_override_value(self.max_iter, max_iter)),
            diff_step=float(_override_value(self.diff_step, diff_step)),
            flat_start=bool(_override_value(self.flat_start, flat_start)),
            bad_threshold=float(_override_value(self.bad_threshold, bad_threshold)),
            max_remove=int(_override_value(self.max_remove, max_remove)),
            targeted_pseudo_measurement_max=int(
                _override_value(self.targeted_pseudo_measurement_max, targeted_pseudo_measurement_max)
            ),
            targeted_pseudo_measurement_redundancy_ratio=float(
                _override_value(
                    self.targeted_pseudo_measurement_redundancy_ratio,
                    targeted_pseudo_measurement_redundancy_ratio,
                )
            ),
            targeted_pseudo_measurement_step=int(
                _override_value(self.targeted_pseudo_measurement_step, targeted_pseudo_measurement_step)
            ),
        )


def load_lf_parameters(path: Optional[Path] = None) -> PowerFlowParameters:
    parameter_file = Path(path) if path is not None else DEFAULT_LF_PARAMETER_FILE
    values = _load_key_value_block(parameter_file, "PowerFlowParameter")
    return PowerFlowParameters(
        tol=float(_require(values, "tol")),
        max_iter=int(_require(values, "max_iter")),
        min_voltage=float(_require(values, "min_voltage")),
        divergence_threshold=float(_require(values, "divergence_threshold")),
    )


def load_se_parameters(path: Optional[Path] = None) -> StateEstimationParameters:
    parameter_file = Path(path) if path is not None else DEFAULT_SE_PARAMETER_FILE
    values = _load_key_value_block(parameter_file, "StateEstimationParameter")
    return StateEstimationParameters(
        tol=float(_require(values, "tol")),
        max_iter=int(_require(values, "max_iter")),
        diff_step=float(_require(values, "diff_step")),
        flat_start=_to_bool(_require(values, "flat_start")),
        bad_threshold=float(_require(values, "bad_threshold")),
        max_remove=int(_require(values, "max_remove")),
        pseudo_measurement_weight=float(_require(values, "pseudo_measurement_weight")),
        targeted_pseudo_measurement_max=int(_require(values, "targeted_pseudo_measurement_max")),
        targeted_pseudo_measurement_redundancy_ratio=float(
            values.get("targeted_pseudo_measurement_redundancy_ratio", "0")
        ),
        targeted_pseudo_measurement_step=int(values.get("targeted_pseudo_measurement_step", "10")),
        voltage_floor=float(_require(values, "voltage_floor")),
        min_current_voltage=float(_require(values, "min_current_voltage")),
        power_flow_tol=float(_require(values, "power_flow_tol")),
        power_flow_max_iter=int(_require(values, "power_flow_max_iter")),
        power_flow_min_voltage=float(_require(values, "power_flow_min_voltage")),
    )
