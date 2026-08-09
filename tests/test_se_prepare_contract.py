from pathlib import Path
import sys

import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src" / "hybrid_power_system_analysis"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.mark.parametrize(
    ("estimator_type", "case_file", "meas_file", "required_attrs"),
    (
        (
            "ac",
            ROOT_DIR / "data" / "model" / "ac" / "ac_net_30.e",
            ROOT_DIR / "data" / "meas" / "ac" / "ac_net_30.meas",
            ("measurements", "active_measurements"),
        ),
        (
            "dc",
            ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
            ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas",
            ("measurements", "active_measurements"),
        ),
        (
            "hybrid",
            ROOT_DIR / "data" / "model" / "hybrid" / "hybrid_net_40.e",
            ROOT_DIR / "data" / "meas" / "hybrid" / "hybrid_net_40.meas",
            (
                "measurements",
                "active_measurements",
                "_ac_sub_estimator",
                "_dc_sub_estimator",
                "calc",
            ),
        ),
    ),
)
def test_estimators_prepare_by_default(estimator_type, case_file, meas_file, required_attrs):
    estimator_class = _estimator_class(estimator_type)

    estimator = estimator_class(
        e_file=case_file,
        meas_file=meas_file,
        flat_start=True,
    )

    assert estimator._prepared
    for attr in required_attrs:
        assert hasattr(estimator, attr), attr


@pytest.mark.parametrize(
    ("estimator_type", "case_file", "meas_file"),
    (
        (
            "ac",
            ROOT_DIR / "data" / "model" / "ac" / "ac_net_30.e",
            ROOT_DIR / "data" / "meas" / "ac" / "ac_net_30.meas",
        ),
        (
            "dc",
            ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
            ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas",
        ),
        (
            "hybrid",
            ROOT_DIR / "data" / "model" / "hybrid" / "hybrid_net_40.e",
            ROOT_DIR / "data" / "meas" / "hybrid" / "hybrid_net_40.meas",
        ),
    ),
)
def test_estimators_allow_explicit_deferred_prepare(estimator_type, case_file, meas_file):
    estimator_class = _estimator_class(estimator_type)
    estimator = estimator_class(
        e_file=case_file,
        meas_file=meas_file,
        flat_start=True,
        auto_prepare=False,
    )

    assert not estimator._prepared
    assert estimator.prepare() is estimator
    assert estimator._prepared
    assert estimator.prepare() is estimator


def _estimator_class(estimator_type):
    if estimator_type == "ac":
        from secore.ac_se import ACStateEstimator

        return ACStateEstimator
    if estimator_type == "dc":
        from secore.dc_se import DCStateEstimator

        return DCStateEstimator
    from secore.hybrid_se import HybridStateEstimator

    return HybridStateEstimator
