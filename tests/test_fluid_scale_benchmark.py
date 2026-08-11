import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_fluid_scale_lf_se.py"


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location("fluid_scale_benchmark", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("network_type", ("heat", "gas", "hydro", "steam"))
def test_generated_fluid_scale_case_runs_lf_and_se_consistently(tmp_path, network_type):
    benchmark = _load_benchmark_module()
    case_file = tmp_path / f"{network_type}_10.e"
    benchmark.generate_case(network_type, 10, case_file)

    loader, _, _ = benchmark.NETWORK_RUNTIME[network_type]
    network = loader(case_file)
    assert len(network.nodes) == 10
    assert len(network.sources) >= 1
    assert len(network.loads) >= 1
    assert np.count_nonzero(network.edge_kind == "pipe") >= 1
    assert np.count_nonzero(network.edge_kind == "valve") >= 1
    assert np.count_nonzero(~np.isin(network.edge_kind, ("pipe", "valve"))) >= 1
    if network_type == "heat":
        assert network.mixed_return
        assert len(network.heat_exchangers) == 1
        np.testing.assert_array_equal(network.exchanger_primary_explicit, [True])
        np.testing.assert_array_equal(network.exchanger_secondary_explicit, [False])

    result = benchmark.benchmark_case(network_type, 10, case_file)
    assert result["lf"]["converged"]
    assert result["se"]["converged"]
    assert result["se"]["rank"] == result["se"]["states"]
    assert result["se"]["pseudo_measurements"] == 0
    assert result["se"]["bad_data"] == 0
    assert max(result["accuracy"].values()) < 2e-9
    assert case_file.with_suffix(".meas").is_file()
