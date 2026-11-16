"""Focused contracts for equivalent object and PPC-backed LF entry points."""

from pathlib import Path
import inspect

import numpy as np


def test_lf_public_defaults_and_prepare_returns_are_parallel():
    from lfcore.ac_lf import ACPowerFlowCalc
    from lfcore.dc_lf import DCPowerFlowCalc
    from lfcore.hybrid_lf import HybridPowerFlowCalc

    classes = (ACPowerFlowCalc, DCPowerFlowCalc, HybridPowerFlowCalc)
    signatures = [inspect.signature(cls.__init__) for cls in classes]
    assert [sig.parameters["linear_solver"].default for sig in signatures] == ["pyklu", "pyklu", "pyklu"]
    assert [sig.parameters["linear_solver"].annotation for sig in signatures] == [
        signatures[0].parameters["linear_solver"].annotation,
    ] * 3

    case = Path(__file__).resolve().parents[1] / "data" / "model" / "hybrid" / "hybrid_net_40.e"
    calc = HybridPowerFlowCalc.from_file_fast(case, result_mode="array", verbose=False)
    assert calc.prepare() is None


def test_hybrid_object_and_fast_paths_keep_dcac_state_layout_consistent():
    from lfcore.hybrid_lf import HybridPowerFlowCalc, HybridPowerNetwork

    case = Path(__file__).resolve().parents[1] / "data" / "model" / "hybrid" / "hybrid_net_40.e"

    object_calc = HybridPowerFlowCalc(
        HybridPowerNetwork.read_from_file(case),
        result_mode="array",
        verbose=False,
    )
    fast_calc = HybridPowerFlowCalc.from_file_fast(
        case,
        result_mode="array",
        verbose=False,
    )

    assert object_calc.run() == 0
    assert fast_calc.run() == 0
    assert object_calc.N_dcac == fast_calc.N_dcac
    assert object_calc.total_vars == fast_calc.total_vars
    assert object_calc.last_jacobian_shape == fast_calc.last_jacobian_shape
    np.testing.assert_allclose(object_calc.x, fast_calc.x, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(object_calc.get_f(object_calc.x), fast_calc.get_f(fast_calc.x), rtol=1e-10, atol=1e-10)
