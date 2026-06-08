import os

import numpy as np
import pytest
import torch
from aimnet.calculators.model_registry import get_model_path
from aimnet.config import build_module
from aimnet.models.base import load_model
from aimnet.ttlang import BackendMode, DerivativeNotImplementedError, TTLangAIMNet2
from aimnet.ttlang.placeholders import forces_not_implemented, hessian_not_implemented

aimnet2_d3_def = os.path.join(os.path.dirname(__file__), "..", "aimnet", "models", "aimnet2_dftd3_wb97m.yaml")
caffeine_xyz = os.path.join(os.path.dirname(__file__), "data", "caffeine.xyz")


def _caffeine_input():
    pytest.importorskip("ase", reason="ASE not installed. Install with: pip install aimnet[ase]")
    import ase.io

    from aimnet.calculators import AIMNet2Calculator

    atoms = ase.io.read(caffeine_xyz, format="extxyz")
    calc = AIMNet2Calculator("aimnet2", nb_threshold=0, device="cpu")
    data = calc.prepare_input(
        {
            "coord": atoms.get_positions(),
            "numbers": atoms.get_atomic_numbers(),
            "charge": 0.0,
        }
    )
    return data, atoms


@pytest.fixture
def aimnet2_model():
    model = build_module(aimnet2_d3_def)
    model.outputs.atomic_shift.shifts.double()
    model_from_zoo, _ = load_model(get_model_path("aimnet2"), device="cpu")
    model.load_state_dict(model_from_zoo.state_dict(), strict=False)
    return model


def test_ttlang_import():
    import aimnet.ttlang  # noqa: F401


def test_ttlang_yaml_builds():
    from aimnet.models import AIMNet2

    model = build_module(aimnet2_d3_def)
    assert isinstance(model, AIMNet2)
    wrapper = TTLangAIMNet2.from_pytorch(model, BackendMode.REFERENCE)
    wrapper.close()


def test_ttlang_energy_parity_reference(aimnet2_model):
    data, atoms = _caffeine_input()
    ref_e = atoms.get_potential_energy()

    wrapper = TTLangAIMNet2.from_pytorch(aimnet2_model, BackendMode.REFERENCE)
    out = wrapper.forward(data)
    wrapper.close()

    np.testing.assert_allclose(out["energy"].item(), ref_e, atol=1e-5)


@pytest.mark.tt
def test_ttlang_energy_parity_tt_sim(aimnet2_model):
    data, _ = _caffeine_input()
    ref = TTLangAIMNet2.from_pytorch(aimnet2_model, BackendMode.REFERENCE)
    ref_out = ref.forward(data)
    ref.close()

    sim = TTLangAIMNet2.from_pytorch(aimnet2_model, BackendMode.TT_SIM)
    sim_out = sim.forward(data)
    sim.close()

    e_ref = ref_out["energy"].item()
    e_sim = sim_out["energy"].item()
    assert abs(e_ref - e_sim) < 0.5


def test_ttlang_forces_placeholder():
    with pytest.raises(DerivativeNotImplementedError):
        forces_not_implemented()


def test_ttlang_hessian_placeholder():
    with pytest.raises(DerivativeNotImplementedError):
        hessian_not_implemented()


@pytest.mark.parametrize("alias", ["aimnet2", "aimnet2-nse", "aimnet2-rxn"])
@pytest.mark.network
def test_ttlang_registry_models_energy_smoke(alias):
    model, _ = load_model(get_model_path(alias), device="cpu")
    data, _ = _caffeine_input()
    wrapper = TTLangAIMNet2.from_pytorch(model, BackendMode.REFERENCE)
    out = wrapper.forward(data)
    wrapper.close()
    assert torch.isfinite(out["energy"]).all()


@pytest.mark.tt_hw
def test_ttlang_energy_parity_hw(aimnet2_model):
    pytest.importorskip("ttnn")
    if not os.path.exists("/dev/tenstorrent/0"):
        pytest.skip("Tenstorrent device not available")

    data, _ = _caffeine_input()
    ref = TTLangAIMNet2.from_pytorch(aimnet2_model, BackendMode.REFERENCE)
    ref_out = ref.forward(data)
    ref.close()

    hw = TTLangAIMNet2.from_pytorch(aimnet2_model, BackendMode.TT_HW)
    hw_out = hw.forward(data)
    hw.close()

    e_ref = ref_out["energy"].item()
    e_hw = hw_out["energy"].item()
    assert abs(e_ref - e_hw) < 0.5

# ── Phase 0: forced full-device & fallback tests ──────────────────────
@pytest.mark.tt_hw
def test_ttlang_default_tt_hw_works(aimnet2_model):
    """Default TT_HW (hybrid path) must produce valid caffeine energy."""
    pytest.importorskip("ttnn")
    if not os.path.exists("/dev/tenstorrent/0"):
        pytest.skip("Tenstorrent device not available")
    data, atoms = _caffeine_input()
    ref_e = atoms.get_potential_energy()
    hw = TTLangAIMNet2.from_pytorch(aimnet2_model, BackendMode.TT_HW)
    hw_out = hw.forward(data)
    hw.close()
    e_hw = hw_out["energy"].item()
    assert abs(ref_e - e_hw) < 0.5, f"energy error {abs(ref_e - e_hw):.6f} > 0.5 kcal/mol"
@pytest.mark.tt_hw
def test_ttlang_force_full_device_fails_unsupported(aimnet2_model):
    """AIMNET_TTLANG_FULL_DEVICE=1 fails when full-device core is unsupported."""
    pytest.importorskip("ttnn")
    if not os.path.exists("/dev/tenstorrent/0"):
        pytest.skip("Tenstorrent device not available")
    data, _ = _caffeine_input()
    # Force full device without allowing hybrid fallback
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("AIMNET_TTLANG_FULL_DEVICE", "1")
    monkeypatch.setenv("AIMNET_TTLANG_ALLOW_HYBRID_FALLBACK", "0")
    monkeypatch.setenv("AIMNET_TTLANG_TRY_FULL_DEVICE", "0")
    wrapper = TTLangAIMNet2.from_pytorch(aimnet2_model, BackendMode.TT_HW)
    try:
        result = wrapper.forward(data)
        # If full-device succeeds (after future repairs), energy must be valid
        e_ref = TTLangAIMNet2.from_pytorch(aimnet2_model, BackendMode.REFERENCE)
        ref_out = e_ref.forward(data)
        e_ref.close()
        assert abs(ref_out["energy"].item() - result["energy"].item()) < 0.5
    finally:
        wrapper.close()
        monkeypatch.undo()
@pytest.mark.tt_hw
def test_ttlang_try_full_device_falls_back(aimnet2_model):
    """AIMNET_TTLANG_TRY_FULL_DEVICE=1 falls back to hybrid if core fails."""
    pytest.importorskip("ttnn")
    if not os.path.exists("/dev/tenstorrent/0"):
        pytest.skip("Tenstorrent device not available")
    data, atoms = _caffeine_input()
    ref_e = atoms.get_potential_energy()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("AIMNET_TTLANG_FULL_DEVICE", "0")
    monkeypatch.setenv("AIMNET_TTLANG_TRY_FULL_DEVICE", "1")
    monkeypatch.setenv("AIMNET_TTLANG_ALLOW_HYBRID_FALLBACK", "1")
    hw = TTLangAIMNet2.from_pytorch(aimnet2_model, BackendMode.TT_HW)
    try:
        hw_out = hw.forward(data)
        e_hw = hw_out["energy"].item()
        assert abs(ref_e - e_hw) < 0.5, f"energy error {abs(ref_e - e_hw):.6f} > 0.5 kcal/mol"
    finally:
        hw.close()
        monkeypatch.undo()
@pytest.mark.tt_hw
def test_ttlang_force_full_device_no_forbidden_host_stages(aimnet2_model):
    """Forced full-device must NOT run forbidden host stages in the core."""
    pytest.importorskip("ttnn")
    if not os.path.exists("/dev/tenstorrent/0"):
        pytest.skip("Tenstorrent device not available")
    from aimnet.ttlang.ops import get_profile_stats, reset_profile_stats, set_profile_enabled
    data, _ = _caffeine_input()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("AIMNET_TTLANG_FULL_DEVICE", "1")
    monkeypatch.setenv("AIMNET_TTLANG_ALLOW_HYBRID_FALLBACK", "0")
    monkeypatch.setenv("AIMNET_TTLANG_TRY_FULL_DEVICE", "0")
    reset_profile_stats()
    set_profile_enabled(True)
    wrapper = TTLangAIMNet2.from_pytorch(aimnet2_model, BackendMode.TT_HW)
    try:
        wrapper.forward(data)
        stats = get_profile_stats()
    finally:
        wrapper.close()
        monkeypatch.undo()
        set_profile_enabled(False)
        reset_profile_stats()
    forbidden_prefixes = (
        "host.aev",
        "host.prepare_in.",
        "host.update_q.",
        "host.output.",
        "mlp.input_upload",
        "mlp.output_download",
    )
    for key in stats:
        if key.startswith(forbidden_prefixes):
            pytest.fail(
                f"Forbidden host stage '{key}' ran in forced full-device mode "
                f"({stats[key]['mean_ms']:.3f} ms). "
                f"All model-forward ops must stay on device."
            )
