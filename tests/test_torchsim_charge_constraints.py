import pytest

from aimnet.calculators import AIMNet2Calculator
from aimnet.calculators.aimnet2torchsim import AIMNet2TorchSim, state_dict_to_aimnet2_data, state_to_aimnet2_data

torch = pytest.importorskip("torch", reason="torch is required for AIMNet2TorchSim tests")
torch_sim = pytest.importorskip("torch_sim", reason="torch-sim is required for AIMNet2TorchSim tests")


def test_torchsim_forwards_region_charge_constraints_nb_mode_1():
    """
    Ensure `AIMNet2TorchSim` forwards `region_mask` / `region_charges` into
    `AIMNet2Calculator`, so region-wise charge sums match constraints.
    """

    base_calc = AIMNet2Calculator("aimnet2_2025", nb_threshold=0)
    model = AIMNet2TorchSim(base_calc)

    # Water (flat input -> nb_mode=1 in AIMNet2Calculator)
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.1173],  # O
            [0.0, 0.7572, -0.4692],  # H
            [0.0, -0.7572, -0.4692],  # H
        ],
        dtype=torch.float32,
        device=base_calc.device,
    )
    atomic_numbers = torch.tensor([8, 1, 1], dtype=torch.long, device=base_calc.device)

    # Two constrained regions: atom 0 and atoms 1-2
    region_mask = torch.tensor([0, 1, 1], dtype=torch.long, device=base_calc.device)
    region_charges = torch.tensor([-1.0, 1.0], dtype=torch.float32, device=base_calc.device)
    total_charge = region_charges.sum()

    state = {
        "positions": positions,
        "atomic_numbers": atomic_numbers,
        "charge": total_charge,
        "region_mask": region_mask,
        "region_charges": region_charges,
    }

    # Use torch-sim static wrapper instead of calling the model directly.
    # Region-constrained runs omit ``charges`` from the return dict by default (TorchSim);
    # request them explicitly for assertions.
    static_model = torch_sim.static(model)
    try:
        out = static_model(state, return_charges=True)
    except TypeError:
        # Some torch-sim versions expose `forward` on the static wrapper.
        try:
            out = static_model.forward(state, return_charges=True)
        except TypeError:
            out = model.forward(state, return_charges=True)
    charges = out["charges"]

    # Region sums should match targets within reasonable numerical tolerance.
    assert torch.isfinite(charges).all()
    assert abs(charges[0].item() - region_charges[0].item()) < 1e-3
    assert abs(charges[1:].sum().item() - region_charges[1].item()) < 1e-3
    assert abs(charges.sum().item() - total_charge.item()) < 1e-3


def test_torchsim_batched_region_charge_constraints_nb_mode_0():
    """Dense batched coord (B, N, 3): variable padded region targets per row."""
    base_calc = AIMNet2Calculator("aimnet2_2025", nb_threshold=256)
    model = AIMNet2TorchSim(base_calc)
    device = base_calc.device

    # Two waters: system 0 groups H together; system 1 uses three regions (O, H, H)
    positions = torch.tensor(
        [
            [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]],
            [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]],
        ],
        dtype=torch.float32,
        device=device,
    )
    atomic_numbers = torch.tensor([[8, 1, 1], [8, 1, 1]], dtype=torch.long, device=device)
    region_mask = torch.tensor([[0, 1, 1], [0, 1, 2]], dtype=torch.long, device=device)
    region_charges = torch.tensor(
        [[-0.2, 0.2, 0.0], [0.05, 0.02, -0.07]],
        dtype=torch.float32,
        device=device,
    )
    charge = region_charges.sum(dim=1)

    state = {
        "positions": positions,
        "atomic_numbers": atomic_numbers,
        "charge": charge,
        "region_mask": region_mask,
        "region_charge": region_charges,
    }

    out = model(state, return_charges=True)
    charges = out["charges"].view(2, 3)

    for b in range(2):
        for r in range(3):
            sel = region_mask[b] == r
            if sel.any():
                assert abs(charges[b, sel].sum().item() - region_charges[b, r].item()) < 2e-3


def test_state_dict_to_aimnet2_data_maps_spin_to_mult():
    state = {
        "positions": torch.zeros((3, 3), dtype=torch.float32),
        "atomic_numbers": torch.tensor([8, 1, 1], dtype=torch.long),
        "charge": torch.tensor(0.0, dtype=torch.float32),
        # TorchSim uses `spin`; AIMNet2 expects `mult` for NSE models.
        "spin": torch.tensor(2.0, dtype=torch.float32),
    }
    out = state_dict_to_aimnet2_data(state)  # type: ignore[arg-type]
    assert "mult" in out
    assert torch.allclose(out["mult"], state["spin"])
    assert "spin" not in out


def test_state_to_aimnet2_data_maps_spin_to_mult():
    class DummySimState:
        def __init__(self):
            self.positions = torch.zeros((3, 3), dtype=torch.float32)
            self.row_vector_cell = torch.zeros((3, 3), dtype=torch.float32)  # non-periodic
            self.atomic_numbers = torch.tensor([8, 1, 1], dtype=torch.long)
            self.charge = torch.tensor(0.0, dtype=torch.float32)
            self.spin = torch.tensor(2.0, dtype=torch.float32)
            self.system_idx = torch.zeros((3,), dtype=torch.int32)
            # region_mask/region_charges intentionally omitted

    out = state_to_aimnet2_data(DummySimState())  # type: ignore[arg-type]
    assert "mult" in out
    assert torch.allclose(out["mult"], torch.tensor(2.0))
    assert "spin" not in out
