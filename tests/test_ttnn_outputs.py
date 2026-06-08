from __future__ import annotations

import pytest
import torch

from aimnet.ttlang.ttnn_outputs import ttnn_output_energy

pytestmark = pytest.mark.tt


@pytest.fixture
def tt_device():
    pytest.importorskip("ttnn")
    import ttnn

    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


def _to_tt(tensor, device):
    import ttnn
    return ttnn.from_torch(
        tensor.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _to_torch(tt_tensor):
    import ttnn
    return ttnn.to_torch(tt_tensor)


def _ref_output_energy(aim, numbers, atomic_shifts, mask_i, mol_idx, nb_mode):
    shifts = atomic_shifts[numbers].squeeze(-1)
    e_atom = aim + shifts
    if mask_i is not None:
        e_atom = e_atom.masked_fill(mask_i, 0.0)
    elif nb_mode == 1:
        e_atom[-1] = 0.0

    if nb_mode in (0, 2):
        energy = e_atom.sum(dim=1)
    elif nb_mode == 1:
        out_size = int(mol_idx[-1].item()) + 1
        if e_atom.ndim == 1:
            energy = torch.zeros(out_size, dtype=e_atom.dtype, device=e_atom.device)
            idx = mol_idx
        else:
            idx = mol_idx.unsqueeze(-1).expand(-1, e_atom.shape[1])
            energy = torch.zeros(out_size, e_atom.shape[1], dtype=e_atom.dtype, device=e_atom.device)
        energy.scatter_add_(0, idx, e_atom)
    else:
        raise ValueError(f"Invalid neighbor mode: {nb_mode}")
    return energy


class TestOutputEnergy:
    def test_mode0(self, tt_device):
        B, N = 2, 4
        num_types = 10

        aim = torch.randn(B, N, dtype=torch.bfloat16)
        numbers = torch.randint(0, num_types, (B, N), dtype=torch.int32)
        atomic_shifts = torch.randn(num_types, 1, dtype=torch.bfloat16)
        mask_i = torch.zeros(B, N, dtype=torch.bool)
        mask_i[0, 2] = True
        mask_i[1, 0] = True

        aim_tt = _to_tt(aim, tt_device)
        numbers_tt = _to_tt(numbers, tt_device)
        atomic_shifts_tt = _to_tt(atomic_shifts, tt_device)
        mask_i_tt = _to_tt(mask_i, tt_device)

        energy_tt = ttnn_output_energy(
            aim_tt, numbers_tt, atomic_shifts_tt, mask_i_tt, None, 0, tt_device
        )

        energy_ref = _ref_output_energy(
            aim, numbers, atomic_shifts, mask_i, None, 0
        )

        assert torch.allclose(_to_torch(energy_tt), energy_ref, atol=1e-2, rtol=1e-2)

    def test_mode2(self, tt_device):
        B, N = 2, 4
        num_types = 10

        aim = torch.randn(B, N, dtype=torch.bfloat16)
        numbers = torch.randint(0, num_types, (B, N), dtype=torch.int32)
        atomic_shifts = torch.randn(num_types, 1, dtype=torch.bfloat16)
        mask_i = torch.zeros(B, N, dtype=torch.bool)
        mask_i[0, 2] = True
        mask_i[1, 0] = True

        aim_tt = _to_tt(aim, tt_device)
        numbers_tt = _to_tt(numbers, tt_device)
        atomic_shifts_tt = _to_tt(atomic_shifts, tt_device)
        mask_i_tt = _to_tt(mask_i, tt_device)

        energy_tt = ttnn_output_energy(
            aim_tt, numbers_tt, atomic_shifts_tt, mask_i_tt, None, 2, tt_device
        )

        energy_ref = _ref_output_energy(
            aim, numbers, atomic_shifts, mask_i, None, 2
        )

        assert torch.allclose(_to_torch(energy_tt), energy_ref, atol=1e-2, rtol=1e-2)

    def test_mode1(self, tt_device):
        n_atoms = 5
        n_mols = 2
        num_types = 10

        aim = torch.randn(n_atoms, dtype=torch.bfloat16)
        # zero last atom (padding) as expected by model
        aim[-1] = 0.0
        numbers = torch.randint(0, num_types, (n_atoms,), dtype=torch.int32)
        # padding atom number is 0
        numbers[-1] = 0
        atomic_shifts = torch.randn(num_types, 1, dtype=torch.bfloat16)
        mol_idx = torch.tensor([0, 0, 1, 1, 1], dtype=torch.int32)

        aim_tt = _to_tt(aim, tt_device)
        numbers_tt = _to_tt(numbers, tt_device)
        atomic_shifts_tt = _to_tt(atomic_shifts, tt_device)
        mol_idx_tt = _to_tt(mol_idx, tt_device)

        energy_tt = ttnn_output_energy(
            aim_tt, numbers_tt, atomic_shifts_tt, None, mol_idx_tt, 1, tt_device
        )

        energy_ref = _ref_output_energy(
            aim, numbers, atomic_shifts, None, mol_idx, 1
        )

        assert torch.allclose(_to_torch(energy_tt), energy_ref, atol=1e-2, rtol=1e-2)
