from __future__ import annotations

from typing import Any

from aimnet.ttlang.ttnn_nbops import ttnn_mask_i_, ttnn_mol_sum


def ttnn_output_energy(
    aim_tt: Any,
    numbers_tt: Any,
    atomic_shifts_tt: Any,
    mask_i_tt: Any | None,
    mol_idx_tt: Any | None,
    nb_mode: int,
    device: Any,
    n_molecules: int = 0,
) -> Any:
    """Compute molecular energy from per-atom energies and atomic shifts on device.

    Mirrors the PyTorch ``AtomicShift`` + ``AtomicSum`` logic.

    Parameters
    ----------
    n_molecules
        Number of molecules (segment count) for mode 1.  Must be > 0 for
        the full-device path.
    """
    import ttnn

    if numbers_tt.dtype != ttnn.uint32:
        numbers_tt = ttnn.typecast(numbers_tt, ttnn.uint32)

    # Gather atomic shifts and squeeze last dimension
    shifts_tt = ttnn.embedding(numbers_tt, atomic_shifts_tt)
    shifts_tt = ttnn.squeeze(shifts_tt, -1)

    # Per-atom energy = aim + shifts
    e_atom_tt = ttnn.add(aim_tt, shifts_tt)

    # Apply atom mask (for mode 1, zero the last padding row directly)
    if nb_mode == 1 and mask_i_tt is None:
        n = e_atom_tt.shape[0]
        if len(e_atom_tt.shape) == 1:
            first = ttnn.slice(e_atom_tt, [0], [n - 1])
            last = ttnn.full_like(ttnn.slice(e_atom_tt, [n - 1], [n]), 0.0)
            e_atom_tt = ttnn.concat([first, last], dim=0)
        else:
            last_dim = e_atom_tt.shape[-1]
            first = ttnn.slice(e_atom_tt, [0, 0], [n - 1, last_dim])
            last = ttnn.full_like(ttnn.slice(e_atom_tt, [n - 1, 0], [n, last_dim]), 0.0)
            e_atom_tt = ttnn.concat([first, last], dim=0)
    else:
        e_atom_tt = ttnn_mask_i_(e_atom_tt, mask_i_tt, 0.0)

    # Sum per molecule
    energy_tt = ttnn_mol_sum(e_atom_tt, nb_mode, mol_idx_tt, device, n_molecules)

    return energy_tt
