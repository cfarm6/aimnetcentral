from __future__ import annotations

from typing import Any

from aimnet.ttlang.ttnn_nbops import ttnn_mol_sum


def ttnn_update_q(
    charge_tt: Any,
    charges_tt: Any,
    a_tt: Any,
    _out_tt: Any,
    num_charge_channels: int,
    mol_idx_tt: Any | None,
    nb_mode: int,
    device: Any,
    delta_q: bool = True,
    epsilon: float = 1.0e-6,
    n_molecules: int = 0,
) -> tuple[Any, Any, Any]:
    """Update charges and atomic features using NSE charge correction on device.

    Mirrors the PyTorch ``AIMNet2._update_q`` + ``ops.nse`` logic.

    Parameters
    ----------
    n_molecules
        Number of molecules (segment count) for mode 1.  Must be > 0 for
        the full-device path; avoids a device-to-host download of mol_idx.
    """
    import ttnn
    from aimnet.ttlang.ops import stage_profile

    # Split in TILE_LAYOUT — all MLP output tensors are already TILE.
    with stage_profile("update_q.split"):
        out_dim = _out_tt.shape[-1]
        _q, _f, delta_a = ttnn.split(
            _out_tt,
            [num_charge_channels, num_charge_channels, out_dim - 2 * num_charge_channels],
            dim=-1,
        )

    # Charge conservation violation of raw _q (for loss)
    delta_Q_tt = ttnn.subtract(charge_tt, ttnn_mol_sum(_q, nb_mode, mol_idx_tt, device, n_molecules))

    # Form uncorrected charge
    if delta_q:
        q_u = ttnn.add(charges_tt, _q)
    else:
        q_u = _q

    # Squared charge weights
    f_u = ttnn.pow(_f, 2)

    # Molecule sums + NSE
    with stage_profile("update_q.mol_sums"):
        F_u = ttnn_mol_sum(f_u, nb_mode, mol_idx_tt, device, n_molecules)
        if epsilon > 0:
            F_u = ttnn.add(F_u, ttnn.full_like(F_u, epsilon))
        Q_u = ttnn_mol_sum(q_u, nb_mode, mol_idx_tt, device, n_molecules)
        dQ = ttnn.subtract(charge_tt, Q_u)

    # Expand per-molecule values for per-atom broadcasting
    if nb_mode in (0, 2):
        F_u = ttnn.unsqueeze(F_u, -2)
        dQ = ttnn.unsqueeze(dQ, -2)
    elif nb_mode == 1:
        if mol_idx_tt is None:
            raise ValueError("mol_idx required for mode 1")
        if mol_idx_tt.dtype != ttnn.uint32:
            mol_idx_tt = ttnn.typecast(mol_idx_tt, ttnn.uint32)
        F_u = ttnn.embedding(mol_idx_tt, F_u)
        dQ = ttnn.embedding(mol_idx_tt, dQ)
    else:
        raise ValueError(f"Invalid neighbor mode: {nb_mode}")

    # NSE correction
    with stage_profile("update_q.nse"):
        f = ttnn.divide(f_u, F_u)
        q = ttnn.add(q_u, ttnn.multiply(f, dQ))

    # Update atomic features — reshape delta_a to match a_tt shape.
    with stage_profile("update_q.delta_a"):
        N = a_tt.shape[0]
        C = a_tt.shape[1]
        S = a_tt.shape[2]
        a_2d = ttnn.reshape(a_tt, (N, C * S))
        a_updated_2d = ttnn.add(a_2d, delta_a)
        a_tt = ttnn.reshape(a_updated_2d, (N, C, S))
    return q, a_tt, delta_Q_tt
