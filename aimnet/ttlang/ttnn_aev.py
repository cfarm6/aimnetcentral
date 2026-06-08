from __future__ import annotations

import math
from typing import Any

import torch


def ttnn_calc_distances(
    coord_tt: Any,
    nbmat_tt: Any | None,
    shifts_tt: Any | None,
    nb_mode: int,
    device: Any,
    pad_value: float = 1.0,
    mask_ij_tt: Any | None = None,
) -> tuple[Any, Any]:
    """Compute pairwise distances and displacement vectors on device.

    Parameters
    ----------
    coord_tt
        Atomic coordinates on device.
    nbmat_tt
        Neighbor matrix (required for mode 1/2).
    shifts_tt
        Periodic shift vectors to add to ``coord_j``.
    nb_mode
        Neighbor mode (0, 1, or 2).
    device
        Tenstorrent device handle.
    pad_value
        Value to write into masked (invalid) pair displacements before sqrt.
    mask_ij_tt
        Optional pair mask. If ``None``, a default mask is generated for
        mode 0 (identity) and mode 1 (padding row). Mode 2 callers must
        supply a mask.

    Returns
    -------
    d_ij_tt
        Pairwise distances ``(..., m)``.
    r_ij_tt
        Pairwise displacement vectors ``(..., m, 3)``.
    """
    import ttnn
    from aimnet.ttlang.ttnn_nbops import ttnn_get_ij, ttnn_mask_ij_
    from aimnet.ttlang.ops import stage_profile

    with stage_profile("aev.dist.get_ij"):
        coord_i, coord_j = ttnn_get_ij(coord_tt, nb_mode, nbmat_tt, device)

    if shifts_tt is not None:
        coord_j = ttnn.add(coord_j, shifts_tt)

    r_ij = ttnn.sub(coord_j, coord_i)

    with stage_profile("aev.dist.mask"):
        if mask_ij_tt is None:
            if nb_mode == 0:
                n = coord_tt.shape[-2]
                mask = torch.eye(n, dtype=torch.bfloat16).unsqueeze(0)
                mask_ij_tt = ttnn.from_torch(
                    mask,
                    dtype=ttnn.bfloat16,
                    device=device,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            elif nb_mode == 1:
                if nbmat_tt is None:
                    raise ValueError("nbmat_tt required for mode 1 mask generation")
                n_atoms = int(coord_tt.shape[-2])
                padding_idx = n_atoms - 1
                if nbmat_tt.dtype != ttnn.uint32:
                    nbmat_tt = ttnn.typecast(nbmat_tt, ttnn.uint32)
                padding_tt = ttnn.full_like(nbmat_tt, padding_idx)
                eq_tt = ttnn.eq(nbmat_tt, padding_tt)
                mask_ij_tt = ttnn.typecast(eq_tt, ttnn.bfloat16)
            else:
                raise ValueError("mask_ij_tt is required for mode 2")

        if mask_ij_tt is not None:
            r_ij = ttnn_mask_ij_(r_ij, mask_ij_tt, pad_value)

    r_ij_sq = ttnn.mul(r_ij, r_ij)
    d_ij_sq = ttnn.sum(r_ij_sq, dim=-1)
    d_ij = ttnn.sqrt(d_ij_sq)
    return d_ij, r_ij


def ttnn_cosine_cutoff(d_ij_tt: Any, rc_tt: Any) -> Any:
    """Cosine cutoff envelope on device.

    Parameters
    ----------
    d_ij_tt
        Pairwise distances ``(..., m)``.
    rc_tt
        Cutoff radius (scalar float or device scalar tensor).

    Returns
    -------
    fc_tt
        Cutoff factors ``(..., m)``.
    """
    import ttnn
    if not hasattr(rc_tt, "shape"):
        rc_tt = ttnn.full_like(d_ij_tt, rc_tt)
    min_val = ttnn.full_like(d_ij_tt, 1e-6)
    clamped = ttnn.clamp(d_ij_tt, min_val, rc_tt)
    pi_tensor = ttnn.full_like(rc_tt, math.pi)
    factor = ttnn.div(pi_tensor, rc_tt)
    angle = ttnn.mul(clamped, factor)
    cos_val = ttnn.cos(angle)
    one = ttnn.full_like(cos_val, 1.0)
    added = ttnn.add(cos_val, one)
    half = ttnn.full_like(added, 0.5)
    fc = ttnn.mul(added, half)
    return fc


def ttnn_exp_expand(d_ij_tt: Any, shifts_tt: Any, eta_tt: Any) -> Any:
    """Gaussian radial basis expansion on device.

    Parameters
    ----------
    d_ij_tt
        Pairwise distances ``(..., m)``.
    shifts_tt
        Gaussian centre positions ``(nshifts,)``.
    eta_tt
        Gaussian width (scalar float or device scalar tensor).

    Returns
    -------
    gs_tt
        Expanded features ``(..., m, nshifts)``.
    """
    import ttnn
    d_ij_exp = ttnn.unsqueeze(d_ij_tt, -1)
    diff = ttnn.sub(d_ij_exp, shifts_tt)
    sq = ttnn.mul(diff, diff)
    if not hasattr(eta_tt, "shape"):
        eta_tt = ttnn.full_like(sq, eta_tt)
    neg_one = ttnn.full_like(eta_tt, -1.0)
    neg_eta = ttnn.mul(eta_tt, neg_one)
    scaled = ttnn.mul(sq, neg_eta)
    return ttnn.exp(scaled)


def ttnn_calc_aev(
    r_ij_tt: Any,
    d_ij_tt: Any,
    rc_s_tt: Any,
    shifts_s_tt: Any,
    eta_s_tt: Any,
    mask_ij_tt: Any,
    device: Any,
) -> Any:
    """Compute scalar+vector AEV on device.

    Parameters
    ----------
    r_ij_tt
        Displacement vectors ``(..., m, 3)``.
    d_ij_tt
        Distances ``(..., m)``.
    rc_s_tt
        Scalar cutoff radius.
    shifts_s_tt
        Scalar basis centres ``(nshifts,)``.
    eta_s_tt
        Scalar basis width.
    mask_ij_tt
        Pair mask (True means invalid).
    device
        Tenstorrent device handle.

    Returns
    -------
    g_sv_tt
        AEV features of shape ``(..., m, nshifts, 4)``.
    """
    import ttnn
    from aimnet.ttlang.ttnn_nbops import ttnn_mask_ij_
    from aimnet.ttlang.ops import stage_profile

    with stage_profile("aev.cutoff"):
        fc = ttnn_cosine_cutoff(d_ij_tt, rc_s_tt)
        fc = ttnn_mask_ij_(fc, mask_ij_tt, 0.0)

    with stage_profile("aev.expand"):
        gs = ttnn_exp_expand(d_ij_tt, shifts_s_tt, eta_s_tt)
        gs = ttnn.mul(gs, ttnn.unsqueeze(fc, -1))

    with stage_profile("aev.combine"):
        d_ij_u = ttnn.unsqueeze(d_ij_tt, -1)
        u_ij = ttnn.div(r_ij_tt, d_ij_u)

        gs_u = ttnn.unsqueeze(gs, -1)
        u_ij_u = ttnn.unsqueeze(u_ij, -2)
        gv = ttnn.mul(gs_u, u_ij_u)

        gs_u = ttnn.unsqueeze(gs, -1)
        g_sv = ttnn.concat([gs_u, gv], dim=-1)
    return g_sv
