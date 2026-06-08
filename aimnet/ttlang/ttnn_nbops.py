from __future__ import annotations

from typing import Any


def ttnn_mask_i_(x_tt: Any, mask_i_tt: Any | None, mask_value: float = 0.0) -> Any:
    """Apply atom mask on device using ttnn.where.

    For mode 1, ``mask_i_tt`` is None and the caller should zero the last row
    directly; this function is a no-op in that case.
    """
    import ttnn
    if mask_i_tt is None:
        return x_tt
    # Expand mask to match x_tt dimensions
    ndim_diff = len(x_tt.shape) - len(mask_i_tt.shape)
    for _ in range(ndim_diff):
        mask_i_tt = ttnn.unsqueeze(mask_i_tt, -1)
    zeros = ttnn.full_like(x_tt, mask_value)
    return ttnn.where(mask_i_tt, zeros, x_tt)

def ttnn_mask_ij_(x_tt: Any, mask_ij_tt: Any | None, mask_value: float = 0.0) -> Any:
    """Apply pair mask on device using ttnn.where."""
    import ttnn

    if mask_ij_tt is None:
        return x_tt
    # Expand mask to match x_tt dimensions
    ndim_diff = len(x_tt.shape) - len(mask_ij_tt.shape)
    for _ in range(ndim_diff):
        mask_ij_tt = ttnn.unsqueeze(mask_ij_tt, -1)
    zeros = ttnn.full_like(x_tt, mask_value)
    return ttnn.where(mask_ij_tt, zeros, x_tt)


def ttnn_get_i(x_tt: Any, nb_mode: int) -> Any:
    """Get the i-component for pairwise expansion.

    mode 0/1/2: ``x.unsqueeze(-2)`` (adds neighbor dim).
    """
    import ttnn

    if nb_mode in (0, 1, 2):
        return ttnn.unsqueeze(x_tt, -2)
    raise ValueError(f"Invalid neighbor mode: {nb_mode}")


def ttnn_get_ij(
    x_tt: Any,
    nb_mode: int,
    nbmat_tt: Any | None,
    device: Any,
) -> tuple[Any, Any]:
    """Get i/j components for pairwise expansion.

    mode 0: full Cartesian product via unsqueeze.
    mode 1: ``x_i = x.unsqueeze(-2)``, ``x_j = embedding(nbmat, x)``.
    mode 2: flatten batch then embedding.
    """
    import ttnn

    if nb_mode == 0:
        x_i = ttnn.unsqueeze(x_tt, -2)
        x_j = ttnn.unsqueeze(x_tt, -3)
        return x_i, x_j
    elif nb_mode == 1:
        x_i = ttnn.unsqueeze(x_tt, -2)
        if nbmat_tt is None:
            raise ValueError("nbmat required for mode 1")
        # Ensure nbmat is uint32 for embedding
        if nbmat_tt.dtype != ttnn.uint32:
            nbmat_tt = ttnn.typecast(nbmat_tt, ttnn.uint32)
        x_j = ttnn.embedding(nbmat_tt, x_tt)
        return x_i, x_j
    elif nb_mode == 2:
        x_i = ttnn.unsqueeze(x_tt, -2)
        if nbmat_tt is None:
            raise ValueError("nbmat required for mode 2")
        if nbmat_tt.dtype != ttnn.uint32:
            nbmat_tt = ttnn.typecast(nbmat_tt, ttnn.uint32)
        # Flatten batch dimension: (B, N, M) -> (B*N, M)
        flat_x = ttnn.reshape(x_tt, (-1, x_tt.shape[-1]))
        x_j = ttnn.embedding(nbmat_tt, flat_x)
        return x_i, x_j
    raise ValueError(f"Invalid neighbor mode: {nb_mode}")


def ttnn_mol_sum(x_tt: Any, nb_mode: int, mol_idx_tt: Any | None, device: Any, n_molecules: int = 0) -> Any:
    """Segmented molecule sum.

    mode 0/2: sum over dim 1.
    mode 1: scatter_add using mol_idx.  ``n_molecules`` is host metadata
        (output segment count) and MUST be provided for mode 1 to avoid
        a device-to-host download.
    """
    import ttnn

    if nb_mode in (0, 2):
        return ttnn.sum(x_tt, dim=1)
    elif nb_mode == 1:
        if mol_idx_tt is None:
            raise ValueError("mol_idx required for mode 1 mol_sum")
        if mol_idx_tt.dtype != ttnn.uint32:
            mol_idx_tt = ttnn.typecast(mol_idx_tt, ttnn.uint32)

        if n_molecules <= 0:
            # Fallback: discover from device (MUST NOT happen in full-device path)
            out_size = int(ttnn.to_torch(mol_idx_tt)[-1].item()) + 1
        else:
            out_size = n_molecules

        # Squeeze unit trailing dim to avoid repeat in 1D scatter_add path.
        was_squeezed = False
        if len(x_tt.shape) == 2 and x_tt.shape[-1] == 1:
            x_tt = ttnn.squeeze(x_tt, -1)
            was_squeezed = True

        if len(x_tt.shape) == 1:
            out_tt = ttnn.zeros((out_size,), dtype=x_tt.dtype, device=device, layout=ttnn.TILE_LAYOUT)
            result = ttnn.scatter_add(out_tt, 0, mol_idx_tt, x_tt)
            if was_squeezed:
                result = ttnn.unsqueeze(result, -1)
            return result
        else:
            out_shape = (out_size, x_tt.shape[-1])
            out_tt = ttnn.zeros(out_shape, dtype=x_tt.dtype, device=device, layout=ttnn.TILE_LAYOUT)
            idx_expanded = ttnn.unsqueeze(mol_idx_tt, -1)
            repeat_count = x_tt.shape[-1]
            idx_expanded = ttnn.repeat(idx_expanded, (1, repeat_count))
            return ttnn.scatter_add(out_tt, 0, idx_expanded, x_tt)
    raise ValueError(f"Invalid neighbor mode: {nb_mode}")
