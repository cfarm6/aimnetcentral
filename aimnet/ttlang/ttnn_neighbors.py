from __future__ import annotations

from typing import Any

import torch


def ttnn_build_neighbors_brute_force(
    coord_tt: Any,
    cutoff: float,
    max_neighbors: int,
    device: Any,
) -> tuple[Any, Any]:
    """Prototype device-side neighbor list via brute-force O(N^2) scan.

    Parameters
    ----------
    coord_tt
        Atomic coordinates on device.
    cutoff
        Distance cutoff in Angstroms.
    max_neighbors
        Target number of neighbors per atom (unused because TTNN lacks top-k).
    device
        Tenstorrent device handle.

    Returns
    -------
    d_ij_tt
        Full pairwise distance matrix ``(N, N)``.
    r_ij_tt
        Full pairwise displacement matrix ``(N, N, 3)``.

    Notes
    -----
    This prototype demonstrates that a brute-force approach reproduces the
    same distances as the CPU neighbor list for tiny systems, but it cannot
    prune to ``max_neighbors`` because TTNN does not provide ``topk`` or
    ``argsort``.  Consequently it is O(N^2) memory and compute, which is not
    viable for production molecules.
    """
    import ttnn
    from aimnet.ttlang.ttnn_nbops import ttnn_get_ij

    # Full Cartesian product (mode 0)
    coord_i, coord_j = ttnn_get_ij(coord_tt, nb_mode=0, nbmat_tt=None, device=device)
    r_ij = ttnn.sub(coord_j, coord_i)
    r_ij_sq = ttnn.mul(r_ij, r_ij)
    d_ij_sq = ttnn.sum(r_ij_sq, dim=-1)
    d_ij = ttnn.sqrt(d_ij_sq)
    return d_ij, r_ij


# ----------------------------------------------------------------------------
# Device-side neighbor-list construction: go / no-go decision
# ----------------------------------------------------------------------------
# We compared the brute-force prototype above against CPU
# ``AIMNet2Calculator.prepare_input`` for a water molecule (3 real atoms).
#
# CPU neighbor list (mode 1, after padding):
#   nbmat = [[1, 2],
#            [0, 2],
#            [0, 1],
#            [3, 3]]
#
# The brute-force O(N^2) distances are numerically correct for this tiny
# system, but TTNN lacks the following primitives required for a production
# neighbor list:
#   - Spatial hashing / cell list (avoids O(N^2) scaling)
#   - top-k / argsort (needed to select nearest max_neighbors)
#   - Dynamic buffer allocation (needed for adaptive neighbor list sizing)
#
# Without these, a device-side neighbor list would be O(N^2) memory and
# compute, which is prohibitive for production molecules.
#
# VERDICT: DEFERRED.
# Neighbor lists should continue to be built on the CPU and uploaded to the
# device as part of the input batch.
