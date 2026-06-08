from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TTInputBatch:
    """Device-resident input tensor bundle for AIMNet2 inference.

    Holds the prepared input tensors on Tenstorrent device memory.
    Host metadata fields (nb_mode, input_padded) are kept as Python scalars
    to avoid device-round-trip ``.item()`` calls.
    """
    # Coordinates (..., n_atoms, 3)
    coord: Any
    # Atomic numbers (..., n_atoms)
    numbers: Any
    # Molecular charge (...,)
    charge: Any

    # Neighbor mode: 0 = dense/single, 1 = flat batch, 2 = padded batch
    nb_mode: int

    # Whether the input contains padding atoms
    input_padded: bool

    # Host metadata: output segment size for mode 1 scatter_reduce (n_molecules,)
    # Avoids downloading mol_idx just for shape discovery.
    n_molecules: int = 0

    # Molecule index for mode 1 flat batches (n_atoms,)
    mol_idx: Any | None = None

    # Molecule sizes (n_molecules,)
    mol_sizes: Any | None = None

    # Number of real atoms per system (n_systems,)
    natom: Any | None = None

    # Neighbor matrix (n_atoms, max_neighbors) for mode 1
    nbmat: Any | None = None

    # Atom mask (..., n_atoms) for mode 0/2
    mask_i: Any | None = None

    # Pair mask (..., n_atoms, max_neighbors) for mode 0/1/2
    mask_ij: Any | None = None

    # Periodic shift vectors (..., n_pairs, 3) when present
    shifts: Any | None = None
