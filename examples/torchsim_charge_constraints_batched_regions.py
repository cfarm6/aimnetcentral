"""Batched charge-region constraints (variable regions per system) with AIMNet2TorchSim.

TorchSim :class:`torch_sim.state.SimState` stores **flat** positions ``(n_atoms_total, 3)``
and ``system_idx`` (not a dense ``(B, N, 3)`` tensor). That geometry maps to AIMNet2
``nb_mode=1`` (neighbor list). For **multiple systems**, per-atom ``region_mask`` ids must be
**disjoint** across systems (here: system 1 uses ids ``+ R_max``) so region-wise sums do not
merge atoms from different systems.

For *dense* ``(B, N, 3)`` input and ``nb_mode=0`` region scatter, pass a state dict to
:class:`aimnet.calculators.aimnet2torchsim.AIMNet2TorchSim` instead of ``SimState``.

With region constraints, :class:`~aimnet.calculators.aimnet2torchsim.AIMNet2TorchSim` prints
per-region charge sums to stdout and omits ``charges`` from the dict returned through
``torch_sim.static`` (pass ``return_charges=True`` if you need per-atom charges in Python).

``AIMNet2ASE`` can instead take per-atom ``atoms.info['region_mask']`` and
``atoms.info['region_charge']`` / ``atoms.info['region_charges']`` (see
:func:`aimnet.calculators.aimnet2ase.AIMNet2ASE._region_constraint_tensors_from_info`).

Run (requires ``torch-sim-atomistic``):

  python examples/torchsim_charge_constraints_batched_regions.py
"""

import os
from typing import TYPE_CHECKING

import ase.io
import torch
import torch_sim as ts

from aimnet.calculators import AIMNet2Calculator
from aimnet.calculators.aimnet2torchsim import AIMNet2TorchSim

if TYPE_CHECKING:
    from ase import Atoms


def main() -> None:
    xyzfile = os.path.join(os.path.dirname(__file__), "taxol_0.xyz")
    atoms = ase.io.read(xyzfile, index=0)
    n_atoms = len(atoms)

    # Same three-region layout as ase_charge_constraints_nb_mode_1.py (single system)
    region_1_indices = [2, 6, 7, 12, 13, 17, 64, 68, 69, 74, 75]
    region_2_indices = [1, 3, 4, 5]
    region_3_indices = [i for i in range(n_atoms) if i not in region_1_indices and i not in region_2_indices]

    rm_a = torch.zeros(n_atoms, dtype=torch.long)
    rm_a[region_1_indices] = 0
    rm_a[region_2_indices] = 1
    rm_a[region_3_indices] = 2

    # Second system: swap roles of region 1 and 2 (same atom count, different targets)
    rm_b = rm_a.clone()
    rm_b[region_1_indices] = 1
    rm_b[region_2_indices] = 0

    region_mask = torch.stack([rm_a, rm_b], dim=0)  # (2, N)

    rc_a = torch.tensor([-1.0, 1.0, 0.0], dtype=torch.float32)
    rc_b = torch.tensor([0.25, -0.25, 0.0], dtype=torch.float32)
    region_charges = torch.stack([rc_a, rc_b], dim=0)  # (2, R_max)

    base_calc = AIMNet2Calculator("aimnet2_2025", nb_threshold=max(n_atoms + 1, 256))
    device = torch.device(base_calc.device)
    dtype = torch.float32
    model = AIMNet2TorchSim(base_calc)

    # Two ASE systems (same geometry, different total charge in info); batched via torch-sim IO.
    atoms_0: Atoms = atoms.copy()
    atoms_1: Atoms = atoms.copy()
    atoms_0.info["charge"] = float(region_charges[0].sum().item())
    atoms_1.info["charge"] = float(region_charges[1].sum().item())
    atoms_0.info["region_mask"] = region_mask[0]
    atoms_1.info["region_mask"] = region_mask[1]
    atoms_0.info["region_charges"] = region_charges[0]
    atoms_1.info["region_charges"] = region_charges[1]

    state = ts.io.atoms_to_state(
        [atoms_0, atoms_1],
        device=device,
        dtype=dtype,
        system_extras_map={
            "charge": "charge",
            "region_mask": "region_mask",
            "region_charges": "region_charges",
        },
    )
    static_model = ts.optimize(
        state,
        model,
        optimizer=ts.Optimizer.fire,
        # autobatcher=False
        # autobatcher=True
    )
    print([out.energy for out in static_model])


if __name__ == "__main__":
    main()
