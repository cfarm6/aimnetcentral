import os

import ase.io
import torch

from aimnet.calculators import AIMNet2Calculator


def torch_show_device_into():
    import torch

    print(f"Torch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA available, version {torch.version.cuda}, device: {torch.cuda.get_device_name()}")  # type: ignore
    else:
        print("CUDA not available")


torch_show_device_into()
# 59 conformations of taxol
xyzfile = os.path.join(os.path.dirname(__file__), "taxol_0.xyz")

# Read a single conformation and duplicate it to form a small batch
atoms = ase.io.read(xyzfile, index=0)
n_atoms = len(atoms)
# Define two charge-constrained regions: selected indices and the complement
region_1_indices = [
    2,
    6,
    7,
    12,
    13,
    17,
    64,
    68,
    69,
    74,
    75,
]
region_2_indices = [1, 3, 4, 5]
region_3_indices = [i for i in range(n_atoms) if i not in region_1_indices and i not in region_2_indices]


# Build a per-atom region ID mask compatible with AIMNet2Calculator /
# AIMNet2Base and constrained_nse:
#   region_mask[i] == 0  -> atom i in the first constrained region
#   region_mask[i] == 1  -> atom i in the second constrained region
region_mask = torch.full((n_atoms + 1,), 0, dtype=torch.long)
region_mask[region_1_indices] = 0
region_mask[region_2_indices] = 1
region_mask[region_3_indices] = 2

region_charges_1 = -1.0
region_charges_2 = 1.0
region_charges_3 = 0.0
# Prepare a simple batched input (two identical molecules)
coords = torch.tensor(atoms.get_positions(), dtype=torch.float32)
numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
charges = 0.0
region_charges = torch.tensor([region_charges_1, region_charges_2, region_charges_3], dtype=torch.float32)
region_mask = torch.tensor(region_mask, dtype=torch.long)

calc = AIMNet2Calculator("aimnet2_2025")
result = calc(
    {
        "coord": coords,
        "numbers": numbers,
        "charge": charges,
        "region_charges": region_charges,
        "region_mask": region_mask,
    },
    forces=True,
)

print(result["energy"])
print("Region 1: ", result["charges"][region_1_indices].sum())
print("Region 2: ", result["charges"][region_2_indices].sum())
print("Region 3: ", result["charges"][region_3_indices].sum())
print("Total: ", result["charges"].sum())
