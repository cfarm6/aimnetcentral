import os
from time import perf_counter

import ase.io
import torch_sim as ts

from aimnet.calculators import AIMNet2Calculator, AIMNet2TorchSim


def torch_show_device_into():
    import torch

    print(f"Torch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA available, version {torch.version.cuda}, device: {torch.cuda.get_device_name()}")  # type: ignore
    else:
        print("CUDA not available")


torch_show_device_into()
# 59 conformations of taxol
xyzfile = os.path.join(os.path.dirname(__file__), "taxol.xyz")

# read the first one
atoms = ase.io.read(xyzfile, index=0)

# create the calculator with default model
base_calc = AIMNet2Calculator("aimnet2")
calc = AIMNet2TorchSim(base_calc)

t0: int | float = perf_counter()
n_systems = 500
systems = [atoms] * n_systems
for _i in range(n_systems):
    systems[_i].info["charge"] = _i / n_systems

print(f"Running optimization for {len(atoms)} atoms molecule with {n_systems} systems.")
final_state = ts.optimize(system=systems, model=calc, optimizer=ts.Optimizer.fire, autobatcher=True, pbar=True)
final_atoms = final_state.to_atoms()
t1 = perf_counter()
print(f"Completed optimization in {t1 - t0:.1f} s")
