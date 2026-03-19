import os

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
ciffile_1 = os.path.join(os.path.dirname(__file__), "2019828.cif")
ciffile_2 = os.path.join(os.path.dirname(__file__), "1119028.cif")

# read the first one
atoms_1 = ase.io.read(ciffile_1)
atoms_2 = ase.io.read(ciffile_2)

# attach the calculator to the atoms object
# create the calculator with default model
base_calc = AIMNet2Calculator("aimnet2")
calc = AIMNet2TorchSim(base_calc, compute_stress=True)

final_state = ts.optimize(
    system=[atoms_1],
    model=calc,
    optimizer=ts.Optimizer.fire,
)

n_steps = 1000
final_state = ts.integrate(
    system=final_state,  # Input atomic system
    model=calc,  # Energy/force model
    integrator=ts.Integrator.nvt_vrescale,  # Integrator to use
    n_steps=n_steps,  # Number of MD steps
    temperature=300,  # Target temperature (K)
    timestep=0.0005,  # Integration timestep (ps)
    # external_pressure=0.0,  # Target external pressure (GPa)
    trajectory_reporter={"filenames": ["2019828.h5"]},
    pbar=True,
)
