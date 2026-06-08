"""Compare PyTorch reference vs TT-Lang energy on caffeine."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aimnet.ttlang import BackendMode

from _common import load_caffeine_atoms, load_default_wrapper, prepare_input


def main() -> None:
    atoms = load_caffeine_atoms()
    data = prepare_input(atoms)

    ref = load_default_wrapper(BackendMode.REFERENCE)
    sim = load_default_wrapper(BackendMode.TT_SIM)

    ref_out = ref.forward(data)
    sim_out = sim.forward(data)
    ref.close()
    sim.close()

    e_ref = ref_out["energy"].item()
    e_sim = sim_out["energy"].item()
    delta_eV = e_sim - e_ref
    delta_kcal_mol = delta_eV * 23.06
    pcc = torch.corrcoef(
        torch.stack([ref_out["charges"].flatten().float(), sim_out["charges"].flatten().float()])
    )[0, 1].item()

    print(f"reference energy: {e_ref:.6f} eV")
    print(f"tt-sim energy:    {e_sim:.6f} eV")
    print(f"energy error:     {delta_kcal_mol:+.6f} kcal/mol ({delta_eV:+.6f} eV)")
    print(f"charge PCC:       {pcc:.6f}")


if __name__ == "__main__":
    main()
