"""Batch energy inference over taxol conformations."""

import os
import sys
from pathlib import Path
from time import perf_counter

import ase.io

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aimnet.ttlang import BackendMode

from _common import load_default_wrapper, prepare_input


def main() -> None:
    xyzfile = os.path.join(os.path.dirname(__file__), "..", "taxol.xyz")
    wrapper = load_default_wrapper(BackendMode.TT_SIM)
    t0 = perf_counter()
    energies = []
    for i, atoms in enumerate(ase.io.iread(xyzfile, index=":")):
        out = wrapper.forward(prepare_input(atoms))
        energies.append(out["energy"].item())
        if i >= 4:
            break
    wrapper.close()
    elapsed = perf_counter() - t0
    print(f"Computed {len(energies)} energies in {elapsed:.2f} s")
    for i, e in enumerate(energies):
        print(f"  conf {i}: {e:.6f} eV")


if __name__ == "__main__":
    main()
