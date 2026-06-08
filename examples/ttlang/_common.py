from __future__ import annotations

import os
from time import perf_counter

import torch

from aimnet.calculators import AIMNet2Calculator
from aimnet.ttlang import BackendMode, TTLangAIMNet2


def load_caffeine_atoms():
    import ase.io

    path = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "data", "caffeine.xyz")
    return ase.io.read(path, format="extxyz")


def load_default_wrapper(backend: BackendMode | str = BackendMode.REFERENCE) -> TTLangAIMNet2:
    calc = AIMNet2Calculator("aimnet2", nb_threshold=0, device="cpu")
    return TTLangAIMNet2.from_pytorch(calc.model, backend)


def prepare_input(atoms) -> dict[str, torch.Tensor]:
    calc = AIMNet2Calculator("aimnet2", nb_threshold=0, device="cpu")
    return calc.prepare_input(
        {
            "coord": atoms.positions,
            "numbers": atoms.numbers,
            "charge": 0.0,
        }
    )


def run_energy(backend: BackendMode | str, *, label: str | None = None) -> dict:
    atoms = load_caffeine_atoms()
    data = prepare_input(atoms)
    wrapper = load_default_wrapper(backend)
    t0 = perf_counter()
    out = wrapper.forward(data)
    elapsed_ms = (perf_counter() - t0) * 1000.0
    wrapper.close()

    energy = out["energy"].item()
    charges = out["charges"].detach().cpu().numpy()
    tag = label or str(backend)
    print(f"[{tag}] energy={energy:.6f} eV  atoms={len(atoms)}  time={elapsed_ms:.2f} ms")
    print(f"[{tag}] charges: min={charges.min():.4f} max={charges.max():.4f} sum={charges.sum():.4f}")
    return {"energy": energy, "elapsed_ms": elapsed_ms, "charges": charges}
