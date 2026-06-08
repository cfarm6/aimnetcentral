"""Shared helpers for AIMNet2 TT-Lang benchmarks."""

from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import Callable

import torch

from aimnet.calculators import AIMNet2Calculator

EV_TO_KCAL_MOL = 23.06


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_systems(root) -> list[dict]:
    import ase.build
    import ase.io

    systems = [
        {"name": "H2O", "atoms": ase.build.molecule("H2O")},
        {"name": "CH4", "atoms": ase.build.molecule("CH4")},
        {"name": "NH3", "atoms": ase.build.molecule("NH3")},
    ]
    caffeine = ase.io.read(root / "tests" / "data" / "caffeine.xyz", format="extxyz")
    systems.append({"name": "caffeine", "atoms": caffeine})

    for cif_name in ("2000054.cif", "1100172.cif"):
        cif_path = root / "tests" / "data" / cif_name
        if cif_path.exists():
            atoms = ase.io.read(cif_path)
            systems.append({"name": cif_name.replace(".cif", ""), "atoms": atoms})

    return systems


def atoms_to_data(atoms, charge: float = 0.0) -> dict[str, torch.Tensor]:
    """Convert ASE Atoms to aimnet input data dict."""
    return {
        "coord": torch.tensor(atoms.get_positions(), dtype=torch.float32),
        "numbers": torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long),
        "charge": torch.tensor([charge], dtype=torch.float32),
    }


def prepare_input(calc: AIMNet2Calculator, atoms) -> dict[str, torch.Tensor]:
    """Prepare input data from ASE Atoms (single molecule)."""
    return calc.prepare_input(atoms_to_data(atoms))


def build_mixed_batch(
    calc: AIMNet2Calculator, systems: list[dict], batch_size: int
) -> tuple[dict[str, torch.Tensor], int]:
    """Build flat batched input (mol_idx) cycling through heterogeneous systems."""
    raw_data = []
    mol_idx = []
    n_atoms_total = 0
    for i in range(batch_size):
        sys = systems[i % len(systems)]
        d = atoms_to_data(sys["atoms"])
        n = d["coord"].shape[0]
        raw_data.append(d)
        mol_idx.extend([i] * n)
        n_atoms_total += n

    # Batch raw data (concatenate before prepare_input to avoid padding mismatch)
    batched = {}
    for key in raw_data[0]:
        if isinstance(raw_data[0][key], torch.Tensor):
            batched[key] = torch.cat([d[key] for d in raw_data], dim=0)
    batched["mol_idx"] = torch.tensor(mol_idx, dtype=torch.long)
    batched["_nb_mode"] = torch.tensor([1])

    # Now prepare the whole batch at once (handles padding correctly)
    return calc.prepare_input(batched), n_atoms_total


def time_forward(
    forward_fn,
    data: dict,
    *,
    warmup: int,
    repeats: int,
    after_warmup: Callable[[], None] | None = None,
) -> dict:
    """Time forward passes."""
    for _ in range(warmup):
        forward_fn(data)
    if after_warmup is not None:
        after_warmup()

    samples = []
    for _ in range(repeats):
        t0 = perf_counter()
        forward_fn(data)
        samples.append((perf_counter() - t0) * 1000.0)
    return {
        "mean_ms": sum(samples) / len(samples),
        "std_ms": float(torch.tensor(samples).std(unbiased=False).item()) if len(samples) > 1 else 0.0,
        "median_ms": sorted(samples)[len(samples) // 2],
        "samples_ms": samples,
    }


def is_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "oom" in msg
