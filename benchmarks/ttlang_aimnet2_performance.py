#!/usr/bin/env python
"""AIMNet2 TT-Lang performance and accuracy benchmark."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aimnet.calculators import AIMNet2Calculator
from aimnet.ttlang import BackendMode, TTLangAIMNet2

DATA_DIR = Path(__file__).resolve().parent / "data"

# eV to kcal/mol (matches aimnet training metrics)
EV_TO_KCAL_MOL = 23.06


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_systems() -> list[dict]:
    import ase.build
    import ase.io

    systems = [
        {"name": "water", "atoms": ase.build.molecule("H2O")},
        {"name": "methane", "atoms": ase.build.molecule("CH4")},
        {"name": "ethanol", "atoms": ase.build.molecule("CH3CH2OH")},
        {"name": "benzene", "atoms": ase.build.molecule("C6H6")},
    ]
    caffeine = ase.io.read(ROOT / "tests" / "data" / "caffeine.xyz", format="extxyz")
    systems.append({"name": "caffeine", "atoms": caffeine})

    for cif_name in ("2000054.cif", "1100172.cif"):
        cif_path = ROOT / "tests" / "data" / cif_name
        if cif_path.exists():
            atoms = ase.io.read(cif_path)
            if len(atoms) > 200:
                atoms = atoms[:200]
            systems.append({"name": cif_name.replace(".cif", ""), "atoms": atoms})

    return systems


def prepare_input(atoms) -> dict[str, torch.Tensor]:
    calc = AIMNet2Calculator("aimnet2", nb_threshold=0, device="cpu")
    charge = 0.0
    if hasattr(atoms, "info") and "charge" in atoms.info:
        charge = atoms.info["charge"]
    return calc.prepare_input(
        {
            "coord": atoms.positions,
            "numbers": atoms.numbers,
            "charge": charge,
        }
    )


def time_forward(wrapper: TTLangAIMNet2, data: dict, *, warmup: int, repeats: int) -> dict:
    for _ in range(warmup):
        wrapper.forward(data)
    samples = []
    for _ in range(repeats):
        t0 = perf_counter()
        wrapper.forward(data)
        samples.append((perf_counter() - t0) * 1000.0)
    return {
        "mean_ms": sum(samples) / len(samples),
        "std_ms": float(torch.tensor(samples).std(unbiased=False).item()) if len(samples) > 1 else 0.0,
        "median_ms": sorted(samples)[len(samples) // 2],
        "samples_ms": samples,
    }


def run_benchmark(mode: str, *, warmup: int = 3, repeats: int = 10) -> dict:
    backend = {
        "reference": BackendMode.REFERENCE,
        "hw": BackendMode.TT_HW,
    }[mode]

    calc = AIMNet2Calculator("aimnet2", nb_threshold=0, device="cpu")
    wrapper = TTLangAIMNet2.from_pytorch(calc.model, backend)

    ref_wrapper = TTLangAIMNet2.from_pytorch(calc.model, BackendMode.REFERENCE)

    results = []
    for entry in load_systems():
        atoms = entry["atoms"]
        data = prepare_input(atoms)
        timing = time_forward(wrapper, data, warmup=warmup, repeats=repeats)
        out = wrapper.forward(data)
        ref_out = ref_wrapper.forward(data)

        e_tt = out["energy"].item()
        e_ref = ref_out["energy"].item()
        energy_error_eV = e_tt - e_ref
        energy_error_kcal_mol = energy_error_eV * EV_TO_KCAL_MOL
        results.append(
            {
                "name": entry["name"],
                "n_atoms": len(atoms),
                "energy_ref_eV": e_ref,
                "energy_tt_eV": e_tt,
                "energy_error_eV": energy_error_eV,
                "energy_abs_error_eV": abs(energy_error_eV),
                "energy_error_kcal_mol": energy_error_kcal_mol,
                "energy_abs_error_kcal_mol": abs(energy_error_kcal_mol),
                "timing": timing,
                "throughput_atoms_per_s": len(atoms) / (timing["mean_ms"] / 1000.0),
            }
        )
        print(
            f"{entry['name']:12s} n={len(atoms):4d}  "
            f"ref={e_ref:12.6f} eV  tt={e_tt:12.6f} eV  "
            f"error={energy_error_kcal_mol:+.4e} kcal/mol  t={timing['mean_ms']:.2f}ms"
        )

    wrapper.close()
    ref_wrapper.close()

    return {
        "schema": "aimnet.ttlang_performance.v1",
        "timestamp": utc_now(),
        "mode": mode,
        "warmup": warmup,
        "repeats": repeats,
        "systems": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["reference", "hw"], default="reference")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.mode == "hw" and not os.path.exists("/dev/tenstorrent/0"):
        print("Tenstorrent device not available; skipping hw benchmark.", file=sys.stderr)
        sys.exit(1)

    payload = run_benchmark(args.mode, warmup=args.warmup, repeats=args.repeats)
    out = args.output or DATA_DIR / f"ttlang_aimnet2_performance_{args.mode}_latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
