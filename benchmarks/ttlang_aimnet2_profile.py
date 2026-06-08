#!/usr/bin/env python
"""Profile AIMNet2 TT-Lang inference for large singles and batch workloads."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
ROOT = BENCH_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BENCH_DIR))

from aimnet.calculators import AIMNet2Calculator
from aimnet.ttlang import BackendMode, TTLangAIMNet2
from aimnet.ttlang.ops import get_profile_stats, reset_profile_stats, set_profile_enabled
from ttlang_benchmark_common import (
    EV_TO_KCAL_MOL,
    build_mixed_batch,
    load_systems,
    prepare_input,
    time_forward,
    utc_now,
)

DATA_DIR = BENCH_DIR / "data"
LARGE_SYSTEMS = ("1100172", "2000054", "caffeine", "benzene")


def _energy_parity(wrapper: TTLangAIMNet2, ref_wrapper: TTLangAIMNet2, data: dict) -> dict:
    out = wrapper.forward(data)
    ref_out = ref_wrapper.forward(data)
    err_eV = (out["energy"] - ref_out["energy"]).abs().max().item()
    return {
        "energy_error_eV": err_eV,
        "energy_error_kcal_mol": err_eV * EV_TO_KCAL_MOL,
    }


def profile_large_singles(
    mode: str,
    *,
    warmup: int,
    repeats: int,
    profile_mlp: bool,
    profile_stages: bool,
) -> list[dict]:
    backend = BackendMode.REFERENCE if mode == "reference" else BackendMode.TT_HW
    calc = AIMNet2Calculator("aimnet2", nb_threshold=0, device="cpu")
    wrapper = TTLangAIMNet2.from_pytorch(calc.model, backend)
    ref_wrapper = TTLangAIMNet2.from_pytorch(calc.model, BackendMode.REFERENCE)
    systems = [s for s in load_systems(ROOT) if s["name"] in LARGE_SYSTEMS]
    results = []
    for entry in systems:
        data = prepare_input(calc, entry["atoms"])
        if profile_mlp or profile_stages:
            reset_profile_stats()
        timing = time_forward(
            wrapper.forward,
            data,
            warmup=warmup,
            repeats=repeats,
            after_warmup=reset_profile_stats if profile_mlp or profile_stages else None,
        )
        profile_stats = get_profile_stats() if profile_mlp or profile_stages else None
        parity = _energy_parity(wrapper, ref_wrapper, data)
        row = {
            "name": entry["name"],
            "n_atoms": len(entry["atoms"]),
            "timing": timing,
            "throughput_atoms_per_s": len(entry["atoms"]) / (timing["mean_ms"] / 1000.0),
            **parity,
        }
        if profile_stats is not None:
            row["mlp_profile"] = profile_stats
        results.append(row)
        print(
            f"{entry['name']:12s} n={len(entry['atoms']):4d}  "
            f"t={timing['mean_ms']:7.2f}ms  "
            f"err={parity['energy_error_kcal_mol']:+.3e} kcal/mol"
        )
    wrapper.close()
    ref_wrapper.close()
    return results



def profile_batch(
    mode: str,
    *,
    batch_size: int,
    warmup: int,
    repeats: int,
    profile_mlp: bool,
    profile_stages: bool,
) -> dict:
    backend = BackendMode.REFERENCE if mode == "reference" else BackendMode.TT_HW
    calc = AIMNet2Calculator("aimnet2", nb_threshold=0, device="cpu")
    wrapper = TTLangAIMNet2.from_pytorch(calc.model, backend)
    ref_wrapper = TTLangAIMNet2.from_pytorch(calc.model, BackendMode.REFERENCE)
    systems = load_systems(ROOT)
    data, n_atoms_total = build_mixed_batch(calc, systems, batch_size)
    if profile_mlp or profile_stages:
        reset_profile_stats()
    timing = time_forward(
        wrapper.forward,
        data,
        warmup=warmup,
        repeats=repeats,
        after_warmup=reset_profile_stats if profile_mlp or profile_stages else None,
    )
    profile_stats = get_profile_stats() if profile_mlp or profile_stages else None
    parity = _energy_parity(wrapper, ref_wrapper, data)

    row = {
        "batch_size": batch_size,
        "n_atoms_total": n_atoms_total,
        "timing": timing,
        "ms_per_molecule": timing["mean_ms"] / batch_size,
        "molecules_per_s": batch_size / (timing["mean_ms"] / 1000.0),
        "atoms_per_s": n_atoms_total / (timing["mean_ms"] / 1000.0),
        **parity,
    }
    if profile_stats is not None:
        row["mlp_profile"] = profile_stats

    print(
        f"batch={batch_size:4d}  atoms={n_atoms_total:6d}  "
        f"t={timing['mean_ms']:8.2f}ms  "
        f"{row['molecules_per_s']:8.1f} mol/s  "
        f"err_max={parity['energy_error_kcal_mol']:+.3e} kcal/mol"
    )

    wrapper.close()
    ref_wrapper.close()
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["reference", "hw"], default="hw")
    parser.add_argument("--focus", choices=["large", "batch1000", "all"], default="all")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--profile-mlp", action="store_true")
    parser.add_argument("--profile-stages", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.mode == "hw" and not os.path.exists("/dev/tenstorrent/0"):
        print("Tenstorrent device not available.", file=sys.stderr)
        sys.exit(1)

    if args.profile_mlp or args.profile_stages:
        os.environ["AIMNET_TTLANG_PROFILE"] = "1"
        set_profile_enabled(True)

    payload: dict = {
        "schema": "aimnet.ttlang_profile.v1",
        "timestamp": utc_now(),
        "mode": args.mode,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "profile_mlp": args.profile_mlp,
        "profile_stages": args.profile_stages,
    }

    if args.focus in {"large", "all"}:
        payload["large_singles"] = profile_large_singles(
            args.mode,
            warmup=args.warmup,
            repeats=args.repeats,
            profile_mlp=args.profile_mlp,
            profile_stages=args.profile_stages,
        )

    if args.focus in {"batch1000", "all"}:
        payload["batch"] = profile_batch(
            args.mode,
            batch_size=args.batch_size,
            warmup=args.warmup,
            repeats=args.repeats,
            profile_mlp=args.profile_mlp,
            profile_stages=args.profile_stages,
        )

    out = args.output or DATA_DIR / f"ttlang_aimnet2_profile_{args.mode}_latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
