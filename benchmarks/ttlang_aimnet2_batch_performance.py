#!/usr/bin/env python
"""AIMNet2 TT-Lang batched inference performance benchmark."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

BENCH_DIR = Path(__file__).resolve().parent
ROOT = BENCH_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BENCH_DIR))

from ttlang_benchmark_common import (
    build_mixed_batch,
    is_oom_error,
    load_systems,
    time_forward,
    utc_now,
)

from aimnet.calculators import AIMNet2Calculator
from aimnet.ttlang import BackendMode, TTLangAIMNet2

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_BATCH_SIZES = [5, 10, 20, 50]


def run_batch_benchmark(
    mode: str,
    *,
    batch_sizes: list[int],
    warmup: int = 3,
    repeats: int = 10,
) -> dict:
    backend = {
        "reference": BackendMode.REFERENCE,
        "hw": BackendMode.TT_HW,
    }[mode]

    calc = AIMNet2Calculator("aimnet2", nb_threshold=0, device="cpu")
    wrapper = TTLangAIMNet2.from_pytorch(calc.model, backend)
    systems = load_systems(ROOT)

    results = []
    for batch_size in batch_sizes:
        entry: dict = {"batch_size": batch_size}
        try:
            data, n_atoms_total = build_mixed_batch(calc, systems, batch_size)
            wrapper.forward(data)
            timing = time_forward(wrapper.forward, data, warmup=warmup, repeats=repeats)
            out = wrapper.forward(data)
            n_mols = int(out["energy"].numel())

            entry.update({
                "status": "ok",
                "n_molecules": n_mols,
                "n_atoms_total": n_atoms_total,
                "timing": timing,
                "ms_per_molecule": timing["mean_ms"] / batch_size,
                "molecules_per_s": batch_size / (timing["mean_ms"] / 1000.0),
                "atoms_per_s": n_atoms_total / (timing["mean_ms"] / 1000.0),
            })
            print(
                f"batch={batch_size:4d}  mols={n_mols:4d}  atoms={n_atoms_total:6d}  "
                f"t={timing['mean_ms']:8.2f}ms  "
                f"{entry['molecules_per_s']:8.1f} mol/s  {entry['atoms_per_s']:10.1f} atom/s"
            )
        except Exception as exc:
            entry.update({"status": "skipped", "reason": str(exc)})
            if is_oom_error(exc):
                entry["reason"] = "out of memory"
            print(f"batch={batch_size:4d}  SKIPPED ({entry['reason']})")

        results.append(entry)

    wrapper.close()

    return {
        "schema": "aimnet.ttlang_batch_performance.v1",
        "timestamp": utc_now(),
        "mode": mode,
        "warmup": warmup,
        "repeats": repeats,
        "batch_sizes_requested": batch_sizes,
        "system_pool": [s["name"] for s in systems],
        "batches": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["reference", "hw"], default="reference")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.mode == "hw" and not os.path.exists("/dev/tenstorrent/0"):
        print("Tenstorrent device not available; skipping hw benchmark.", file=sys.stderr)
        sys.exit(1)

    payload = run_batch_benchmark(
        args.mode,
        batch_sizes=args.batch_sizes,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    out = args.output or DATA_DIR / f"ttlang_aimnet2_batch_performance_{args.mode}_latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
