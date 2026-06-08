#!/usr/bin/env python
"""Generate accuracy parity and timing performance plots from benchmark JSON."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

plt.style.use(["bright", "science", "nature"])
plt.rcParams.update({"text.usetex": False})

DATA_DIR = Path(__file__).resolve().parent / "data"
PLOTS_DIR = Path(__file__).resolve().parent / "plots"

# eV to kcal/mol (matches aimnet training metrics)
EV_TO_KCAL_MOL = 23.06


def _load_json(name: str) -> dict | None:
    path = DATA_DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _annotate_energy_error(ax, errors: list[float]) -> None:
    import numpy as np

    err = np.array(errors)
    mae = float(np.mean(np.abs(err)))
    max_err = float(np.max(np.abs(err)))
    ax.text(
        0.05,
        0.95,
        f"MAE={mae:.4e} kcal/mol\nmax |error|={max_err:.4e} kcal/mol",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )


def plot_energy_parity(mode: str, ref_data: dict, tt_data: dict) -> Path:
    ref_by_name = {s["name"]: s for s in ref_data["systems"]}
    entries: list[tuple[str, int, float]] = []
    for s in tt_data["systems"]:
        ref = ref_by_name.get(s["name"])
        if ref is None:
            continue
        e_ref = ref["energy_ref_eV"]
        e_tt = s["energy_tt_eV"]
        entries.append((s["name"], s["n_atoms"], (e_tt - e_ref) * EV_TO_KCAL_MOL))

    entries.sort(key=lambda item: item[1])
    names = [e[0] for e in entries]
    errors = [e[2] for e in entries]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = list(range(len(names)))
    colors = ["C3" if err > 0 else "C0" for err in errors]
    ax.bar(x, errors, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Energy error (kcal/mol)\n(TT − CPU reference)")
    ax.set_xlabel("Chemical system")
    ax.set_title(f"Energy error vs CPU reference: {mode}")
    _annotate_energy_error(ax, errors)
    fig.tight_layout()

    out = PLOTS_DIR / f"energy_parity_{mode}.svg"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    return out


def _rate_stats(count: int, samples_ms: list[float]) -> tuple[float, float]:
    import numpy as np

    rates = np.array([count / (t / 1000.0) for t in samples_ms], dtype=np.float64)
    return float(rates.mean()), float(rates.std(ddof=0))


def plot_timing(data_by_mode: dict[str, dict], metric: str, ylabel: str, filename: str) -> Path:
    modes = list(data_by_mode.keys())
    systems = sorted(
        {s["name"] for d in data_by_mode.values() for s in d["systems"]},
        key=lambda n: next(s["n_atoms"] for d in data_by_mode.values() for s in d["systems"] if s["name"] == n),
    )

    x = list(range(len(systems)))
    width = 0.8 / max(len(modes), 1)
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, mode in enumerate(modes):
        data = data_by_mode[mode]
        by_name = {s["name"]: s for s in data["systems"]}
        vals = [by_name[n][metric] if metric != "timing" else by_name[n]["timing"]["mean_ms"] for n in systems]
        stds = [0.0 if metric != "timing" else by_name[n]["timing"]["std_ms"] for n in systems]
        offset = (i - (len(modes) - 1) / 2) * width
        ax.bar([xi + offset for xi in x], vals, width=width, yerr=stds, capsize=3, label=mode)

    ax.set_xticks(x)
    ax.set_xticklabels(systems, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(filename.replace("_", " ").replace(".svg", ""))
    ax.legend()
    fig.tight_layout()

    out = PLOTS_DIR / filename
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    return out


def plot_batch_performance(data_by_mode: dict[str, dict]) -> Path:
    modes = list(data_by_mode.keys())
    batch_sizes = sorted({
        b["batch_size"] for d in data_by_mode.values() for b in d["batches"] if b.get("status") == "ok"
    })

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for mode in modes:
        ok = {b["batch_size"]: b for b in data_by_mode[mode]["batches"] if b.get("status") == "ok"}
        latencies = [ok[bs]["timing"]["mean_ms"] for bs in batch_sizes if bs in ok]
        lat_stds = [ok[bs]["timing"]["std_ms"] for bs in batch_sizes if bs in ok]
        mol_tp = [ok[bs]["molecules_per_s"] for bs in batch_sizes if bs in ok]
        mol_stds = [
            _rate_stats(ok[bs]["batch_size"], ok[bs]["timing"]["samples_ms"])[1] for bs in batch_sizes if bs in ok
        ]
        x = [bs for bs in batch_sizes if bs in ok]
        axes[0].errorbar(x, latencies, yerr=lat_stds, marker="o", capsize=3, label=mode)
        axes[1].errorbar(x, mol_tp, yerr=mol_stds, marker="o", capsize=3, label=mode)

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Batch size (molecules)")
    axes[0].set_ylabel("Mean latency (ms)")
    axes[0].set_title("Batched inference latency")
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Batch size (molecules)")
    axes[1].set_ylabel("Throughput (molecules/s)")
    axes[1].set_title("Batched inference throughput")
    axes[1].legend()
    axes[1].grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    out = PLOTS_DIR / "batch_performance.svg"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    ref = _load_json("ttlang_aimnet2_performance_reference_latest.json")
    hw = _load_json("ttlang_aimnet2_performance_hw_latest.json")

    if ref is None:
        raise SystemExit("Missing reference benchmark JSON. Run ttlang_aimnet2_performance.py --mode reference first.")

    generated = []
    if hw is not None:
        generated.append(plot_energy_parity("tt-hw", ref, hw))

    timing_data = {"reference": ref}
    if hw is not None:
        timing_data["hw"] = hw

    generated.append(plot_timing(timing_data, "timing", "Mean latency (ms)", "timing_latency.svg"))

    modes = list(timing_data.keys())
    systems = sorted(
        {s["name"] for d in timing_data.values() for s in d["systems"]},
        key=lambda n: next(s["n_atoms"] for d in timing_data.values() for s in d["systems"] if s["name"] == n),
    )
    x = list(range(len(systems)))
    width = 0.8 / max(len(modes), 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, mode in enumerate(modes):
        by_name = {s["name"]: s for s in timing_data[mode]["systems"]}
        vals, stds = [], []
        for n in systems:
            s = by_name[n]
            mean_tp, std_tp = _rate_stats(s["n_atoms"], s["timing"]["samples_ms"])
            vals.append(mean_tp)
            stds.append(std_tp)
        offset = (i - (len(modes) - 1) / 2) * width
        ax.bar([xi + offset for xi in x], vals, width=width, yerr=stds, capsize=3, label=mode)
    ax.set_xticks(x)
    ax.set_xticklabels(systems, rotation=30, ha="right")
    ax.set_ylabel("Throughput (atoms/s)")
    ax.set_title("timing throughput")
    ax.legend()
    fig.tight_layout()
    out = PLOTS_DIR / "timing_throughput.svg"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    generated.append(out)

    batch_data = {}
    for mode, fname in [
        ("reference", "ttlang_aimnet2_batch_performance_reference_latest.json"),
        ("hw", "ttlang_aimnet2_batch_performance_hw_latest.json"),
    ]:
        payload = _load_json(fname)
        if payload is not None:
            batch_data[mode] = payload
    if batch_data:
        generated.append(plot_batch_performance(batch_data))

    for path in generated:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
