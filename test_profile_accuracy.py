#!/usr/bin/env python
"""Minimal test to verify profiling timing accuracy."""

import os

os.environ["AIMNET_TTLANG_PROFILE"] = "1"

from time import perf_counter

import ase.build
import torch

from aimnet.calculators import AIMNet2Calculator
from aimnet.ttlang import BackendMode, TTLangAIMNet2
from aimnet.ttlang.ops import get_profile_stats, reset_profile_stats

# Build single molecule
print("Building H2O molecule...")
atoms = ase.build.molecule("H2O")

# Prepare input using calculator (correct way)
calc = AIMNet2Calculator("aimnet2", nb_threshold=0, device="cpu")
data = calc.prepare_input(atoms)

# Add batch dimension for mode 1
data["coord"] = data["coord"].unsqueeze(0)
data["numbers"] = data["numbers"].unsqueeze(0)
data["charge"] = torch.tensor([0.0])
data["mol_idx"] = torch.tensor([0])
data["_nb_mode"] = torch.tensor([1])

print("\nInitializing TT-Lang (hardware)...")
wrapper = TTLangAIMNet2.from_pytorch(calc.model, BackendMode.TT_HW)
print("Done.")

# Warmup
print("\nWarmup...")
reset_profile_stats()
out = wrapper.forward(data)
print(f"Warmup energy: {out['energy'].item():.6f} eV")

# Timed run
print("\nTimed run (with profiling enabled)...")
reset_profile_stats()
t0 = perf_counter()
out = wrapper.forward(data)
t1 = perf_counter()

total_ms = (t1 - t0) * 1000
print(f"\nTotal wall-clock time: {total_ms:.2f} ms")

# Get profile stats
stats = get_profile_stats()
print(f"\nProfile stages:")
stage_sum = 0
for key, vals in stats.items():
    print(f"  {key}: {vals['mean_ms']:.2f} ms")
    stage_sum += vals["total_ms"]

print(f"\nSum of stages: {stage_sum:.2f} ms")
print(f"Difference (total - stages): {abs(total_ms - stage_sum):.2f} ms")

if abs(total_ms - stage_sum) < 50:  # Allow 50ms overhead
    print("\n✓ Profiling timing looks ACCURATE (sync is working)")
else:
    print("\n✗ Profiling timing is INACCURATE (async execution not synchronized)")

print("\nDetailed stage breakdown:")
for key, vals in stats.items():
    print(f"  {key}: mean={vals['mean_ms']:.2f} ms, total={vals['total_ms']:.2f} ms")

wrapper.close()
