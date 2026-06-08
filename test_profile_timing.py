#!/usr/bin/env python
"""Quick test to verify profiling timing fix."""

import os
import sys
from time import perf_counter

import ase.build
import torch

# Enable profiling
os.environ["AIMNET_TTLANG_PROFILE"] = "1"

from aimnet.calculators import AIMNet2Calculator
from aimnet.ttlang import BackendMode, TTLangAIMNet2
from aimnet.ttlang.ops import get_profile_stats, reset_profile_stats

# Build test input
print("Building test molecule (H2O)...")
atoms = ase.build.molecule("H2O")
calc = AIMNet2Calculator("aimnet2", nb_threshold=0, device="cpu")
data = calc.prepare_input(atoms, charge=0.0)

# Add batch dimension (mode 1 - single molecule)
data["coord"] = data["coord"].unsqueeze(0)  # [1, 3]
data["numbers"] = data["numbers"].unsqueeze(0)  # [1]
data["charge"] = torch.tensor([0.0])
data["mol_idx"] = torch.tensor([0])
data["_nb_mode"] = torch.tensor([1])

# Create TT-Lang wrapper
print("\nInitializing TT-Lang (hardware)...")
wrapper = TTLangAIMNet2.from_pytorch(calc.model, "tt-hw")
print("Done.")

# Warmup
print("\nWarmup...")
reset_profile_stats()
out = wrapper.forward(data)
print(f"Warmup energy: {out['energy'].item():.6f} eV")

# Timed run with profiling
print("\nTimed run (with profiling)...")
reset_profile_stats()
t0 = perf_counter()
out = wrapper.forward(data)
t1 = perf_counter()
print(f"Total time: {(t1 - t0) * 1000:.2f} ms")

# Get profile stats
stats = get_profile_stats()
print(f"\nProfile stats:")
for key, vals in stats.items():
    print(f"  {key}: mean={vals['mean_ms']:.2f} ms, total={vals['total_ms']:.2f} ms")

# Compare: total time vs sum of stages
total_ms = (t1 - t0) * 1000
stage_sum = sum(v["total_ms"] for v in stats.values())
print(f"\nTotal wall-clock: {total_ms:.2f} ms")
print(f"Sum of stages: {stage_sum:.2f} ms")
print(f"Difference: {abs(total_ms - stage_sum):.2f} ms")

if abs(total_ms - stage_sum) < 50:  # Allow some overhead
    print("\n✓ Profiling timing looks accurate (sync is working)")
else:
    print("\n✗ Profiling timing is inaccurate (async execution not synchronized)")

wrapper.close()
