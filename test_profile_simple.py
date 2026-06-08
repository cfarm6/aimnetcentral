#!/usr/bin/env python
"""Minimal test to verify profiling timing fix works."""

import os

# Enable profiling BEFORE any ttnn import
os.environ["AIMNET_TTLANG_PROFILE"] = "1"

from time import perf_counter

import ase.build
import torch

from aimnet.calculators import AIMNet2Calculator
from aimnet.ttlang import BackendMode, TTLangAIMNet2
from aimnet.ttlang.ops import _PROFILE, get_profile_stats, reset_profile_stats

print(f"_PROFILE = {_PROFILE}")

# Build test input - use a simple molecule
print("\nBuilding test molecule (H2O)...")
atoms = ase.build.molecule("H2O")
calc = AIMNet2Calculator("aimnet2", nb_threshold=0, device="cpu")

# Prepare input the RIGHT way - let calculator handle ASE Atoms
# Prepare input the RIGHT way - pass data dict, not Atoms
print("Preparing input...")
data = calc.prepare_input({"coord": torch.tensor(atoms.get_positions(), dtype=torch.float32),
                              "numbers": torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long),
                              "charge": torch.tensor(0.0)})
data["coord"] = data["coord"].unsqueeze(0)  # [1, N, 3]
data["numbers"] = data["numbers"].unsqueeze(0)  # [1, N]
data["charge"] = torch.tensor([0.0])
data["mol_idx"] = torch.tensor([0])
data["_nb_mode"] = torch.tensor([1])

# Create TT-Lang wrapper
print("\nInitializing TT-Lang (hardware)...")
wrapper = TTLangAIMNet2.from_pytorch(calc.model, BackendMode.TT_HW)
print("Done.")

# Warmup
print("\nWarmup...")
reset_profile_stats()
out = wrapper.forward(data)
print(f"Warmup energy: {out['energy'].item():.6f} eV")

# Timed run WITH profiling (sync should happen due to _PROFILE=True)
print("\nTimed run (with profiling enabled)...")
reset_profile_stats()
t0 = perf_counter()
out = wrapper.forward(data)
t1 = perf_counter()

total_ms = (t1 - t0) * 1000
print(f"Total wall-clock time: {total_ms:.2f} ms")

# Get profile stats
stats = get_profile_stats()
print(f"\nProfile stats:")
for key, vals in stats.items():
    print(f"  {key}: mean={vals['mean_ms']:.2f} ms, total={vals['total_ms']:.2f} ms")

# Compare: total time vs sum of stages
stage_sum = sum(v["total_ms"] for v in stats.values())
print(f"\nTotal wall-clock: {total_ms:.2f} ms")
print(f"Sum of stages: {stage_sum:.2f} ms")
print(f"Difference: {abs(total_ms - stage_sum):.2f} ms")

# Check accuracy
if abs(total_ms - stage_sum) < 50:  # Allow some overhead
    print("\n✓ Profiling timing looks ACCURATE (synchronization is working)")
    print("  - device.core should now show ~469ms (actual computation)")
    print("  - device.download should now show <1ms (actual download)")
else:
    print("\n✗ Profiling timing is INACCURATE (async execution not synchronized)")
    print("  - Check that _PROFILE=True and sync code is executing")

wrapper.close()
