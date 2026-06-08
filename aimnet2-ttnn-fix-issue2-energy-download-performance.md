# Fix Energy Download Performance Issue - ANALYSIS ONLY

## Issue (Original)
`device.download` takes 139 ms for final energy in `_forward_tt_full_device`.
- Location: `aimnet/ttlang/model.py:_forward_tt_full_device` (line 286-287)
- For batch-50 mode 1, `out["energy"]` is shape (50,) but in `TILE_LAYOUT` with padded shape (64,)
- 139 ms to download 50 bf16 scalars is ~1000× slower than expected

## Root Cause Analysis (COMPREHENSIVE)

**UPDATE: This is NOT a performance issue. It's a measurement error.**

### Why Download Appeared Slow
TT-Lang operations are **asynchronous**. Without explicit synchronization:
1. `device.core` stage launches kernels and returns immediately (~10ms for kernel launch)
2. Computation runs asynchronously on device (actually ~469ms)
3. `device.download` calls `ttnn.to_torch()` which **blocks until computation completes**
4. Measured time without sync: `device.core`=10ms, `device.download`=459ms
5. **Actual time with sync**: `device.core`=469ms (computation), `device.download`=<1ms

### Verification with Synchronization
Added `ttnn.synchronize_device()` before `device.download` staging:
- `device.core`: 469ms (actual computation time)
- `device.download`: **0.34ms** ✓ (target: < 5ms)

**The download performance is already excellent (< 1ms for 50 scalars).**

## Investigation: ROW_MAJOR_LAYOUT Change

### Attempted Fix
Changed `ttnn_mol_sum()` in `aimnet/ttlang/ttnn_nbops.py` (lines 113, 117) to use `layout=ttnn.ROW_MAJOR_LAYOUT` instead of `ttnn.TILE_LAYOUT`.

### Result: INCORRECT - Causes Energy Calculation Errors

**Test failures:**
```
E   AssertionError: energy error 33.319397 > 0.5 kcal/mol
E   assert np.float64(33.31939722361494) < 0.5
```

**Root cause:** `ttnn.scatter_add()` produces incorrect results when the output tensor is in ROW_MAJOR_LAYOUT. The operation expects TILE_LAYOUT for correct execution.

**Conclusion:** MUST keep TILE_LAYOUT for `ttnn_mol_sum()` output tensor.

## Actual Performance

| Metric | Measured Time (no sync) | Actual Time (with sync) | Status |
|--------|---------------|--------------|--------|
| `device.download` | 459ms (includes async computation) | **0.34ms** | ✓ EXCELLENT |
| `device.core` | 10ms (kernel launch only) | 469ms | See note |
| Energy accuracy | 0.0 kcal/mol error | 0.0 kcal/mol error | ✓ PASSED |

**Note:** The `device.core` computation (469ms) is the actual bottleneck, NOT download. Optimizing computation performance requires separate effort.

## Pre-existing Test Failures (UNRELATED)

The following tests fail even without my changes:
- `test_ttlang_default_tt_hw_works`
- `test_ttlang_try_full_device_falls_back`

**Verification:** Running tests on clean repo (git stash) shows same failures.

**These are pre-existing issues** in the repository, not caused by this analysis.

## Conclusion

**Issue RESOLVED - No Code Change Required.**

1. **Download performance is already excellent** (< 1ms for 50 scalars)
2. **The reported "slow download" was a measurement error** due to async execution
3. **ROW_MAJOR_LAYOUT change is NOT safe** - causes incorrect `scatter_add` results
4. **TILE_LAYOUT must be kept** for `ttnn_mol_sum()` output tensor

## Recommendations

### For Accurate Profiling
Add `ttnn.synchronize_device()` before measuring download time:
```python
with stage_profile("device.core"):
    out = self._device_core.forward(batch)
    ttnn.synchronize_device(self._ttnn_device)  # Add this

with stage_profile("device.download"):
    energy = ttnn.to_torch(out["energy"]).to(...)
```

**Note:** This is only for profiling accuracy. Remove in production code (sync hurts performance).

### For Computation Optimization (Separate Ticket)
The `device.core` computation (469ms for batch-50) is the actual bottleneck:
- Kernel fusion for MLP operations
- Better parallelization of scatter_add operations
- Memory access pattern optimization

### Files Modified
**NONE.** No code changes required. The issue was a measurement error, not a performance bug.

## Verification Command

```bash
uv run python benchmarks/ttlang_aimnet2_profile.py --mode hw --focus batch1000 \
    --batch-size 50 --warmup 1 --repeats 1 --profile-mlp --profile-stages
```

**Expected results:**
- Total time: ~470ms (includes async computation + download)
- Energy error: 0.0 kcal/mol ✓
- Actual download time: < 1ms ✓ (verified with sync)
