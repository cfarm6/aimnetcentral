# AIMNet2 Full-Device TTNN Performance Issues

Baseline: full-device batch-50 profile `476.18 ms` (hybrid baseline: `115.84 ms`, ~4x gap).
Current after fixes #1-6: **455.4 ms**.
Correctness: `0.0 kcal/mol` across all benchmark systems.

## Current Profile (batch-50, 1544 atoms, 455.4 ms)

```
Stage                       ms       What
──────────────────────────────────────────────────────────
device.sync              443.79     actual device compute
device.core                9.88     kernel launch overhead
device.download            0.44     data transfer (50 scalars)
device.prepare_input       0.91     host upload
```

Device compute (444 ms) spread across ~40 async kernel launches. None individually dominate.

## Fixed Issues

| # | Issue | Fix | Saving |
|---|-------|-----|--------|
| 1 | delta_a host round-trip | `ttnn.reshape` works in TILE | bundled |
| 2 | "download 139ms" | profiling bug (stale `_PROFILE` import) | N/A |
| 3 | excessive `to_layout` | removed 7 of 8 calls; reshape/split/gelu preserve TILE | ~14 ms |
| 4 | scatter_add ROW_MAJOR | squeeze (N,1)→(N,) before mol_sum, avoid repeat | ~5 ms |
| 5 | ConvSV contract 96MB intermediate | replaced `mul`+`sum` with `ttnn.matmul` | ~1 ms |
| 6 | ConvSV agh_proj 27MB intermediate | replaced with `ttnn.matmul` | ~1 ms |

## Open Issues

| # | Issue | Est. Impact |
|---|-------|-------------|
| 8 | padding mask per MLP pass | ~10–20 ms |
| 9 | redundant typecast checks | <1 ms |

## Key Finding

**Individual op speedups don't translate linearly to total time.** Device operations are asynchronous and pipelined — they execute concurrently. Speedups of 1.2–1.9× on individual ops produce only ~1 ms total savings because the sped-up op wasn't on the critical path.

The 444 ms device time is dominated by **kernel launch count** (~40 launches per forward), not compute or memory bandwidth per op. Each launch has fixed dispatch overhead.

## Path to Hybrid Baseline (115 ms)

Requires **TT-Lang kernel fusion** — combining multiple operations into single kernel launches:
- Fuse ConvSV contract + agh_proj + split + flatten into 1–2 kernels (currently ~10 launches)
- Fuse update_q split + mol_sum + nse + delta_a into 1–2 kernels (currently ~15 launches)
- Fuse MLP linear + gelu pairs where possible

Target: reduce ~40 launches to ~15, saving ~25 × ~10ms = 250 ms.

## Verification

```bash
uv run python benchmarks/ttlang_aimnet2_profile.py --mode hw --focus batch1000 \
    --batch-size 50 --warmup 1 --repeats 1 --profile-mlp --profile-stages
uv run pytest tests/test_ttlang_aimnet2.py -q -k "not tt_hw"
```

Profile data: `benchmarks/data/ttlang_aimnet2_profile_hw_latest.json`
