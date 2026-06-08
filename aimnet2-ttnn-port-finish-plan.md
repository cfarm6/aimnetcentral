# Plan: Finish AIMNet2 TTNN Port

## Purpose

Finish the AIMNet2 TT hardware backend so supported inference runs through TTNN/TT-Lang instead of PyTorch for the model forward path. This plan is an execution blueprint only; do not execute it as part of plan creation.

## Current State

The recovered working path is a hybrid backend:

- `TTLangAIMNet2` opens a TTNN device, compiles selected host PyTorch modules, uploads MLP weights, and defaults to `_forward_tt` in `aimnet/ttlang/model.py`.
- The stable `TT_HW` path still uses PyTorch for input preparation, embedding, AEV, ConvSV preparation, charge update, masking, output modules, and final reductions.
- TTNN is currently reliable for MLP execution through `run_mlp_ttnn` and `run_mlp_ttnn_device` in `aimnet/ttlang/ops.py`.
- A first full-device core exists in `aimnet/ttlang/device_model.py`, but it is experimental and opt-in via `AIMNET_TTLANG_FULL_DEVICE` or `AIMNET_TTLANG_TRY_FULL_DEVICE`.
- TTNN module-level tests exist for nbops, AEV, charge update, ConvSV, and output energy under `tests/test_ttnn_*.py`.
- The latest recovered batch-50 profile reported `115.843803 ms`, `431.615664 molecules/s`, and `0.0 kcal/mol` energy error. The dominant measured stages were host AEV and host ConvSV input preparation.

Known host hotspots from the repaired profile:

- `host.aev`: `55.293863 ms`
- `host.prepare_in.pass_0`: `25.530291 ms`
- `host.prepare_in.pass_1`: `13.379624 ms`
- `host.prepare_in.pass_2`: `12.705730 ms`
- `mlp.input_upload`: `2.709432 ms` total across four MLP invocations

## Scope

Initial completion scope:

- AIMNet2 energy inference only.
- `BackendMode.TT_HW` first.
- Batch sizes `<= 50`.
- Mode 0 dense/single inputs and mode 1 flat prepared batches.
- Existing HW tolerance: `energy_error_kcal_mol <= 0.5`.
- CPU-side neighbor-list construction is acceptable for the first finished TTNN model path if device-side construction does not pass a prototype gate.

Out of scope for the first completion milestone:

- Forces, Hessians, autograd, and training.
- Mode 2 padded batches.
- Batch sizes above 50.
- Making the existing experimental full-device path default before forced full-device parity and performance are proven.

## Target Architecture

Keep the stable hybrid path as a fallback while finishing a strict full-device core:

1. Host performs input conversion and, if needed, CPU neighbor-list construction.
2. Inputs are uploaded once into a `TTInputBatch`.
3. All model-forward operations after upload run on TTNN/TT-Lang tensors:
   - embedding,
   - AEV distance and basis expansion,
   - ConvSV for atomic and charge features,
   - iterative MLPs,
   - charge update/NSE,
   - output MLP,
   - atomic shift,
   - molecule reductions.
4. Only final output tensors are downloaded.
5. `AIMNET_TTLANG_FULL_DEVICE=1` forces the full-device path and fails if unsupported.
6. Default `TT_HW` switches to full-device only after forced mode passes correctness and performance gates.

## Acceptance Criteria

Correctness:

- `uv run pytest tests/test_ttlang_aimnet2.py::test_ttlang_energy_parity_reference -q` passes.
- `uv run pytest tests/test_ttlang_aimnet2.py::test_ttlang_energy_parity_hw -q` passes.
- `uv run pytest tests/test_ttnn_nbops.py tests/test_ttnn_aev.py tests/test_ttnn_charge.py tests/test_ttnn_convsv.py tests/test_ttnn_outputs.py -q` passes.
- New forced-full-device tests pass for mode 0 and mode 1.
- Batch-50 energy error is `<= 0.5 kcal/mol`.

Performance:

- The required evaluator remains single-pass:
  `uv run python benchmarks/ttlang_aimnet2_profile.py --mode hw --focus batch1000 --batch-size 50 --warmup 1 --repeats 1 --profile-mlp --profile-stages --output .omx/goals/performance/ttlang-aimnet-runtime/evaluator-batch50-full-device.json`
- The finished full-device path should beat the recovered hybrid batch-50 runtime of `115.843803 ms`.
- The profile artifact must show no `host.aev`, `host.prepare_in.*`, `host.update_q.*`, or intermediate `mlp.input_upload` / `mlp.output_download` entries inside the measured model-forward core.

Fallback behavior:

- Default `TT_HW` may fall back only while full-device support is under development.
- `AIMNET_TTLANG_FULL_DEVICE=1` must not silently fall back unless `AIMNET_TTLANG_ALLOW_HYBRID_FALLBACK=1` is explicitly set.

## Implementation Phases

### Phase 0: Lock The Recovered Baseline

Files:

- `aimnet/ttlang/model.py`
- `benchmarks/ttlang_aimnet2_profile.py`
- `benchmarks/ttlang_benchmark_common.py`
- `.omx/goals/performance/ttlang-aimnet-runtime/evaluator-batch50-repair.json`

Steps:

1. Preserve the recovered hybrid path as the fallback implementation.
2. Add a checked-in or documented baseline profile summary for batch 50.
3. Add a forced-full-device smoke test that currently may be marked expected-fail until the full device core is repaired.
4. Add a fallback assertion test:
   - default `TT_HW` works,
   - forced full-device fails loudly when unsupported,
   - try-full-device can fall back only when configured.

Exit criteria:

- Current recovered behavior is protected by tests.
- Any future regression in default small-batch `TT_HW` is caught before full-device work continues.

### Phase 1: Make Device Batch Metadata Complete

Files:

- `aimnet/ttlang/device_data.py`
- `aimnet/ttlang/model.py`
- `aimnet/ttlang/ttnn_nbops.py`

Steps:

1. Extend `TTInputBatch` with host metadata that currently causes device-to-host round trips:
   - `n_molecules`,
   - `n_atoms`,
   - `padding_atom_index`,
   - `batch_shape`,
   - `feature_shapes`.
2. Populate those fields in `_data_to_device_batch`.
3. Remove host shape discovery from runtime TTNN helpers where possible.
4. For mode 1, pass output segment size directly into segmented reductions instead of computing it with `ttnn.to_torch(mol_idx_tt)`.

Exit criteria:

- `ttnn_mol_sum` does not download `mol_idx` for output shape discovery.
- Mode 0 and mode 1 nbops tests still pass.

### Phase 2: Repair AEV For Forced Full-Device

Files:

- `aimnet/ttlang/ttnn_aev.py`
- `aimnet/ttlang/device_model.py`
- `tests/test_ttnn_aev.py`

Steps:

1. Require `mask_ij` from `TTInputBatch` for full-device mode instead of constructing masks in `ttnn_calc_distances`.
2. Remove PyTorch mask creation from `ttnn_calc_distances` during forced full-device forward.
3. Validate mode 0 and mode 1 distance tensors against `aimnet.ops.calc_distances`.
4. Validate full `g_sv` against `AEVSV._calc_aev` for:
   - water mode 0,
   - caffeine mode 1,
   - mixed batch-50 mode 1.
5. Record per-stage maximum absolute error.

Exit criteria:

- AEV tests pass for mode 0 and mode 1.
- Forced full-device profile no longer has `host.aev` in the measured model-forward core.

### Phase 3: Repair ConvSV For Mode 1 First

Files:

- `aimnet/ttlang/ttnn_convsv.py`
- `aimnet/ttlang/device_model.py`
- `tests/test_ttnn_convsv.py`

Steps:

1. Treat mode 1 flat batches as the primary ConvSV target.
2. Validate tensor layout assumptions for:
   - `d2features=True` atomic features,
   - `d2features=False` charge features.
3. Replace fragile reshape/permutation paths with one clear implementation per shape family.
4. Compare `conv_a` and `conv_q` outputs against PyTorch `ConvSV` for small controlled tensors and real AIMNet shapes.
5. If primitive TTNN matmul/reshape is slower than host ConvSV for batch 50, design a repo-local TT-Lang kernel that fuses:
   - neighbor gather,
   - scalar radial accumulation,
   - vector projection through `agh`,
   - square/sum accumulation.

Exit criteria:

- Mode 1 ConvSV parity passes for real AIMNet feature shapes.
- Batch-50 profile removes `host.prepare_in.pass_*` from the measured model-forward core.

### Phase 4: Keep Iterative State On Device

Files:

- `aimnet/ttlang/device_model.py`
- `aimnet/ttlang/ops.py`
- `aimnet/ttlang/ttnn_charge.py`
- `tests/test_ttnn_charge.py`

Steps:

1. Ensure all iterative MLP inputs and outputs stay TTNN tensors.
2. Ensure `_prepare_in_a`, `_prepare_in_q`, `_run_mlp`, and `ttnn_update_q` do not download intermediates.
3. Pass molecule segment metadata into `ttnn_update_q` so mode 1 reductions remain device-side.
4. Validate charge conservation and `a` update against PyTorch for all passes.
5. Add a profile assertion that intermediate MLP upload/download counters are zero in forced full-device mode.

Exit criteria:

- No intermediate `mlp.input_upload` or `mlp.output_download` appears in forced full-device batch-50 profile.
- Charge update parity remains within tolerance.

### Phase 5: Finish Device Output Path

Files:

- `aimnet/ttlang/ttnn_outputs.py`
- `aimnet/ttlang/device_model.py`
- `tests/test_ttnn_outputs.py`
- `tests/test_ttlang_aimnet2.py`

Steps:

1. Keep `outputs.energy_mlp` device-resident.
2. Apply atomic shifts with TTNN embedding.
3. Sum per molecule on device using explicit segment metadata.
4. Download only final `energy`.
5. Return output keys and shapes matching the current wrapper.

Exit criteria:

- Energy output shape matches reference for mode 0 and mode 1.
- Forced full-device caffeine and batch-50 parity pass.

### Phase 6: Add PyTorch-Usage Guard

Files:

- `aimnet/ttlang/model.py`
- `aimnet/ttlang/ops.py`
- `tests/test_ttlang_aimnet2.py`

Steps:

1. Add an instrumentation context for forced full-device mode that records forbidden host stages.
2. Fail forced full-device tests if any of these stages run inside the measured core:
   - `host.aev`,
   - `host.prepare_in.*`,
   - `host.update_q.*`,
   - `host.output.*` other than final download,
   - intermediate MLP upload/download.
3. Keep `prepare_input` outside the forbidden region unless device-side neighbor-list construction is accepted later.

Exit criteria:

- Forced full-device test proves the model-forward core is TTNN-only after input upload.

### Phase 7: Switch Default TT_HW Path

Files:

- `aimnet/ttlang/model.py`
- `tests/test_ttlang_aimnet2.py`
- `benchmarks/ttlang_aimnet2_profile.py`

Steps:

1. Route supported mode 0 and mode 1 inputs through full-device by default.
2. Keep hybrid fallback behind `AIMNET_TTLANG_ALLOW_HYBRID_FALLBACK=1`.
3. Emit a clear profile/status field:
   - `backend_path: full-device`
   - `backend_path: hybrid-fallback`
4. Fail loudly for unsupported mode 2 in forced mode.

Exit criteria:

- Default `TT_HW` uses full-device for batch sizes `<= 50` in mode 0 and mode 1.
- Hybrid fallback is explicit, visible, and test-covered.

### Phase 8: Device-Side Neighbor Prototype Gate

Files:

- `aimnet/ttlang/ttnn_neighbors.py`
- optional `aimnet/ttlang/kernels/`
- `benchmarks/ttlang_aimnet2_profile.py`

Steps:

1. Prototype device-side neighbor-list construction only after the model-forward core is stable.
2. Compare against `AIMNet2Calculator.prepare_input` for exact index semantics.
3. Measure neighbor-prep time with `batch-size 50`, `warmup 1`, `repeats 1`.
4. Accept only if:
   - `nbmat` and masks match CPU semantics,
   - energy parity remains `<= 0.5 kcal/mol`,
   - runtime beats or matches CPU neighbor prep,
   - memory usage remains stable for batch 50.

Exit criteria:

- Written go/no-go record.
- If rejected, CPU neighbor prep remains the boundary for the first full TTNN model-forward milestone.

## Verification Matrix

Unit verification:

- `tests/test_ttnn_nbops.py`: masks, gathers, segmented sums.
- `tests/test_ttnn_aev.py`: distances, cutoff, radial expansion, `g_sv`.
- `tests/test_ttnn_convsv.py`: ConvSV mode 1 real shapes.
- `tests/test_ttnn_charge.py`: NSE and charge conservation.
- `tests/test_ttnn_outputs.py`: atomic shift and molecule energy sum.

Integration verification:

- `tests/test_ttlang_aimnet2.py::test_ttlang_energy_parity_reference`
- `tests/test_ttlang_aimnet2.py::test_ttlang_energy_parity_hw`
- new forced-full-device mode 0 parity test
- new forced-full-device mode 1 batch-50 parity test
- new no-forbidden-host-stage test

Performance verification:

- Always use one measured pass unless explicitly changed:
  `--warmup 1 --repeats 1`
- Always keep batch size `<= 50` for this milestone.
- Compare against recovered hybrid baseline:
  - baseline: `115.843803 ms`
  - target: lower than baseline with `0.0` to `<=0.5 kcal/mol` error

## Risks And Mitigations

- Risk: TTNN primitive ConvSV remains slower than PyTorch host einsum.
  - Mitigation: prototype primitive path first; write repo-local TT-Lang fused kernel only when primitive path cannot beat batch-50 baseline.

- Risk: segmented reductions require host shape discovery.
  - Mitigation: carry output segment sizes as host metadata in `TTInputBatch`; do not discover them by downloading device tensors.

- Risk: BF16 math drifts beyond HW tolerance.
  - Mitigation: use per-stage parity and keep sensitive reductions in FP32 where TTNN supports it.

- Risk: fallback hides incomplete porting.
  - Mitigation: forced full-device mode must fail on fallback and must assert forbidden host stages are absent.

- Risk: device-side neighbor construction consumes effort without improving the first milestone.
  - Mitigation: keep it behind Phase 8 prototype gate and do not block the model-forward TTNN milestone on it.

## Stop Conditions

Stop implementation when any of these are true:

- Forced full-device mode 0 and mode 1 batch-50 parity pass.
- Batch-50 full-device runtime beats recovered hybrid baseline.
- No PyTorch tensor operations run inside the measured full-device model-forward core.
- Default `TT_HW` uses full-device for supported inputs and falls back only when explicitly configured.

Do not expand to mode 2 or batches above 50 until the above stop conditions are satisfied.

