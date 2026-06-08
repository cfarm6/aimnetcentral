# AIMNet2 TT-Lang Examples

Prerequisites:

```bash
uv sync --extra tt
uv run tt-lang-setup
```

Forces and hessians are not supported on the tt-lang path. Requesting them raises `DerivativeNotImplementedError`.

## Examples

| Script | Description |
|--------|-------------|
| `energy_single_point.py` | Energy + charges for caffeine |
| `energy_batch.py` | Batch inference over taxol conformations |
| `compare_pytorch.py` | PyTorch reference vs TT-Lang energy comparison |
| `run_simulator.py` | Simulator-oriented entry point with timing |
| `run_hardware.py` | Hardware entry point (requires `/dev/tenstorrent/0`) |

## Run commands

Simulator path (host torch MLP, no hardware):

```bash
uv run python examples/ttlang/run_simulator.py
uv run python examples/ttlang/energy_single_point.py
```

Note: `ttlang-sim` is intended for standalone `@ttl.operation` scripts (see `tutorials/`). AimNet examples load PyTorch model weights and should be run with `uv run python`.

Hardware (Blackhole):

```bash
uv run python examples/ttlang/run_hardware.py
uv run python examples/ttlang/energy_single_point.py
```

Reference (PyTorch baseline):

```bash
uv run python examples/ttlang/compare_pytorch.py
```
