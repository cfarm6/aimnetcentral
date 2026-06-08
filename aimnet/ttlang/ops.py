from __future__ import annotations

import os
from time import perf_counter
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from aimnet.ttlang.weights import MLPWeights, TTMLPWeights

_PROFILE = os.environ.get("AIMNET_TTLANG_PROFILE", "").lower() in {"1", "true", "yes"}
_PROFILE_STATS: dict[str, list[float]] = {}


def set_profile_enabled(enabled: bool) -> None:
    global _PROFILE
    _PROFILE = enabled


def reset_profile_stats() -> None:
    _PROFILE_STATS.clear()


def get_profile_stats() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for key, samples in _PROFILE_STATS.items():
        if not samples:
            continue
        out[key] = {
            "mean_ms": sum(samples) / len(samples),
            "total_ms": sum(samples),
            "count": float(len(samples)),
        }
    return out

class StageProfile:
    """Context manager for profiling a named stage."""

    def __init__(self, key: str) -> None:
        self.key = key
        self.t0 = 0.0

    def __enter__(self) -> "StageProfile":
        if _profile_enabled():
            self.t0 = perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        if _profile_enabled():
            _record_profile(self.key, perf_counter() - self.t0)


def stage_profile(key: str) -> StageProfile:
    """Return a StageProfile context manager for the given key."""
    return StageProfile(key)


def _profile_enabled() -> bool:
    return _PROFILE


def _record_profile(key: str, elapsed_s: float) -> None:
    if _profile_enabled():
        _PROFILE_STATS.setdefault(key, []).append(elapsed_s * 1000.0)


def run_mlp_torch(x: Tensor, weights: MLPWeights) -> Tensor:
    """Run an MLP on host PyTorch."""
    out = x
    n_layers = len(weights.layers)
    for i, layer in enumerate(weights.layers):
        out = F.linear(out, layer.weight, layer.bias)
        if not (weights.last_linear and i == n_layers - 1):
            out = F.gelu(out)
    return out


def _to_ttnn_tensor(
    tensor: Tensor,
    device: Any,
    *,
    layout: Any,
) -> Any:
    import ttnn

    return ttnn.from_torch(
        tensor,
        dtype=ttnn.bfloat16,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _run_mlp_ttnn_uncached(x_tt: Any, weights: MLPWeights, device: Any) -> Any:
    import ttnn

    n_layers = len(weights.layers)
    for i, layer in enumerate(weights.layers):
        weight_t = layer.weight_t_bf16
        if weight_t is None:
            weight_t = layer.weight.to(dtype=torch.bfloat16).T.contiguous()
        weight_tt = _to_ttnn_tensor(weight_t, device, layout=ttnn.TILE_LAYOUT)

        t0 = perf_counter()
        if layer.bias_bf16 is not None:
            bias_tt = _to_ttnn_tensor(layer.bias_bf16, device, layout=ttnn.ROW_MAJOR_LAYOUT)
            x_tt = ttnn.linear(x_tt, weight_tt, bias=bias_tt)
        else:
            x_tt = ttnn.matmul(x_tt, weight_tt)
        _record_profile("mlp.linear", perf_counter() - t0)

        if not (weights.last_linear and i == n_layers - 1):
            t0 = perf_counter()
            x_tt = ttnn.gelu(x_tt)
            _record_profile("mlp.gelu", perf_counter() - t0)

    return x_tt


def _run_mlp_ttnn_cached(x_tt: Any, weights: TTMLPWeights) -> Any:
    import ttnn

    n_layers = len(weights.layers)
    for i, layer in enumerate(weights.layers):
        t0 = perf_counter()
        if layer.bias is not None:
            x_tt = ttnn.linear(x_tt, layer.weight, bias=layer.bias)
        else:
            x_tt = ttnn.matmul(x_tt, layer.weight)
        _record_profile("mlp.linear", perf_counter() - t0)

        if not (weights.last_linear and i == n_layers - 1):
            t0 = perf_counter()
            x_tt = ttnn.gelu(x_tt)
            _record_profile("mlp.gelu", perf_counter() - t0)
    return x_tt

def run_mlp_ttnn_device(x_tt: Any, weights: MLPWeights | TTMLPWeights, device: Any) -> Any:
    """Run an MLP on device with ttnn linear/gelu. Returns a ttnn tensor."""
    if isinstance(weights, TTMLPWeights):
        return _run_mlp_ttnn_cached(x_tt, weights)
    return _run_mlp_ttnn_uncached(x_tt, weights, device)


def run_mlp_ttnn(
    x: Tensor,
    weights: MLPWeights | TTMLPWeights,
    device: Any | None,
) -> Tensor:
    """Run an MLP with ttnn linear/gelu, falling back to host PyTorch."""
    if device is None:
        if isinstance(weights, TTMLPWeights):
            raise TypeError("TTMLPWeights require a Tenstorrent device.")
        return run_mlp_torch(x, weights)

    orig_shape = x.shape
    in_features = orig_shape[-1]
    x2d = x.reshape(-1, in_features)

    try:
        import ttnn

        t0 = perf_counter()
        x_tt = _to_ttnn_tensor(x2d.to(torch.bfloat16), device, layout=ttnn.TILE_LAYOUT)
        _record_profile("mlp.input_upload", perf_counter() - t0)

        if isinstance(weights, TTMLPWeights):
            x_tt = _run_mlp_ttnn_cached(x_tt, weights)
            out_features = weights.out_features
        else:
            x_tt = _run_mlp_ttnn_uncached(x_tt, weights, device)
            out_features = weights.layers[-1].weight.shape[0]

        t0 = perf_counter()
        out2d = ttnn.to_torch(x_tt).to(dtype=x.dtype, device=x.device)
        _record_profile("mlp.output_download", perf_counter() - t0)

        return out2d.reshape(*orig_shape[:-1], out_features)
    except Exception:
        if isinstance(weights, MLPWeights):
            return run_mlp_torch(x, weights)
        raise
