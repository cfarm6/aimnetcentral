from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass
class LinearWeights:
    weight: Tensor
    bias: Tensor | None
    weight_t_bf16: Tensor | None = None
    bias_bf16: Tensor | None = None


@dataclass
class MLPWeights:
    layers: list[LinearWeights]
    last_linear: bool


@dataclass
class TTLinearWeights:
    weight: Any
    bias: Any | None


@dataclass
class TTMLPWeights:
    layers: list[TTLinearWeights]
    last_linear: bool
    out_features: int


def _prepare_linear_host_tensors(layer: LinearWeights) -> None:
    layer.weight_t_bf16 = layer.weight.to(dtype=torch.bfloat16).T.contiguous()
    layer.bias_bf16 = layer.bias.to(dtype=torch.bfloat16) if layer.bias is not None else None


def _extract_mlp_weights(mlp: nn.Sequential) -> MLPWeights:
    layers: list[LinearWeights] = []
    for module in mlp:
        if isinstance(module, nn.Linear):
            bias = module.bias.detach().clone() if module.bias is not None else None
            layer = LinearWeights(weight=module.weight.detach().clone(), bias=bias)
            _prepare_linear_host_tensors(layer)
            layers.append(layer)
    return MLPWeights(layers=layers, last_linear=isinstance(mlp[-1], nn.Linear))


def extract_mlp_weights_from_state_dict(
    state_dict: dict[str, Tensor],
    prefix: str,
) -> MLPWeights | None:
    """Extract MLP linear weights from a state dict prefix such as ``mlps.0``."""
    weight_keys = sorted(k for k in state_dict if k.startswith(f"{prefix}.") and k.endswith(".weight"))
    if not weight_keys:
        return None

    layers: list[LinearWeights] = []
    for weight_key in weight_keys:
        layer_prefix = weight_key.removesuffix(".weight")
        bias_key = f"{layer_prefix}.bias"
        bias = state_dict[bias_key].clone() if bias_key in state_dict else None
        layer = LinearWeights(weight=state_dict[weight_key].clone(), bias=bias)
        _prepare_linear_host_tensors(layer)
        layers.append(layer)

    last_idx = int(weight_keys[-1].split(".")[-2])
    activation_key = f"{prefix}.{last_idx + 1}.weight"
    last_linear = activation_key not in state_dict
    return MLPWeights(layers=layers, last_linear=last_linear)


def extract_aimnet2_mlp_weights(model: nn.Module) -> dict[str, MLPWeights]:
    """Extract all MLP weight bundles from an AIMNet2 model."""
    weights: dict[str, MLPWeights] = {}

    if hasattr(model, "mlps"):
        for idx, mlp in enumerate(model.mlps):
            if isinstance(mlp, nn.Sequential):
                weights[f"mlps.{idx}"] = _extract_mlp_weights(mlp)

    outputs = getattr(model, "outputs", None)
    if outputs is not None:
        energy_mlp = getattr(outputs, "energy_mlp", None)
        if energy_mlp is not None and hasattr(energy_mlp, "mlp") and isinstance(energy_mlp.mlp, nn.Sequential):
            weights["outputs.energy_mlp"] = _extract_mlp_weights(energy_mlp.mlp)

    return weights


def upload_mlp_weights_to_device(weights: MLPWeights, device: Any) -> TTMLPWeights:
    """Upload preprocessed MLP weights to Tenstorrent device memory once."""
    import ttnn

    tt_layers: list[TTLinearWeights] = []
    for layer in weights.layers:
        weight_t = layer.weight_t_bf16
        if weight_t is None:
            _prepare_linear_host_tensors(layer)
            weight_t = layer.weight_t_bf16
        weight_tt = ttnn.from_torch(
            weight_t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        bias_tt = None
        if layer.bias_bf16 is not None:
            bias_tt = ttnn.from_torch(
                layer.bias_bf16,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        tt_layers.append(TTLinearWeights(weight=weight_tt, bias=bias_tt))

    out_features = weights.layers[-1].weight.shape[0]
    return TTMLPWeights(layers=tt_layers, last_linear=weights.last_linear, out_features=out_features)
