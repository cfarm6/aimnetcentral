from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from aimnet.ttlang.weights import TTMLPWeights, upload_mlp_weights_to_device


@dataclass
class TTDeviceWeights:
    """All non-MLP model weights uploaded to Tenstorrent device memory."""

    # Embedding weights (num_embeddings, embedding_dim)
    afv_weight: Any

    # AEV scalar basis constants
    rc_s: Any
    eta_s: Any
    shifts_s: Any

    # ConvSV projection matrices
    agh_a: Any
    agh_q: Any

    # AEV vector basis constants (optional, None when dual_basis=False)
    rc_v: Any | None = None
    eta_v: Any | None = None
    shifts_v: Any | None = None

    # Output atomic shifts (num_embeddings, 1)
    atomic_shifts: Any | None = None

@dataclass
class TTDeviceState:
    """Mutable device-resident tensors that persist across MLP passes."""

    a: Any
    charges: Any

@dataclass
class TTFullWeights:
    """Complete weight bundle for device-resident AIMNet2 inference."""

    device_weights: TTDeviceWeights
    mlp_weights: dict[str, TTMLPWeights]


def extract_device_weights(model: nn.Module, device: Any) -> TTDeviceWeights:
    """Extract non-MLP weights from an AIMNet2 model and upload to device."""
    import ttnn

    aev = model.aev
    afv_weight = model.afv.weight

    # AEV constants
    rc_s = ttnn.from_torch(
        aev.rc_s.detach().to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    eta_s = ttnn.from_torch(
        aev.eta_s.detach().to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    shifts_s = ttnn.from_torch(
        aev.shifts_s.detach().to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    if getattr(aev, "_dual_basis", False):
        rc_v = ttnn.from_torch(
            aev.rc_v.detach().to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        eta_v = ttnn.from_torch(
            aev.eta_v.detach().to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        shifts_v = ttnn.from_torch(
            aev.shifts_v.detach().to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    else:
        rc_v = None
        eta_v = None
        shifts_v = None

    # ConvSV agh matrices
    agh_a = ttnn.from_torch(
        model.conv_a.agh.detach().to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    agh_q = ttnn.from_torch(
        model.conv_q.agh.detach().to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Atomic shifts from output modules
    atomic_shifts: Any | None = None
    for module in model.outputs.children():
        if hasattr(module, "shifts"):
            atomic_shifts = ttnn.from_torch(
                module.shifts.weight.detach().to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            break

    afv_weight_tt = ttnn.from_torch(
        afv_weight.detach().to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return TTDeviceWeights(
        afv_weight=afv_weight_tt,
        rc_s=rc_s,
        eta_s=eta_s,
        shifts_s=shifts_s,
        rc_v=rc_v,
        eta_v=eta_v,
        shifts_v=shifts_v,
        agh_a=agh_a,
        agh_q=agh_q,
        atomic_shifts=atomic_shifts,
    )


def extract_full_weights(model: nn.Module, device: Any) -> TTFullWeights:
    """Extract all weights (device + MLP) from an AIMNet2 model."""
    from aimnet.ttlang.weights import extract_aimnet2_mlp_weights

    device_weights = extract_device_weights(model, device)
    mlp_weights_dict = extract_aimnet2_mlp_weights(model)
    mlp_weights = {
        key: upload_mlp_weights_to_device(weights, device)
        for key, weights in mlp_weights_dict.items()
    }
    return TTFullWeights(device_weights=device_weights, mlp_weights=mlp_weights)
