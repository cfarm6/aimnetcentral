from __future__ import annotations

import os
from typing import Any

import torch
from torch import Tensor, nn

from aimnet import nbops
from aimnet.ttlang.backend import BackendMode, TTBackendConfig
from aimnet.ttlang.device_model import TTDeviceAIMNet2Core
from aimnet.ttlang.errors import TTBackendUnavailableError
from aimnet.ttlang import ops as tt_ops
from aimnet.ttlang.ops import run_mlp_torch, run_mlp_ttnn, stage_profile
from aimnet.ttlang.weights import (
    MLPWeights,
    TTMLPWeights,
    extract_aimnet2_mlp_weights,
    upload_mlp_weights_to_device,
)


class TTLangAIMNet2:
    """AIMNet2 inference wrapper with optional Tenstorrent MLP acceleration."""
    def __init__(self, model: nn.Module, config: TTBackendConfig) -> None:
        self._model = model
        self._config = config
        self._mlp_weights: dict[str, MLPWeights] | None = None
        self._tt_mlp_weights: dict[str, TTMLPWeights] | None = None
        self._ttnn_device: Any | None = None
        self._device_core: TTDeviceAIMNet2Core | None = None
        if config.uses_ttnn:
            self._compile_host_modules()
            self._init_ttnn_device()
            self._upload_mlp_weights()
            # Always try to init device core; full-device is the default path.
            self._init_device_core(required=False)

    @classmethod
    def from_pytorch(
        cls,
        model: nn.Module,
        backend: BackendMode | str,
        *,
        device_id: int = 0,
    ) -> TTLangAIMNet2:
        if isinstance(backend, str):
            backend = BackendMode(backend)
        return cls(model, TTBackendConfig(mode=backend, device_id=device_id))

    def _init_ttnn_device(self) -> None:
        try:
            import ttnn

            self._ttnn_device = ttnn.open_device(device_id=self._config.device_id)
        except Exception as exc:
            raise TTBackendUnavailableError(
                f"Failed to open Tenstorrent device {self._config.device_id} for backend {self._config.mode!r}."
            ) from exc

    def _compile_host_modules(self) -> None:
        if os.environ.get("AIMNET_TTLANG_COMPILE_HOST", "1").lower() in {"0", "false", "no"}:
            return
        if getattr(self._model, "_ttlang_host_compiled", False):
            return
        for name in ("aev", "conv_a", "conv_q"):
            module = getattr(self._model, name, None)
            if module is None:
                continue
            try:
                setattr(self._model, name, torch.compile(module, mode="reduce-overhead"))
            except Exception:
                return
        setattr(self._model, "_ttlang_host_compiled", True)

    def _get_mlp_weights(self) -> dict[str, MLPWeights]:
        if self._mlp_weights is None:
            self._mlp_weights = extract_aimnet2_mlp_weights(self._model)
        return self._mlp_weights

    def _upload_mlp_weights(self) -> None:
        if self._ttnn_device is None:
            return
        self._tt_mlp_weights = {
            key: upload_mlp_weights_to_device(weights, self._ttnn_device)
            for key, weights in self._get_mlp_weights().items()
        }

    def _init_device_core(self, *, required: bool = False) -> None:
        if self._ttnn_device is None:
            return
        try:
            from aimnet.ttlang.device_weights import extract_full_weights

            weights = extract_full_weights(self._model, self._ttnn_device)
            self._device_core = TTDeviceAIMNet2Core(
                weights,
                self._ttnn_device,
                num_charge_channels=self._model.num_charge_channels,
                d2features=self._model.d2features,
                nshifts_s=self._model.nshifts_s,
                nshifts_v=self._model.conv_a.nshifts_v,
                nfeature=self._model.nfeature,
                ncomb_v=self._model.conv_a.ncomb_v,
            )
        except Exception as exc:
            self._device_core = None
            if required:
                raise TTBackendUnavailableError("Failed to initialize experimental full-device AIMNet2 core.") from exc

    def _data_to_device_batch(self, data: dict[str, Tensor]) -> Any:
        """Convert a prepared PyTorch data dict to a TTInputBatch on device."""
        import ttnn
        from aimnet.ttlang.device_data import TTInputBatch

        device = self._ttnn_device
        nb_mode = int(data["_nb_mode"].item())
        input_padded = bool(data["_input_padded"].item())

        # Compute n_molecules from mol_idx or batch shape without device download
        mol_idx_t = data.get("mol_idx")
        if nb_mode == 1 and mol_idx_t is not None:
            n_molecules = int(mol_idx_t[-1].item()) + 1
        elif nb_mode in (0, 2):
            n_molecules = data["charge"].shape[0] if data["charge"].dim() > 0 else 1
        else:
            n_molecules = 0


        def _to_tt(t: Tensor) -> Any:
            return ttnn.from_torch(
                t.to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        def _to_tt_int(t: Tensor) -> Any:
            """Upload integer index tensors in ROW_MAJOR as uint32 for scatter/gather."""
            return ttnn.from_torch(
                t.to(torch.int32).to(torch.uint32),
                dtype=ttnn.uint32,
                device=device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        batch = TTInputBatch(
            coord=_to_tt(data["coord"]),
            numbers=_to_tt(data["numbers"].to(torch.int64)),
            charge=_to_tt(data["charge"]),
            nb_mode=nb_mode,
            input_padded=input_padded,
            n_molecules=n_molecules,
            mol_idx=_to_tt_int(data["mol_idx"]) if "mol_idx" in data else None,
            mol_sizes=_to_tt(data["mol_sizes"]) if "mol_sizes" in data else None,
            natom=_to_tt(data["_natom"]) if "_natom" in data else None,
            nbmat=_to_tt(data["nbmat"].to(torch.int64)) if "nbmat" in data else None,
            mask_i=_to_tt(data["mask_i"]) if "mask_i" in data else None,
            mask_ij=_to_tt(data["mask_ij"]) if "mask_ij" in data else None,
            shifts=_to_tt(data["shifts"]) if "shifts" in data else None,
        )
        return batch

    def _run_mlp(self, key: str, x: Tensor, fallback_mlp: nn.Module) -> Tensor:
        if self._config.uses_ttnn and self._tt_mlp_weights is not None:
            tt_weights = self._tt_mlp_weights.get(key)
            if tt_weights is not None:
                return run_mlp_ttnn(x, tt_weights, self._ttnn_device)

        weights = self._get_mlp_weights().get(key)
        if weights is None:
            return fallback_mlp(x)
        if self._config.uses_ttnn:
            return run_mlp_ttnn(x, weights, self._ttnn_device)
        if self._config.uses_torch_mlp:
            return run_mlp_torch(x, weights)
        return fallback_mlp(x)

    def forward(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        if self._config.mode is BackendMode.REFERENCE:
            return self._model(data)

        nb_mode = int(data.get("_nb_mode", torch.tensor(0)).item())
        force_full = _env_flag("AIMNET_TTLANG_FULL_DEVICE")
        allow_fallback = _env_flag("AIMNET_TTLANG_ALLOW_HYBRID_FALLBACK", default=False)

        if force_full:
            # Forced full-device: fail if unsupported mode
            if nb_mode == 2:
                raise RuntimeError(
                    "Mode 2 (padded batch) is not supported in forced full-device mode. "
                    "Set AIMNET_TTLANG_ALLOW_HYBRID_FALLBACK=1 to use hybrid fallback."
                )
            if self._device_core is None:
                self._init_device_core(required=True)
            return self._forward_tt_full_device(data, _nb_mode=nb_mode)

        if self._device_core is not None:
            try:
                return self._forward_tt_full_device(data, _nb_mode=nb_mode)
            except Exception:
                if allow_fallback:
                    return self._forward_tt(data)
                raise

        return self._forward_tt(data)

    def _forward_tt(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        model = self._model
        with stage_profile("host.prepare_input"):
            data = model.prepare_input(data)

        with stage_profile("host.afv"):
            a: Tensor = model.afv(data["numbers"])
            if model.d2features:
                a = a.unflatten(-1, (model.nfeature, model.nshifts_s))
            data["a"] = a

        with stage_profile("host.preprocess_charge"):
            if model.num_charge_channels == 2:
                data = model._preprocess_spin_polarized_charge(data)
            else:
                data["charge"] = data["charge"].unsqueeze(-1)

        with stage_profile("host.aev"):
            data = model.aev(data)

        _npass = len(model.mlps)
        for ipass, mlp in enumerate(model.mlps):
            with stage_profile(f"host.prepare_in.pass_{ipass}"):
                if ipass == 0:
                    _in = model._prepare_in_a(data)
                else:
                    _in = torch.cat([model._prepare_in_a(data), model._prepare_in_q(data)], dim=-1)

            with stage_profile(f"mlp.pass_{ipass}"):
                _out = self._run_mlp(f"mlps.{ipass}", _in, mlp)
            if data["_input_padded"].item():
                _out = nbops.mask_i_(_out, data, mask_value=0.0)

            with stage_profile(f"host.update_q.pass_{ipass}"):
                if ipass == 0:
                    data = model._update_q(data, _out, delta_q=False)
                elif ipass < _npass - 1:
                    data = model._update_q(data, _out, delta_q=True)
                else:
                    data["aim"] = _out

        with stage_profile("host.postprocess_charge"):
            if model.num_charge_channels == 2:
                data = model._postprocess_spin_polarized_charge(data)
            else:
                data["charges"] = data["charges"].squeeze(-1)
                data["charge"] = data["charge"].squeeze(-1)

        for name, module in _iter_output_modules(model.outputs):
            with stage_profile(f"host.output.{name}"):
                if name == "energy_mlp" and hasattr(module, "mlp"):
                    v = self._run_mlp("outputs.energy_mlp", data[module.key_in], module.mlp).squeeze(-1)
                    if data["_input_padded"].item():
                        v = nbops.mask_i_(v, data, mask_value=0.0)
                    data[module.key_out] = v
                else:
                    data = module(data)

        data["_ttlang_backend_path"] = "hybrid-fallback"
        return data
    def _forward_tt_full_device(self, data: dict[str, Tensor], _nb_mode: int = 0) -> dict[str, Tensor]:
        """Full device-resident energy forward path."""
        import ttnn

        if _nb_mode == 2 and _env_flag("AIMNET_TTLANG_FULL_DEVICE") and not _env_flag("AIMNET_TTLANG_ALLOW_HYBRID_FALLBACK"):
            raise RuntimeError(
                "Mode 2 (padded batch) is not supported in forced full-device mode. "
                "Set AIMNET_TTLANG_ALLOW_HYBRID_FALLBACK=1 to use hybrid fallback."
            )

        with stage_profile("device.prepare_input"):
            data = self._model.prepare_input(data)
            batch = self._data_to_device_batch(data)
        with stage_profile("device.core"):
            out = self._device_core.forward(batch)

        # Synchronize to measure actual device compute time separately from download.
        with stage_profile("device.sync"):
            if tt_ops._PROFILE:
                ttnn.synchronize_device(self._ttnn_device)

        with stage_profile("device.download"):
            energy = ttnn.to_torch(out["energy"]).to(dtype=data["coord"].dtype, device=data["coord"].device)

        data["energy"] = energy
        data["_ttlang_backend_path"] = "full-device"
        return data
    def close(self) -> None:
        if self._ttnn_device is not None:
            import ttnn

            ttnn.close_device(self._ttnn_device)
            self._ttnn_device = None

    def __del__(self) -> None:
        self.close()


def _iter_output_modules(outputs: nn.ModuleList | nn.ModuleDict) -> list[tuple[str, nn.Module]]:
    if isinstance(outputs, nn.ModuleDict):
        return list(outputs.items())
    return [(str(i), module) for i, module in enumerate(outputs)]


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}
