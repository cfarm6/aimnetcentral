from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from aimnet.ttlang.device_data import TTInputBatch
from aimnet.ttlang.device_weights import TTFullWeights
from aimnet.ttlang.ops import run_mlp_ttnn_device
from aimnet.ttlang.ttnn_aev import ttnn_calc_aev, ttnn_calc_distances
from aimnet.ttlang.ttnn_charge import ttnn_update_q
from aimnet.ttlang.ttnn_convsv import ttnn_conv_sv
from aimnet.ttlang.ttnn_nbops import ttnn_get_i, ttnn_mask_i_
from aimnet.ttlang.ttnn_outputs import ttnn_output_energy


class TTDeviceAIMNet2Core:
    """Device-resident AIMNet2 energy inference core.

    Owns all model weights in device memory and runs the full forward
    path without PyTorch tensor operations.
    """

    def __init__(
        self,
        weights: TTFullWeights,
        device: Any,
        *,
        num_charge_channels: int = 1,
        d2features: bool = False,
        nshifts_s: int = 16,
        nshifts_v: int = 16,
        nfeature: int = 64,
        ncomb_v: int = 16,
    ) -> None:
        self._weights = weights
        self._device = device
        self._num_charge_channels = num_charge_channels
        self._d2features = d2features
        self._nshifts_s = nshifts_s
        self._nshifts_v = nshifts_v
        self._nfeature = nfeature
        self._ncomb_v = ncomb_v

    def _prepare_in_a(self, a_tt: Any, g_sv_tt: Any, nb_mode: int, nbmat_tt: Any | None) -> Any:
        """Prepare input for the first MLP pass (atomic features)."""
        import ttnn

        dw = self._weights.device_weights
        a_i = ttnn_get_i(a_tt, nb_mode)
        avf_a = ttnn_conv_sv(
            {"g_sv": g_sv_tt}, a_tt, dw.agh_a, nb_mode, nbmat_tt, self._device, self._d2features
        )
        if self._d2features:
            # a_i is (..., 1, C, S) -> flatten C,S and strip the singleton dim
            a_i_shape = tuple(a_i.shape)
            a_i = ttnn.reshape(a_i, a_i_shape[:-3] + (-1,))
        else:
            # a_i is (..., 1, C) -> strip the singleton dim
            a_i_shape = tuple(a_i.shape)
            a_i = ttnn.reshape(a_i, a_i_shape[:-2] + (a_i_shape[-1],))
        return ttnn.concat([a_i, avf_a], dim=-1)
    def _prepare_in_q(self, charges_tt: Any, g_sv_tt: Any, nb_mode: int, nbmat_tt: Any | None) -> Any:
        """Prepare input for subsequent MLP passes (charge features)."""
        import ttnn

        dw = self._weights.device_weights
        q_i = ttnn_get_i(charges_tt, nb_mode)
        avf_q = ttnn_conv_sv(
            {"g_sv": g_sv_tt}, charges_tt, dw.agh_q, nb_mode, nbmat_tt, self._device, False
        )
        # q_i is (..., 1, C) -> strip the singleton dim
        q_i_shape = tuple(q_i.shape)
        q_i = ttnn.reshape(q_i, q_i_shape[:-2] + (q_i_shape[-1],))
        return ttnn.concat([q_i, avf_q], dim=-1)

    def _run_mlp(self, key: str, x_tt: Any) -> Any:
        """Run an MLP on device and return a device tensor."""
        import ttnn

        weights = self._weights.mlp_weights[key]
        # Elementwise ops may leave tensor in ROW_MAJOR with unpadded shape.
        # to_layout handles padding to tile boundaries.
        x_tt = ttnn.to_layout(x_tt, ttnn.TILE_LAYOUT)
        return run_mlp_ttnn_device(x_tt, weights, self._device)

    def forward(self, batch: TTInputBatch) -> dict[str, Any]:
        """Run the full device-resident energy forward path.

        Parameters
        ----------
        batch : TTInputBatch
            Device-resident input tensors.

        Returns
        -------
        dict
            Dictionary with device tensors.  The caller must ``ttnn.to_torch``
            any values it needs on host.
        """
        import ttnn

        dw = self._weights.device_weights
        device = self._device
        nb_mode = batch.nb_mode

        # --- AEV ---
        from aimnet.ttlang.ops import stage_profile

        with stage_profile("aev.distances"):
            d_ij_tt, r_ij_tt = ttnn_calc_distances(
                batch.coord,
                batch.nbmat,
                batch.shifts,
                nb_mode,
                device,
                pad_value=1.0,
                mask_ij_tt=batch.mask_ij,
            )
        with stage_profile("aev.aev"):
            g_sv_tt = ttnn_calc_aev(
                r_ij_tt,
                d_ij_tt,
                dw.rc_s,
                dw.shifts_s,
                dw.eta_s,
                batch.mask_ij,
                device,
            )

        # --- Initial embedding ---
        with stage_profile("init.embed"):
            if batch.numbers.dtype != ttnn.uint32:
                numbers_u32 = ttnn.typecast(batch.numbers, ttnn.uint32)
            else:
                numbers_u32 = batch.numbers
            a_tt = ttnn.embedding(numbers_u32, dw.afv_weight)

            if self._d2features:
                a_shape = tuple(a_tt.shape)
                a_tt = ttnn.reshape(a_tt, a_shape[:-1] + (self._nfeature, self._nshifts_s))

        # --- Charge preprocessing ---
        if self._num_charge_channels == 2:
            charge_tt = batch.charge
            charges_tt = ttnn.zeros_like(a_tt[..., : self._num_charge_channels])
        else:
            charge_tt = ttnn.unsqueeze(batch.charge, -1)
            charges_tt = ttnn.zeros_like(charge_tt)

        # --- MLP passes ---
        mlp_keys = [k for k in self._weights.mlp_weights if k.startswith("mlps.")]
        n_iter = len(mlp_keys)
        for ipass, key in enumerate(mlp_keys):
            with stage_profile(f"iter{ipass}.prepare"):
                if ipass == 0:
                    _in = self._prepare_in_a(a_tt, g_sv_tt, nb_mode, batch.nbmat)
                else:
                    _in_a = self._prepare_in_a(a_tt, g_sv_tt, nb_mode, batch.nbmat)
                    _in_q = self._prepare_in_q(charges_tt, g_sv_tt, nb_mode, batch.nbmat)
                    _in = ttnn.concat([_in_a, _in_q], dim=-1)

            with stage_profile(f"iter{ipass}.mlp"):
                _out = self._run_mlp(key, _in)

            with stage_profile(f"iter{ipass}.mask"):
                # Mask padding atoms
                if batch.input_padded:
                    if nb_mode == 1 and batch.mask_i is None:
                        n = _out.shape[0]
                        last_dim = _out.shape[-1]
                        first = ttnn.slice(_out, [0, 0], [n - 1, last_dim])
                        last = ttnn.full_like(ttnn.slice(_out, [n - 1, 0], [n, last_dim]), 0.0)
                        _out = ttnn.concat([first, last], dim=0)
                    else:
                        _out = ttnn_mask_i_(_out, batch.mask_i, 0.0)

            with stage_profile(f"iter{ipass}.update_q"):
                if ipass == 0:
                    charges_tt, a_tt, _ = ttnn_update_q(
                        charge_tt, charges_tt, a_tt, _out,
                        self._num_charge_channels, batch.mol_idx, nb_mode, device,
                        delta_q=False, n_molecules=batch.n_molecules,
                    )
                elif ipass < n_iter - 1:
                    charges_tt, a_tt, _ = ttnn_update_q(
                        charge_tt, charges_tt, a_tt, _out,
                        self._num_charge_channels, batch.mol_idx, nb_mode, device,
                        delta_q=True, n_molecules=batch.n_molecules,
                    )
                else:
                    aim_tt = _out

        # --- Postprocess charge ---
        if self._num_charge_channels == 2:
            charges_tt = ttnn.sum(charges_tt, dim=-1)
            charge_tt = ttnn.sum(charge_tt, dim=-1)
        else:
            charges_tt = ttnn.squeeze(charges_tt, -1)
            charge_tt = ttnn.squeeze(charge_tt, -1)
        # --- Output MLP ---
        with stage_profile("output.mlp"):
            aim_mlp = self._run_mlp("outputs.energy_mlp", aim_tt)
            if aim_mlp.shape[-1] == 1:
                aim_mlp = ttnn.squeeze(aim_mlp, -1)

        # --- Final energy ---
        with stage_profile("output.energy"):
            energy_tt = ttnn_output_energy(
                aim_mlp,
                numbers_u32,
                dw.atomic_shifts,
                batch.mask_i,
                batch.mol_idx,
                nb_mode,
                device,
                n_molecules=batch.n_molecules,
            )

        return {"energy": energy_tt}
