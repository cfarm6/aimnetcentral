"""torch-sim ModelInterface wrapper for AimNet2.

This module provides a TorchSim wrapper of the AimNet2 model for computing
energies, forces, and stresses for atomistic systems.

Region charge constraints can be passed on the state or state dict as
``region_mask`` and ``region_charges`` (or the alias ``region_charge``). ASE
users can set the same data on ``atoms.info`` when using
:class:`aimnet.calculators.aimnet2ase.AIMNet2ASE`.
"""

from collections.abc import Callable
from pathlib import Path
from typing import cast

import torch

from .calculator import AIMNet2Calculator

try:
    import torch_sim as ts
    from torch_sim.models.interface import ModelInterface, SimState, StateDict
except ImportError:
    raise ImportError("torch-sim is not installed. Please install it using `pip install torch-sim-atomistic`.")  # noqa: B904


def state_to_aimnet2_data(state: ts.SimState) -> dict[str, torch.Tensor]:
    positions = state.positions.contiguous()
    cell = state.row_vector_cell
    z = state.atomic_numbers.long().contiguous()
    charge = state.charge.contiguous()
    mol_idx = state.system_idx.contiguous()

    # TorchSim "spin" corresponds to AIMNet2 NSE "mult" (multiplicity).
    # For closed-shell models, this value is ignored by AIMNet2.
    mult = getattr(state, "spin", None)
    data = {
        "coord": positions,
        "numbers": z,
        "charge": charge,
        "mol_idx": mol_idx,
    }
    if isinstance(mult, torch.Tensor):
        data["mult"] = mult.contiguous()
    # Optional charge-region constraints for NSE / region-constrained charge equilibration.
    # These are consumed by AIMNet2Calculator (ops.nse via `region_mask` / `region_charges`).
    region_mask = getattr(state, "region_mask", None)
    if isinstance(region_mask, torch.Tensor):
        # Base calculator expects `region_mask` without an extra feature dim.
        if region_mask.ndim > 1 and region_mask.shape[-1] == 1:
            region_mask = region_mask.squeeze(-1)
        data["region_mask"] = region_mask
    region_charges = getattr(state, "region_charges", None)
    if not isinstance(region_charges, torch.Tensor):
        region_charges = getattr(state, "region_charge", None)
    if isinstance(region_charges, torch.Tensor):
        # For flat coord inputs, AIMNet2Calculator will do an extra
        # `unsqueeze(-1)` on `region_charges`, so keep it 1D to avoid
        # accidentally producing shape (R, 1, 1).
        if region_charges.ndim > 1 and region_charges.shape[-1] == 1:
            region_charges = region_charges.squeeze(-1)
        data["region_charges"] = region_charges
    # Handle periodic cells:
    # - If cell is all zeros, treat as non-periodic and omit "cell"
    # - If all batched cells are identical, use a single (3, 3) cell
    # - Otherwise, keep the batched (B, 3, 3) cell so each system can have its own box
    # Ensure we are working with a tensor (torch-sim may return None for non-periodic systems)
    # Keep the cell tensor rank consistent with the incoming state to avoid
    # downstream shape differences (e.g. stress rank changes) across chunks.
    if isinstance(cell, torch.Tensor) and not torch.allclose(cell, torch.zeros_like(cell)):
        data["cell"] = cell.contiguous()
    return data


def state_dict_to_aimnet2_data(state: StateDict) -> dict[str, torch.Tensor]:
    # TorchSim's `StateDict` typing is narrow; for wrapper logic we treat it as a
    # generic string->tensor mapping (runtime keys may include `charge`, `spin`, ...).
    state_dict = cast(dict[str, torch.Tensor], state)
    data: dict[str, torch.Tensor] = {}

    if "positions" in state_dict:
        data["coord"] = state_dict["positions"].contiguous()
    if "cell" in state_dict:
        data["cell"] = state_dict["cell"].contiguous()
    if "atomic_numbers" in state_dict:
        data["numbers"] = state_dict["atomic_numbers"].contiguous()
    if "charge" in state_dict:
        data["charge"] = state_dict["charge"].contiguous()

    # TorchSim uses `spin`; AIMNet2 expects NSE `mult` (multiplicity).
    mult = state_dict.get("mult")
    if mult is None:
        mult = state_dict.get("spin")
    if isinstance(mult, torch.Tensor):
        data["mult"] = mult.contiguous()

    # torch-sim typically uses `system_idx` but we also accept `mol_idx` if present.
    if "mol_idx" in state_dict:
        data["mol_idx"] = state_dict["mol_idx"].contiguous()
    elif "system_idx" in state_dict:
        data["mol_idx"] = state_dict["system_idx"].contiguous()

    # Optional region-wise charge constraints
    if "region_mask" in state_dict:
        region_mask = state_dict["region_mask"]
        if region_mask.ndim > 1 and region_mask.shape[-1] == 1:
            region_mask = region_mask.squeeze(-1)
        data["region_mask"] = region_mask
    _rc = state_dict.get("region_charges")
    if _rc is None:
        _rc = state_dict.get("region_charge")
    if isinstance(_rc, torch.Tensor):
        region_charges = _rc
        if region_charges.ndim > 1 and region_charges.shape[-1] == 1:
            region_charges = region_charges.squeeze(-1)
        data["region_charges"] = region_charges

    cell = state_dict.get("cell", None)
    if isinstance(cell, torch.Tensor) and not torch.allclose(cell, torch.zeros_like(cell)):
        data["cell"] = cell.contiguous()
    return data


class AIMNet2TorchSim(ModelInterface):
    """Computes energies, forces, and stresses for atomistic systems using the AIMNet2 model.

    Attributes
    ----------
        model : nn.Module
            The loaded AIMNet2 model.
        _device : str
            Device the model is running on ("cuda" or "cpu").
        _dtype: torch.dtype
        _compute_stress: bool
        implemented_properties: list[str]

    """

    def __init__(
        self,
        base_calc: AIMNet2Calculator,
        neighbor_list_fn: Callable | None = None,
        *,
        model_cache_dir: str | Path | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        compute_stress: bool = False,
    ):
        """Initial the AIMNet2TorchSim model.

        Args:
            base_calc: AIMNet2Calculator
                The AIMNet2 calculator to use.
        """
        super().__init__()
        self.model = base_calc
        self._device = torch.device(base_calc.device)
        try:
            params_fn = getattr(base_calc.model, "parameters", None)
            model_dtype = next(params_fn()).dtype if callable(params_fn) else torch.float32
        except StopIteration:
            model_dtype = torch.float32
        self._dtype = dtype or model_dtype
        self._compute_stress = compute_stress
        self._compute_forces = True
        self._memory_scales_with = "n_atoms_x_density"
        if neighbor_list_fn is not None:
            raise NotImplementedError("Custom neighbor list is not supported for the AIMNet2 Model.")
        self.predictor = base_calc.eval
        self.implemented_properties = ["energy", "forces", "charges"]
        if base_calc.is_nse:
            self.implemented_properties.append("spin_charges")
        if self._compute_stress:
            self.implemented_properties.append("stress")

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    def forward(self, state: SimState | StateDict, return_charges: bool = True, **kwargs) -> dict[str, torch.Tensor]:
        """Compute energies, forces, and other properties.

        Args:
            state (SimState): State object containing positions, cells, atomic numbers,
                and other system information.
            **_kwargs: Unused; accepted for interface compatibility.

        Returns:
            dict: Dictionary of model predictions, which may include:
                - energy (torch.Tensor): Energy with shape [batch_size]
                - forces (torch.Tensor): Forces with shape [n_atoms, 3]
                - stress (torch.Tensor): Stress tensor with shape [batch_size, 3, 3]
        """

        def _maybe_bool(v: object, default: bool) -> bool:
            if v is None:
                return default
            if isinstance(v, torch.Tensor):
                # TorchSim may pass scalar tensors; keep this robust.
                return bool(v.item()) if v.ndim == 0 else bool(v.any().item())
            return bool(v)

        compute_forces = _maybe_bool(kwargs.get("forces", self._compute_forces), self._compute_forces)
        compute_stress = _maybe_bool(kwargs.get("stress", self._compute_stress), self._compute_stress)
        print(state.region_charges)
        if isinstance(state, SimState):
            if state.device != self._device:
                state = state.to(self._device)
            data = state_to_aimnet2_data(state)
        else:
            # Accept both torch-sim `StateDict` and plain dict-like objects.
            data = state_dict_to_aimnet2_data(state)
        results = self.model(data, forces=compute_forces, stress=compute_stress)

        return results
