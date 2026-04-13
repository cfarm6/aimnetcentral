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
    from torch_sim.models.interface import ModelInterface
    from torch_sim.state import SimState
except ImportError:
    raise ImportError("torch-sim is not installed. Please install it using `pip install torch-sim-atomistic`.")  # noqa: B904


def ts_data_to_aimnet2_data(
    positions: torch.Tensor,
    cell: torch.Tensor | None,
    numbers: torch.Tensor,
    charge: torch.Tensor | None,
    mult: torch.Tensor | None,
    mol_idx: torch.Tensor,
    region_mask: torch.Tensor | None,
    region_charges: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    data = dict[str, torch.Tensor]()
    data["coord"] = positions.contiguous()
    data["numbers"] = numbers.contiguous()
    if charge is not None:
        data["charge"] = charge.contiguous()
    if mult is not None:
        data["mult"] = mult.contiguous()
    if mol_idx is not None:
        data["mol_idx"] = mol_idx.contiguous()
    if region_mask is not None:
        data["region_mask"] = region_mask.contiguous()
    if region_charges is not None:
        data["region_charges"] = region_charges.flatten().contiguous()
    if isinstance(cell, torch.Tensor) and not torch.allclose(cell, torch.zeros_like(cell)):
        data["cell"] = cell.contiguous()
    return data


def state_to_aimnet2_data(state: ts.SimState) -> dict[str, torch.Tensor]:
    positions = state.positions.contiguous()
    cell = state.row_vector_cell
    z = state.atomic_numbers.long().contiguous()
    charge = state.system_extras.get("charge", None)
    if charge is not None:
        charge = charge.contiguous()
    mol_idx = state.system_idx.contiguous()
    mult = getattr(state, "spin", None)
    region_mask = state.system_extras.get("region_mask", None)
    region_charges = state.system_extras.get("region_charges", None)
    return ts_data_to_aimnet2_data(positions, cell, z, charge, mult, mol_idx, region_mask, region_charges)


def state_dict_to_aimnet2_data(state: dict) -> dict[str, torch.Tensor]:
    state_dict = cast(dict[str, torch.Tensor], state)
    positions = state_dict.get("positions")
    assert positions is not None, "positions is required"
    cell = state_dict.get("cell", None)
    numbers = state_dict.get("atomic_numbers", None)
    assert numbers is not None, "atomic_numbers is required"
    charge = state_dict.get("charge", None)
    mult = state_dict.get("mult", None)
    mol_idx = state_dict.get("mol_idx", None)
    assert mol_idx is not None, "mol_idx is required"
    region_mask = state_dict.get("region_mask", None)
    region_charges = state_dict.get("region_charges", None)
    return ts_data_to_aimnet2_data(positions, cell, numbers, charge, mult, mol_idx, region_mask, region_charges)


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
        if device is None:
            device = torch.device(base_calc.device)
        self._device = device
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

    def forward(self, state: SimState | dict, return_charges: bool = True, **kwargs) -> dict[str, torch.Tensor]:
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
        if isinstance(state, SimState):
            if state.device != self._device:
                state = state.to(self._device)
            data = state_to_aimnet2_data(state)
        else:
            # Accept both torch-sim `StateDict` and plain dict-like objects.
            data = state_dict_to_aimnet2_data(state)
        results = self.model(data, forces=compute_forces, stress=compute_stress)

        return results
