"""torch-sim ModelInterface wrapper for AimNet2.

This module provides a TorchSim wrapper of the AimNet2 model for computing
energies, forces, and stresses for atomistic systems.
"""

from collections.abc import Callable
from pathlib import Path

import torch

from .calculator import AIMNet2Calculator

try:
    import torch_sim as ts
    from torch_sim.models.interface import ModelInterface, SimState, StateDict
except ImportError:
    raise ImportError("torch-sim is not installed. Please install it using `pip install torch-sim-atomistic`.")  # noqa: B904


def state_to_aimnet2_data(state: ts.SimState) -> dict[str, torch.Tensor]:
    positions = state.positions
    cell = state.row_vector_cell
    z = state.atomic_numbers.long()
    charge = state.charge
    spin = state.spin
    mol_idx = state.system_idx
    data = {
        "coord": positions,
        "numbers": z,
        "charge": charge,
        "spin": spin,
        "mol_idx": mol_idx,
    }
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
    data: dict[str, torch.Tensor] = {}
    if "positions" in state:
        data["coord"] = state["positions"]
    if "cell" in state:
        data["cell"] = state["cell"]
    if "atomic_numbers" in state:
        data["numbers"] = state["atomic_numbers"]
    if "charge" in state:
        data["charge"] = state["charge"]
    if "mol_idx" in state:
        data["mol_idx"] = state["system_idx"]
    cell = state["cell"]
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
        self._device = base_calc.device
        self._dtype = dtype or torch.float32
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

    def forward(self, state: SimState | StateDict, **kwargs) -> dict[str, torch.Tensor]:
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
        if isinstance(state, SimState):
            if state.device != self._device:
                state = state.to(self._device)
            data = state_to_aimnet2_data(state)
            # Ensure system_idx has integer dtype
            if state.system_idx.dtype != torch.int64:
                data["mol_idx"] = data["mol_idx"].to(torch.int64)
        elif isinstance(state, StateDict):
            data = state_dict_to_aimnet2_data(state)

        results = self.model(data, forces=True, stress=self._compute_stress)

        return results
