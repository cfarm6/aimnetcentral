from dataclasses import dataclass

import numpy as np
import torch

try:
    from ase.calculators.calculator import Calculator, PropertyNotImplementedError, all_changes  # type: ignore
except ImportError:
    raise ImportError("ASE is not installed. Please install ASE to use this module.") from None

from .calculator import AIMNet2Calculator


@dataclass
class ChargeSpinConstraint:
    region_indices: list[int]
    region_value: float


class AIMNet2ASE(Calculator):
    from typing import ClassVar

    implemented_properties: ClassVar[list[str]] = [
        "energy",
        "forces",
        "free_energy",
        "charges",
        "stress",
        "dipole_moment",
    ]

    def __init__(
        self,
        base_calc: AIMNet2Calculator | str = "aimnet2",
        charge=0,
        mult=1,
        charge_constraints: list[ChargeSpinConstraint] | None = None,
        spin_constraints: list[ChargeSpinConstraint] | None = None,
    ):
        super().__init__()
        if isinstance(base_calc, str):
            base_calc = AIMNet2Calculator(base_calc)
        self.base_calc = base_calc
        if self.base_calc.is_nse:
            self.implemented_properties = [*self.__class__.implemented_properties, "spin_charges"]
        self.reset()
        self.charge = charge
        self.mult = mult
        self.charge_constraints = charge_constraints
        self.spin_constraints = spin_constraints
        self.update_tensors()
        # list of implemented species — read from model metadata
        _meta = getattr(base_calc.model, "_metadata", None)
        _species = _meta.get("implemented_species") if _meta is not None else None
        self.implemented_species = np.array(_species, dtype=np.int64) if _species else None

    def reset(self):
        super().reset()
        self._t_numbers = None
        self._t_charge = None
        self._t_mult = None

    def set_atoms(self, atoms):
        if self.implemented_species is not None and not np.isin(atoms.numbers, self.implemented_species).all():
            raise ValueError("Some species are not implemented in the AIMNet2Calculator")
        self.reset()
        self.atoms = atoms

    @staticmethod
    def _info_entry_equal(a: object, b: object) -> bool:
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        if isinstance(a, np.ndarray | np.generic) or isinstance(b, np.ndarray | np.generic):
            return np.array_equal(np.asarray(a), np.asarray(b))
        return bool(a == b)

    def check_state(self, atoms, tol=1e-15):
        state = super().check_state(atoms, tol=tol)
        if (not state) and getattr(self, "atoms", None) is not None:
            # Check for specific keys in info that affect the calculation
            old_info = getattr(self.atoms, "info", {})
            new_info = getattr(atoms, "info", {})

            # Check charge
            if (
                old_info.get("charge") != new_info.get("charge")
                or not self._info_entry_equal(old_info.get("region_mask"), new_info.get("region_mask"))
                or not self._info_entry_equal(
                    old_info.get("region_charges", old_info.get("region_charge")),
                    new_info.get("region_charges", new_info.get("region_charge")),
                )
            ):
                state.append("info")

            # Check spin/multiplicity (NSE models only)
            elif self.base_calc.is_nse:
                old_spin = old_info.get("spin", old_info.get("mult"))
                new_spin = new_info.get("spin", new_info.get("mult"))
                if old_spin != new_spin:
                    state.append("info")
        return state

    def set_charge(self, charge):
        self.charge = charge
        self._t_charge = None
        self.update_tensors()

    def set_mult(self, mult):
        self.mult = mult
        self._t_mult = None
        self.update_tensors()

    def set_charge_constraints(self, charge_constraints):
        self.charge_constraints = charge_constraints
        self._t_charge_constraints = None
        self.update_tensors()

    def set_spin_constraints(self, spin_constraints):
        self.spin_constraints = spin_constraints
        self._t_spin_constraints = None
        self.update_tensors()

    def _update_charge_spin_from_info(self):
        atoms = getattr(self, "atoms", None)
        if atoms is None:
            return
        info = getattr(atoms, "info", {})

        # Order of precedence for charge:
        # 1. atoms.info['charge']
        # 2. calculator.charge (passed to constructor or set_charge)
        charge = info.get("charge")
        if charge is not None and charge != self.charge:
            self.charge = charge
            self._t_charge = None

        if self.base_calc.is_nse:
            # Support both "mult" (AIMNet2 style) and "spin" (MACE style)
            # Both represent multiplicity (2S+1)
            mult = info.get("mult", info.get("spin"))
            if mult is not None and mult != self.mult:
                self.mult = mult
                self._t_mult = None

    def _update_charge_spin_constraints_from_info(self):
        atoms = getattr(self, "atoms", None)
        if atoms is None:
            return
        info = getattr(atoms, "info", {})

        # Order of precedence for charge:
        # 1. atoms.info['charge']
        # 2. calculator.charge (passed to constructor or set_charge)
        charge_constraints = info.get("charge_constraints")
        if charge_constraints is not None and charge_constraints != self.charge_constraints:
            self.charge_constraints = charge_constraints
            self._t_charge = None

        if self.base_calc.is_nse:
            # Support both "mult" (AIMNet2 style) and "spin" (MACE style)
            # Both represent multiplicity (2S+1)
            mult = info.get("mult", info.get("spin"))
            if mult is not None and mult != self.mult:
                self.mult = mult
                self._t_mult = None

    def update_tensors(self):
        if self._t_numbers is None and getattr(self, "atoms", None):
            self._t_numbers = torch.tensor(self.atoms.numbers, dtype=torch.int64, device=self.base_calc.device)
        if self._t_charge is None:
            self._t_charge = torch.tensor(self.charge, dtype=torch.float32, device=self.base_calc.device)
        if self._t_mult is None:
            self._t_mult = torch.tensor(self.mult, dtype=torch.float32, device=self.base_calc.device)

    def _region_constraint_tensors_from_info(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Build `region_mask` / `region_charges` from `atoms.info` if present.

        Supported keys: ``region_mask`` (per-atom region indices) and
        ``region_charges`` or ``region_charge`` (target charge per region, length
        ``max(mask)+1``).
        """
        atoms = getattr(self, "atoms", None)
        if atoms is None:
            return None
        info = getattr(atoms, "info", {})
        if "region_mask" not in info:
            return None
        targets = info.get("region_charges", info.get("region_charge"))
        if targets is None:
            return None
        rm = np.asarray(info["region_mask"], dtype=np.int64)
        if rm.shape != (len(atoms),):
            raise ValueError(f"atoms.info['region_mask'] must have shape ({len(atoms)},), got {rm.shape}")
        rt = np.asarray(targets, dtype=np.float64).reshape(-1)
        device = self.base_calc.device
        mask_t = torch.as_tensor(rm, device=device, dtype=torch.int64).unsqueeze(-1)
        charges_t = torch.as_tensor(rt, dtype=torch.float32, device=device)
        return mask_t, charges_t

    def get_dipole_moment(self, atoms):
        charges = self.get_charges()[:, np.newaxis]
        positions = atoms.get_positions()
        return np.sum(charges * positions, axis=0)

    def get_spin_charges(self, atoms=None):
        if "spin_charges" not in self.results:
            raise PropertyNotImplementedError("spin_charges is not available. Use an NSE model (e.g. 'aimnet2nse').")
        return self.results["spin_charges"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if properties is None:
            properties = ["energy"]
        super().calculate(atoms, properties, system_changes)
        self._update_charge_spin_from_info()
        self._update_charge_spin_constraints_from_info()
        self.update_tensors()

        cell = self.atoms.cell.array if self.atoms.cell is not None and self.atoms.pbc.any() else None

        _in = {
            "coord": torch.as_tensor(
                self.atoms.positions,  # [N_Atoms, 3]
                dtype=torch.float32,
                device=self.base_calc.device,
            ),
            "numbers": self._t_numbers,  # [N_atoms]
            "charge": self._t_charge,  # []
            "mult": self._t_mult,  # []
        }
        _unsqueezed = False
        region_from_info = self._region_constraint_tensors_from_info()
        if region_from_info is not None:
            _in["region_mask"], _in["region_charges"] = region_from_info
        elif self.charge_constraints is not None:
            mask = torch.zeros(len(self.atoms), dtype=torch.int64, device=self.base_calc.device)
            for constraint_idx, constraint in enumerate(self.charge_constraints):
                mask[constraint.region_indices] = constraint_idx
            charges = [constraint.region_value for constraint in self.charge_constraints]
            _in["region_mask"] = mask.unsqueeze(-1)
            _in["region_charges"] = torch.tensor(charges, dtype=torch.float32, device=self.base_calc.device)

        if cell is not None:
            _in["cell"] = cell
        else:
            for k, v in _in.items():
                _in[k] = v.unsqueeze(0)
            _unsqueezed = True
        results = self.base_calc(
            _in,
            forces="forces" in properties,
            stress="stress" in properties,
        )
        for k, v in results.items():
            if _unsqueezed:
                v = v.squeeze(0)
            results[k] = v.detach().cpu().numpy()  # type: ignore

        self.results["energy"] = results["energy"].item()
        self.results["charges"] = results["charges"]
        self.results["dipole_moment"] = self.get_dipole_moment(self.atoms)

        if "forces" in properties:
            self.results["forces"] = results["forces"]
        if "stress" in properties:
            self.results["stress"] = results["stress"]
        if "spin_charges" in results:
            self.results["spin_charges"] = results["spin_charges"]
