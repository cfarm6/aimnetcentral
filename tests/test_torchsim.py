"""Tests for TorchSim Calculator Interface."""

import warnings

import numpy as np
import pytest
from conftest import CAFFEINE_FILE, CIF_SPIRO

# All tests in this module require TorchSim
pytestmark = [pytest.mark.torch_sim, pytest.mark.ase]

MODELS = ("aimnet2", "aimnet2_2025")
NSE_MODEL = "aimnet2nse"


class TestBasicCalculator:
    """Basic TorchSim calculator tests."""

    def test_energy_calculation(self):
        """Test that energy calculation works."""
        pytest.importorskip("torch_sim", reason="torch_sim not installed")
        pytest.importorskip("ase", reason="ASE not installed")
        import torch_sim as ts
        from ase.io import read

        from aimnet.calculators import AIMNet2Calculator, AIMNet2TorchSim

        for model in MODELS:
            atoms = read(CAFFEINE_FILE)
            base_calc = AIMNet2Calculator(model)
            state = ts.static(system=atoms, model=AIMNet2TorchSim(base_calc))
            e = state[0]["potential_energy"].item()
            assert isinstance(e, float)
            assert np.isfinite(e)


class TestForces:
    """Tests for force calculations."""

    def test_forces_shape(self):
        """Test that forces have correct shape."""
        pytest.importorskip("ase", reason="ASE not installed")
        pytest.importorskip("torch_sim", reason="torch_sim not installed")
        import torch_sim as ts
        from ase.io import read

        from aimnet.calculators import AIMNet2Calculator, AIMNet2TorchSim

        atoms = read(CAFFEINE_FILE)
        base_calc = AIMNet2Calculator("aimnet2")
        state = ts.static(system=atoms, model=AIMNet2TorchSim(base_calc))

        f = state[0]["forces"].cpu().numpy()
        assert f.shape == (len(atoms), 3)
        assert np.isfinite(f).all()

    def test_forces_sum_nearly_zero(self):
        """Test that total force is nearly zero (Newton's third law)."""
        pytest.importorskip("ase", reason="ASE not installed")
        pytest.importorskip("torch_sim", reason="torch_sim not installed")
        import torch_sim as ts
        from ase.io import read

        from aimnet.calculators import AIMNet2Calculator, AIMNet2TorchSim

        atoms = read(CAFFEINE_FILE)
        base_calc = AIMNet2Calculator("aimnet2")
        state = ts.static(system=atoms, model=AIMNet2TorchSim(base_calc))

        f = state[0]["forces"].cpu().numpy()
        total_force = np.sum(f, axis=0)
        # Total force should be nearly zero
        assert np.allclose(total_force, 0, atol=1e-4)


class TestPBC:
    """Tests for periodic boundary conditions."""

    def test_pbc_energy(self):
        """Test energy calculation for periodic system."""
        pytest.importorskip("ase", reason="ASE not installed")
        pytest.importorskip("torch_sim", reason="torch_sim not installed")
        import torch_sim as ts
        from ase.io import read

        from aimnet.calculators import AIMNet2Calculator, AIMNet2TorchSim

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="crystal system.*monoclinic", category=UserWarning)
            atoms = read(CIF_SPIRO)

        base_calc = AIMNet2Calculator("aimnet2")
        state = ts.static(system=atoms, model=AIMNet2TorchSim(base_calc))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Switching to DSF Coulomb", category=UserWarning)
            warnings.filterwarnings("ignore", message="Model has embedded Coulomb module", category=UserWarning)
            e = state[0]["potential_energy"].item()
        assert isinstance(e, float)
        assert np.isfinite(e)

    def test_pbc_forces(self):
        """Test force calculation for periodic system."""
        pytest.importorskip("ase", reason="ASE not installed")
        pytest.importorskip("torch_sim", reason="torch_sim not installed")
        import torch_sim as ts
        from ase.io import read

        from aimnet.calculators import AIMNet2Calculator, AIMNet2TorchSim

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="crystal system.*monoclinic", category=UserWarning)
            atoms = read(CIF_SPIRO)
        base_calc = AIMNet2Calculator("aimnet2")
        state = ts.static(system=atoms, model=AIMNet2TorchSim(base_calc))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Switching to DSF Coulomb", category=UserWarning)
            warnings.filterwarnings("ignore", message="Model has embedded Coulomb module", category=UserWarning)
            f = state[0]["forces"]
        assert f.shape == (len(atoms), 3)
        assert np.isfinite(f.cpu().numpy()).all()

    def test_pbc_stress_tensor(self):
        """Test stress tensor calculation for periodic system."""
        pytest.importorskip("ase", reason="ASE not installed")
        pytest.importorskip("torch_sim", reason="torch_sim not installed")
        import torch_sim as ts
        from ase.io import read

        from aimnet.calculators import AIMNet2Calculator, AIMNet2TorchSim

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="crystal system.*monoclinic", category=UserWarning)
            atoms = read(CIF_SPIRO)
        base_calc = AIMNet2Calculator("aimnet2")
        state = ts.static(system=atoms, model=AIMNet2TorchSim(base_calc, compute_stress=True))

        # Get stress tensor (Voigt notation: xx, yy, zz, yz, xz, xy)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Switching to DSF Coulomb", category=UserWarning)
            warnings.filterwarnings("ignore", message="Model has embedded Coulomb module", category=UserWarning)
            stress = state[0]["stress"].cpu().numpy()
        assert stress.shape == (1, 3, 3)
        assert np.isfinite(stress).all()

    def test_pbc_stress_volume_normalized(self):
        """Test that stress is volume normalized (units are pressure)."""
        pytest.importorskip("ase", reason="ASE not installed")
        pytest.importorskip("torch_sim", reason="torch_sim not installed")
        import torch_sim as ts
        from ase.io import read

        from aimnet.calculators import AIMNet2Calculator, AIMNet2TorchSim

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="crystal system.*monoclinic", category=UserWarning)
            atoms = read(CIF_SPIRO)
        base_calc = AIMNet2Calculator("aimnet2")
        state = ts.static(system=atoms, model=AIMNet2TorchSim(base_calc, compute_stress=True))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Switching to DSF Coulomb", category=UserWarning)
            warnings.filterwarnings("ignore", message="Model has embedded Coulomb module", category=UserWarning)
            stress = state[0]["stress"]
        # Stress values should be reasonable (not extremely large)
        # Typical stress values are in GPa range (1e-4 to 1e-1 eV/Å³)
        assert np.abs(stress.cpu().numpy()).max() < 10.0  # Sanity check


class TestOptimization:
    """Tests for geometry optimization."""

    def test_energy_decreases_on_optimization(self):
        """Test that energy decreases during optimization."""
        pytest.importorskip("ase", reason="ASE not installed")
        pytest.importorskip("torch_sim", reason="torch_sim not installed")
        import torch_sim as ts
        from ase.io import read

        from aimnet.calculators import AIMNet2Calculator, AIMNet2TorchSim

        atoms = read(CAFFEINE_FILE)
        # Add small random displacement
        atoms.positions += np.random.randn(*atoms.positions.shape) * 0.1
        base_calc = AIMNet2Calculator("aimnet2")
        state = ts.static(system=atoms, model=AIMNet2TorchSim(base_calc))
        e_initial = state[0]["potential_energy"].item()

        # Run a few optimization steps
        final_state = ts.optimize(
            system=atoms, model=AIMNet2TorchSim(base_calc), optimizer=ts.Optimizer.fire, pbar=True, max_steps=5
        )

        e_final = final_state.energy.detach().cpu().numpy()
        # Energy should decrease (or stay same if already at minimum)
        assert e_final <= e_initial + 1e-3


class TestMultipleModels:
    """Tests for multiple model support."""

    @pytest.mark.parametrize("model", MODELS)
    def test_model_gives_finite_energy(self, model):
        """Test each model gives finite energy."""
        pytest.importorskip("ase", reason="ASE not installed")
        pytest.importorskip("torch_sim", reason="torch_sim not installed")
        import torch_sim as ts
        from ase.io import read

        from aimnet.calculators import AIMNet2Calculator, AIMNet2TorchSim

        atoms = read(CAFFEINE_FILE)
        base_calc = AIMNet2Calculator(model)
        state = ts.static(system=atoms, model=AIMNet2TorchSim(base_calc))

        e = state[0]["potential_energy"].detach().cpu().numpy()
        assert np.isfinite(e)

    @pytest.mark.parametrize("model", MODELS)
    def test_model_gives_finite_forces(self, model):
        """Test each model gives finite forces."""
        pytest.importorskip("ase", reason="ASE not installed")
        pytest.importorskip("torch_sim", reason="torch_sim not installed")
        import torch_sim as ts
        from ase.io import read

        from aimnet.calculators import AIMNet2Calculator, AIMNet2TorchSim

        atoms = read(CAFFEINE_FILE)
        base_calc = AIMNet2Calculator(model)
        state = ts.static(system=atoms, model=AIMNet2TorchSim(base_calc))

        f = state[0]["forces"].cpu().numpy()
        assert np.isfinite(f).all()
