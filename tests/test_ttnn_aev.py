from __future__ import annotations

import pytest
import torch

from aimnet import nbops, ops
from aimnet.calculators.calculator import AIMNet2Calculator
from aimnet.calculators.model_registry import get_model_path
from aimnet.models.base import load_model
from aimnet.modules.aev import AEVSV
from aimnet.ttlang.ttnn_aev import (
    ttnn_calc_aev,
    ttnn_calc_distances,
    ttnn_cosine_cutoff,
    ttnn_exp_expand,
)

pytestmark = pytest.mark.tt


@pytest.fixture
def device():
    pytest.importorskip("ttnn")
    import ttnn

    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


def _to_tt(tensor, device):
    import ttnn

    return ttnn.from_torch(
        tensor.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _to_torch(tt_tensor):
    import ttnn

    return ttnn.to_torch(tt_tensor)


@pytest.fixture
def aev_module():
    model, _ = load_model(get_model_path("aimnet2"))
    return model.aev


@pytest.fixture
def water_mode0():
    coord = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [-0.24, 0.93, 0.0],
        ],
        dtype=torch.float32,
    )
    numbers = torch.tensor([8, 1, 1], dtype=torch.long)
    data = {
        "coord": coord.unsqueeze(0),
        "numbers": numbers.unsqueeze(0),
        "charge": torch.tensor([0.0], dtype=torch.float32),
    }
    nbops.set_nb_mode(data)
    nbops.calc_masks(data)
    d_ij, r_ij = ops.calc_distances(data)
    return data, d_ij, r_ij


@pytest.fixture
def water_mode1():
    coord = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [-0.24, 0.93, 0.0],
        ],
        dtype=torch.float32,
    )
    numbers = torch.tensor([8, 1, 1], dtype=torch.long)
    data = {
        "coord": coord,
        "numbers": numbers,
        "charge": torch.tensor(0.0, dtype=torch.float32),
    }
    calc = AIMNet2Calculator("aimnet2")
    data = calc.prepare_input(data)
    nbops.set_nb_mode(data)
    nbops.calc_masks(data)
    d_ij, r_ij = ops.calc_distances(data)
    return data, d_ij, r_ij


class TestCalcDistances:
    def test_mode0(self, device, water_mode0):
        data, d_ij_cpu, r_ij_cpu = water_mode0
        coord_tt = _to_tt(data["coord"], device)
        d_ij_tt, r_ij_tt = ttnn_calc_distances(
            coord_tt,
            nbmat_tt=None,
            shifts_tt=None,
            nb_mode=0,
            device=device,
            pad_value=1.0,
        )
        d_ij_out = _to_torch(d_ij_tt).to(torch.float32)
        r_ij_out = _to_torch(r_ij_tt).to(torch.float32)
        assert torch.allclose(d_ij_out, d_ij_cpu, atol=1e-2)
        assert torch.allclose(r_ij_out, r_ij_cpu, atol=1e-2)

    def test_mode1(self, device, water_mode1):
        data, d_ij_cpu, r_ij_cpu = water_mode1
        coord_tt = _to_tt(data["coord"], device)
        nbmat_tt = _to_tt(data["nbmat"].to(torch.int32), device)
        mask_ij = (data["nbmat"] == data["nbmat"].shape[0] - 1).to(torch.bfloat16)
        mask_ij_tt = _to_tt(mask_ij, device)
        d_ij_tt, r_ij_tt = ttnn_calc_distances(
            coord_tt,
            nbmat_tt=nbmat_tt,
            shifts_tt=None,
            nb_mode=1,
            device=device,
            pad_value=1.0,
            mask_ij_tt=mask_ij_tt,
        )
        d_ij_out = _to_torch(d_ij_tt).to(torch.float32)
        r_ij_out = _to_torch(r_ij_tt).to(torch.float32)
        assert torch.allclose(d_ij_out, d_ij_cpu, atol=1e-2)
        assert torch.allclose(r_ij_out, r_ij_cpu, atol=1e-2)


class TestCosineCutoff:
    def test_matches_cpu(self, device, aev_module):
        d = torch.tensor([[0.5, 1.0, 5.0, 6.0]], dtype=torch.float32)
        d_tt = _to_tt(d, device)
        rc = aev_module.rc_s.item()
        rc_tt = _to_tt(torch.tensor(rc, dtype=torch.float32), device)
        fc_tt = ttnn_cosine_cutoff(d_tt, rc_tt)
        fc_cpu = ops.cosine_cutoff(d, rc)
        fc_out = _to_torch(fc_tt).to(torch.float32)
        assert torch.allclose(fc_out, fc_cpu, atol=1e-2)


class TestExpExpand:
    def test_matches_cpu(self, device, aev_module):
        d = torch.tensor([[0.5, 1.0, 2.0]], dtype=torch.float32)
        d_tt = _to_tt(d, device)
        shifts = aev_module.shifts_s.data
        eta = aev_module.eta_s.item()
        shifts_tt = _to_tt(shifts, device)
        eta_tt = _to_tt(torch.tensor(eta, dtype=torch.float32), device)
        gs_tt = ttnn_exp_expand(d_tt, shifts_tt, eta_tt)
        gs_cpu = ops.exp_expand(d, shifts, eta)
        gs_out = _to_torch(gs_tt).to(torch.float32)
        assert torch.allclose(gs_out, gs_cpu, atol=5e-2)


class TestCalcAEV:
    def test_mode0(self, device, water_mode0, aev_module):
        data, d_ij_cpu, r_ij_cpu = water_mode0
        coord_tt = _to_tt(data["coord"], device)
        d_ij_tt, r_ij_tt = ttnn_calc_distances(
            coord_tt,
            nbmat_tt=None,
            shifts_tt=None,
            nb_mode=0,
            device=device,
            pad_value=1.0,
        )
        rc_s = aev_module.rc_s.item()
        shifts_s = aev_module.shifts_s.data
        eta_s = aev_module.eta_s.item()
        rc_tt = _to_tt(torch.tensor(rc_s, dtype=torch.float32), device)
        shifts_tt = _to_tt(shifts_s, device)
        eta_tt = _to_tt(torch.tensor(eta_s, dtype=torch.float32), device)
        mask_ij_tt = _to_tt(data["mask_ij"].to(torch.bfloat16), device)
        g_sv_tt = ttnn_calc_aev(
            r_ij_tt,
            d_ij_tt,
            rc_tt,
            shifts_tt,
            eta_tt,
            mask_ij_tt,
            device,
        )
        g_sv_cpu = aev_module._calc_aev(r_ij_cpu, d_ij_cpu, data)
        g_sv_out = _to_torch(g_sv_tt).to(torch.float32)
        assert torch.allclose(g_sv_out, g_sv_cpu, atol=1e-2)

    def test_mode1(self, device, water_mode1, aev_module):
        data, d_ij_cpu, r_ij_cpu = water_mode1
        coord_tt = _to_tt(data["coord"], device)
        nbmat_tt = _to_tt(data["nbmat"].to(torch.int32), device)
        mask_ij = (data["nbmat"] == data["nbmat"].shape[0] - 1).to(torch.bfloat16)
        mask_ij_tt = _to_tt(mask_ij, device)
        d_ij_tt, r_ij_tt = ttnn_calc_distances(
            coord_tt,
            nbmat_tt=nbmat_tt,
            shifts_tt=None,
            nb_mode=1,
            device=device,
            pad_value=1.0,
            mask_ij_tt=mask_ij_tt,
        )
        rc_s = aev_module.rc_s.item()
        shifts_s = aev_module.shifts_s.data
        eta_s = aev_module.eta_s.item()
        rc_tt = _to_tt(torch.tensor(rc_s, dtype=torch.float32), device)
        shifts_tt = _to_tt(shifts_s, device)
        eta_tt = _to_tt(torch.tensor(eta_s, dtype=torch.float32), device)
        g_sv_tt = ttnn_calc_aev(
            r_ij_tt,
            d_ij_tt,
            rc_tt,
            shifts_tt,
            eta_tt,
            mask_ij_tt,
            device,
        )
        g_sv_cpu = aev_module._calc_aev(r_ij_cpu, d_ij_cpu, data)
        g_sv_out = _to_torch(g_sv_tt).to(torch.float32)
        assert torch.allclose(g_sv_out, g_sv_cpu, atol=1e-2)
