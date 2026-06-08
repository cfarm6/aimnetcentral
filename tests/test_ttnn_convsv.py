from __future__ import annotations

import os

import pytest
import torch

from aimnet.modules.aev import ConvSV
from aimnet.ttlang.ttnn_convsv import ttnn_conv_sv

pytestmark = pytest.mark.tt


@pytest.fixture
def device():
    pytest.importorskip("ttnn")
    import ttnn

    if not os.path.exists("/dev/tenstorrent/0"):
        pytest.skip("No Tenstorrent device available")
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


def _to_tt(tensor, device):
    import ttnn

    return ttnn.from_torch(
        tensor.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )


def _to_torch(tt_tensor):
    import ttnn

    return ttnn.to_torch(tt_tensor)


def _run_pytorch_convsv(data, a, agh, nchannel, nshifts, ncomb_v, d2features):
    """Run PyTorch ConvSV reference and return output."""
    conv = ConvSV(
        nshifts_s=nshifts,
        nchannel=nchannel,
        d2features=d2features,
        nshifts_v=nshifts,
        ncomb_v=ncomb_v,
    )
    conv.agh.data = agh
    out = conv(data, a)
    return out


class TestConvSVMode1:
    def test_mode1_non_d2features(self, device):
        n_atoms, max_neighbors = 4, 4
        nchannel, nshifts, ncomb_v = 3, 5, 6
        a = torch.randn(n_atoms, nchannel) * 0.1
        g_sv = torch.randn(n_atoms, max_neighbors, nshifts, 4) * 0.1
        nbmat = torch.randint(0, n_atoms, (n_atoms, max_neighbors))
        agh = torch.randn(nchannel, nshifts, ncomb_v) * 0.1

        data = {
            "g_sv": g_sv,
            "nbmat": nbmat,
            "_nb_mode": torch.tensor(1),
        }
        expected = _run_pytorch_convsv(data, a, agh, nchannel, nshifts, ncomb_v, d2features=False)

        a_tt = _to_tt(a, device)
        g_sv_tt = _to_tt(g_sv, device)
        nbmat_tt = _to_tt(nbmat.to(torch.int32), device)
        agh_tt = _to_tt(agh, device)

        data_tt = {"g_sv": g_sv_tt}
        out_tt = ttnn_conv_sv(data_tt, a_tt, agh_tt, 1, nbmat_tt, device, d2features=False)
        out = _to_torch(out_tt)

        assert out.shape == expected.shape
        assert torch.allclose(out, expected.to(torch.bfloat16), atol=0.1)

    def test_mode1_d2features(self, device):
        n_atoms, max_neighbors = 4, 4
        nchannel, nshifts, ncomb_v = 3, 5, 6
        a = torch.randn(n_atoms, nchannel, nshifts) * 0.1
        g_sv = torch.randn(n_atoms, max_neighbors, nshifts, 4) * 0.1
        nbmat = torch.randint(0, n_atoms, (n_atoms, max_neighbors))
        agh = torch.randn(nchannel, nshifts, ncomb_v) * 0.1

        data = {
            "g_sv": g_sv,
            "nbmat": nbmat,
            "_nb_mode": torch.tensor(1),
        }
        expected = _run_pytorch_convsv(data, a, agh, nchannel, nshifts, ncomb_v, d2features=True)

        a_tt = _to_tt(a, device)
        g_sv_tt = _to_tt(g_sv, device)
        nbmat_tt = _to_tt(nbmat.to(torch.int32), device)
        agh_tt = _to_tt(agh, device)

        data_tt = {"g_sv": g_sv_tt}
        out_tt = ttnn_conv_sv(data_tt, a_tt, agh_tt, 1, nbmat_tt, device, d2features=True)
        out = _to_torch(out_tt)

        assert out.shape == expected.shape
        assert torch.allclose(out, expected.to(torch.bfloat16), atol=0.1)

    def test_mode1_small(self, device):
        # Use small values to avoid bfloat16 precision issues
        n_atoms, max_neighbors = 2, 2
        nchannel, nshifts, ncomb_v = 2, 3, 4
        a = torch.randn(n_atoms, nchannel) * 0.1
        g_sv = torch.randn(n_atoms, max_neighbors, nshifts, 4) * 0.1
        nbmat = torch.tensor([[0, 1], [1, 0]])
        agh = torch.randn(nchannel, nshifts, ncomb_v) * 0.1

        data = {
            "g_sv": g_sv,
            "nbmat": nbmat,
            "_nb_mode": torch.tensor(1),
        }
        expected = _run_pytorch_convsv(data, a, agh, nchannel, nshifts, ncomb_v, d2features=False)

        a_tt = _to_tt(a, device)
        g_sv_tt = _to_tt(g_sv, device)
        nbmat_tt = _to_tt(nbmat.to(torch.int32), device)
        agh_tt = _to_tt(agh, device)

        data_tt = {"g_sv": g_sv_tt}
        out_tt = ttnn_conv_sv(data_tt, a_tt, agh_tt, 1, nbmat_tt, device, d2features=False)
        out = _to_torch(out_tt)

        assert out.shape == expected.shape
        assert torch.allclose(out, expected.to(torch.bfloat16), atol=0.1)
