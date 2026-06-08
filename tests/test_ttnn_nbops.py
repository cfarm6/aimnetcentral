from __future__ import annotations

import os

import pytest
import torch

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
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )


def _to_torch(tt_tensor):
    import ttnn

    return ttnn.to_torch(tt_tensor)


class TestMaskI:
    def test_mode0_mask(self, device):
        from aimnet.ttlang.ttnn_nbops import ttnn_mask_i_

        x = torch.ones(3, 4)
        mask = torch.tensor([False, True, False])
        x_tt = _to_tt(x, device)
        mask_tt = _to_tt(mask, device)
        out = _to_torch(ttnn_mask_i_(x_tt, mask_tt, 0.0))
        expected = x.clone().to(torch.bfloat16)
        expected[1] = 0.0
        assert torch.allclose(out, expected)

    def test_mode1_no_mask(self, device):
        from aimnet.ttlang.ttnn_nbops import ttnn_mask_i_

        x = torch.ones(3, 4)
        x_tt = _to_tt(x, device)
        out = _to_torch(ttnn_mask_i_(x_tt, None, 0.0))
        assert torch.allclose(out, x.to(torch.bfloat16))


class TestMaskIJ:
    def test_pair_mask(self, device):
        from aimnet.ttlang.ttnn_nbops import ttnn_mask_ij_

        x = torch.ones(3, 4, 5)
        mask = torch.tensor([[False, True, False, False], [True, False, True, False], [False, False, False, True]])
        x_tt = _to_tt(x, device)
        mask_tt = _to_tt(mask, device)
        out = _to_torch(ttnn_mask_ij_(x_tt, mask_tt, 0.0))
        expected = x.clone().to(torch.bfloat16)
        expected[mask] = 0.0
        assert torch.allclose(out, expected)


class TestGetI:
    def test_mode0(self, device):
        from aimnet.ttlang.ttnn_nbops import ttnn_get_i

        x = torch.arange(6).reshape(3, 2).to(torch.bfloat16)
        x_tt = _to_tt(x, device)
        out = _to_torch(ttnn_get_i(x_tt, 0))
        assert out.shape == (3, 1, 2)

    def test_mode1(self, device):
        from aimnet.ttlang.ttnn_nbops import ttnn_get_i

        x = torch.arange(6).reshape(3, 2).to(torch.bfloat16)
        x_tt = _to_tt(x, device)
        out = _to_torch(ttnn_get_i(x_tt, 1))
        assert out.shape == (3, 1, 2)


class TestGetIJ:
    def test_mode0(self, device):
        from aimnet.ttlang.ttnn_nbops import ttnn_get_ij

        x = torch.arange(6).reshape(3, 2).to(torch.bfloat16)
        x_tt = _to_tt(x, device)
        x_i_tt, x_j_tt = ttnn_get_ij(x_tt, 0, None, device)
        x_i = _to_torch(x_i_tt)
        x_j = _to_torch(x_j_tt)
        assert x_i.shape == (3, 1, 2)
        assert x_j.shape == (1, 3, 2)

    def test_mode1(self, device):
        from aimnet.ttlang.ttnn_nbops import ttnn_get_ij

        x = torch.arange(6).reshape(3, 2).to(torch.bfloat16)
        nbmat = torch.tensor([[0, 1], [1, 2], [0, 2]]).to(torch.uint32)
        x_tt = _to_tt(x, device)
        nbmat_tt = _to_tt(nbmat, device)
        # typecast to uint32 for embedding
        import ttnn

        nbmat_tt = ttnn.typecast(nbmat_tt, ttnn.uint32)
        x_i_tt, x_j_tt = ttnn_get_ij(x_tt, 1, nbmat_tt, device)
        x_i = _to_torch(x_i_tt)
        x_j = _to_torch(x_j_tt)
        assert x_i.shape == (3, 1, 2)
        assert x_j.shape == (3, 2, 2)
        expected = torch.embedding(x, nbmat.long())
        assert torch.allclose(x_j, expected)


class TestMolSum:
    def test_mode0(self, device):
        from aimnet.ttlang.ttnn_nbops import ttnn_mol_sum

        x = torch.arange(6).reshape(2, 3).to(torch.bfloat16)
        x_tt = _to_tt(x, device)
        out = _to_torch(ttnn_mol_sum(x_tt, 0, None, device))
        expected = x.sum(dim=1)
        assert torch.allclose(out, expected)

    def test_mode1(self, device):
        from aimnet.ttlang.ttnn_nbops import ttnn_mol_sum

        x = torch.tensor([1.0, 2.0, 3.0, 4.0]).to(torch.bfloat16)
        mol_idx = torch.tensor([0, 0, 1, 1]).to(torch.uint32)
        x_tt = _to_tt(x, device)
        mol_idx_tt = _to_tt(mol_idx, device)
        import ttnn

        mol_idx_tt = ttnn.typecast(mol_idx_tt, ttnn.uint32)
        out = _to_torch(ttnn_mol_sum(x_tt, 1, mol_idx_tt, device))
        expected = torch.zeros(2).to(torch.bfloat16)
        expected.scatter_add_(0, mol_idx.long(), x)
        assert torch.allclose(out, expected)
