from __future__ import annotations

import pytest
import torch

from aimnet.ops import nse
from aimnet.ttlang.ttnn_charge import ttnn_update_q

pytestmark = pytest.mark.tt


@pytest.fixture
def tt_device():
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
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _to_torch(tt_tensor):
    import ttnn
    return ttnn.to_torch(tt_tensor)


def _ref_update_q(
    charge,
    charges,
    a,
    _out,
    num_charge_channels,
    mol_idx,
    nb_mode,
    delta_q=True,
    epsilon=1.0e-6,
):
    _q, _f, delta_a = _out.split(
        [
            num_charge_channels,
            num_charge_channels,
            _out.shape[-1] - 2 * num_charge_channels,
        ],
        dim=-1,
    )

    # delta_Q (charge conservation violation of raw _q)
    if nb_mode in (0, 2):
        delta_Q = charge - _q.sum(dim=1)
    elif nb_mode == 1:
        out_size = int(mol_idx[-1].item()) + 1
        if _q.ndim == 2:
            idx = mol_idx.unsqueeze(-1).expand(-1, _q.shape[1])
            q_sum = torch.zeros(out_size, _q.shape[1], dtype=_q.dtype, device=_q.device)
        else:
            idx = mol_idx
            q_sum = torch.zeros(out_size, dtype=_q.dtype, device=_q.device)
        q_sum.scatter_add_(0, idx, _q)
        delta_Q = charge - q_sum
    else:
        raise ValueError(f"Invalid neighbor mode: {nb_mode}")

    q_u = charges + _q if delta_q else _q
    f_u = _f.pow(2)

    data = {"_nb_mode": torch.tensor(nb_mode)}
    if nb_mode in (0, 2):
        data["_input_padded"] = torch.tensor(False)
    elif nb_mode == 1:
        mol_sizes = torch.bincount(mol_idx)
        mol_sizes[-1] -= 1
        data["mol_idx"] = mol_idx
        data["mol_sizes"] = mol_sizes
        data["_input_padded"] = torch.tensor(True)

    q = nse(charge, q_u, f_u, data, epsilon=epsilon)
    a = a + delta_a.view_as(a)

    return q, a, delta_Q


class TestUpdateQ:
    def test_mode0_delta_q_false(self, tt_device):
        B, N, out_features = 2, 4, 6
        num_charge_channels = 1

        charge = torch.tensor([[1.0], [2.0]], dtype=torch.bfloat16)
        charges = torch.randn(B, N, num_charge_channels, dtype=torch.bfloat16)
        a = torch.randn(B, N, out_features - 2 * num_charge_channels, dtype=torch.bfloat16)
        _out = torch.randn(B, N, out_features, dtype=torch.bfloat16)

        charge_tt = _to_tt(charge, tt_device)
        charges_tt = _to_tt(charges, tt_device)
        a_tt = _to_tt(a, tt_device)
        _out_tt = _to_tt(_out, tt_device)

        q_tt, a_tt_new, delta_Q_tt = ttnn_update_q(
            charge_tt,
            charges_tt,
            a_tt,
            _out_tt,
            num_charge_channels,
            None,
            0,
            tt_device,
            delta_q=False,
            epsilon=1.0e-6,
        )

        q_ref, a_ref, delta_Q_ref = _ref_update_q(
            charge,
            charges,
            a,
            _out,
            num_charge_channels,
            None,
            0,
            delta_q=False,
            epsilon=1.0e-6,
        )

        assert torch.allclose(_to_torch(q_tt), q_ref, atol=5e-2, rtol=5e-2)
        assert torch.allclose(_to_torch(a_tt_new), a_ref, atol=5e-2, rtol=5e-2)
        assert torch.allclose(_to_torch(delta_Q_tt), delta_Q_ref, atol=5e-2, rtol=5e-2)

    def test_mode0_delta_q_true(self, tt_device):
        B, N, out_features = 2, 4, 6
        num_charge_channels = 1

        charge = torch.tensor([[1.0], [2.0]], dtype=torch.bfloat16)
        charges = torch.randn(B, N, num_charge_channels, dtype=torch.bfloat16)
        a = torch.randn(B, N, out_features - 2 * num_charge_channels, dtype=torch.bfloat16)
        _out = torch.randn(B, N, out_features, dtype=torch.bfloat16)

        charge_tt = _to_tt(charge, tt_device)
        charges_tt = _to_tt(charges, tt_device)
        a_tt = _to_tt(a, tt_device)
        _out_tt = _to_tt(_out, tt_device)

        q_tt, a_tt_new, delta_Q_tt = ttnn_update_q(
            charge_tt,
            charges_tt,
            a_tt,
            _out_tt,
            num_charge_channels,
            None,
            0,
            tt_device,
            delta_q=True,
            epsilon=1.0e-6,
        )

        q_ref, a_ref, delta_Q_ref = _ref_update_q(
            charge,
            charges,
            a,
            _out,
            num_charge_channels,
            None,
            0,
            delta_q=True,
            epsilon=1.0e-6,
        )

        assert torch.allclose(_to_torch(q_tt), q_ref, atol=5e-2, rtol=5e-2)
        assert torch.allclose(_to_torch(a_tt_new), a_ref, atol=5e-2, rtol=5e-2)
        assert torch.allclose(_to_torch(delta_Q_tt), delta_Q_ref, atol=5e-2, rtol=5e-2)

    def test_mode1_delta_q_false(self, tt_device):
        n_atoms = 5
        n_mols = 2
        out_features = 6
        num_charge_channels = 1

        mol_idx = torch.tensor([0, 0, 1, 1, 1], dtype=torch.int32)
        charge = torch.tensor([[1.0], [2.0]], dtype=torch.bfloat16)
        charges = torch.randn(n_atoms, num_charge_channels, dtype=torch.bfloat16)
        a = torch.randn(n_atoms, out_features - 2 * num_charge_channels, dtype=torch.bfloat16)
        _out = torch.randn(n_atoms, out_features, dtype=torch.bfloat16)

        charge_tt = _to_tt(charge, tt_device)
        charges_tt = _to_tt(charges, tt_device)
        a_tt = _to_tt(a, tt_device)
        _out_tt = _to_tt(_out, tt_device)
        mol_idx_tt = _to_tt(mol_idx, tt_device)

        q_tt, a_tt_new, delta_Q_tt = ttnn_update_q(
            charge_tt,
            charges_tt,
            a_tt,
            _out_tt,
            num_charge_channels,
            mol_idx_tt,
            1,
            tt_device,
            delta_q=False,
            epsilon=1.0e-6,
        )

        q_ref, a_ref, delta_Q_ref = _ref_update_q(
            charge,
            charges,
            a,
            _out,
            num_charge_channels,
            mol_idx,
            1,
            delta_q=False,
            epsilon=1.0e-6,
        )

        assert torch.allclose(_to_torch(q_tt), q_ref, atol=5e-2, rtol=5e-2)
        assert torch.allclose(_to_torch(a_tt_new), a_ref, atol=5e-2, rtol=5e-2)
        assert torch.allclose(_to_torch(delta_Q_tt), delta_Q_ref, atol=5e-2, rtol=5e-2)

    def test_mode0_two_channels(self, tt_device):
        B, N, out_features = 2, 4, 8
        num_charge_channels = 2

        charge = torch.tensor([[1.0, 0.5], [2.0, 1.0]], dtype=torch.bfloat16)
        charges = torch.randn(B, N, num_charge_channels, dtype=torch.bfloat16)
        a = torch.randn(B, N, out_features - 2 * num_charge_channels, dtype=torch.bfloat16)
        _out = torch.randn(B, N, out_features, dtype=torch.bfloat16)

        charge_tt = _to_tt(charge, tt_device)
        charges_tt = _to_tt(charges, tt_device)
        a_tt = _to_tt(a, tt_device)
        _out_tt = _to_tt(_out, tt_device)

        q_tt, a_tt_new, delta_Q_tt = ttnn_update_q(
            charge_tt,
            charges_tt,
            a_tt,
            _out_tt,
            num_charge_channels,
            None,
            0,
            tt_device,
            delta_q=True,
            epsilon=1.0e-6,
        )

        q_ref, a_ref, delta_Q_ref = _ref_update_q(
            charge,
            charges,
            a,
            _out,
            num_charge_channels,
            None,
            0,
            delta_q=True,
            epsilon=1.0e-6,
        )

        assert torch.allclose(_to_torch(q_tt), q_ref, atol=5e-2, rtol=5e-2)
        assert torch.allclose(_to_torch(a_tt_new), a_ref, atol=5e-2, rtol=5e-2)
        assert torch.allclose(_to_torch(delta_Q_tt), delta_Q_ref, atol=5e-2, rtol=5e-2)
