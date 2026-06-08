from __future__ import annotations

from typing import Any

from aimnet.ttlang.ttnn_nbops import ttnn_get_ij


def ttnn_conv_sv(
    data: dict[str, Any],
    a_tt: Any,
    agh_tt: Any,
    nb_mode: int,
    nbmat_tt: Any | None,
    device: Any,
    d2features: bool = False,
) -> Any:
    """TTNN ConvSV operation.

    Computes the AIMNet2 type convolution on device using TTNN operations.
    """
    import ttnn
    from aimnet.ttlang.ops import stage_profile

    g_sv_tt = data["g_sv"]

    # Get a_j (and a_i, but we only need a_j for the contraction)
    with stage_profile("convsv.get_ij"):
        if d2features:
            a_shape = a_tt.shape
            nchannel = a_shape[-2]
            nshifts = a_shape[-1]
            a_flat = ttnn.reshape(a_tt, (-1, nchannel * nshifts))
            _, a_j_tt = ttnn_get_ij(a_flat, nb_mode, nbmat_tt, device)
            # Reshape a_j back: (..., M, nchannel*nshifts) -> (..., M, nchannel, nshifts)
            a_j_shape = tuple(a_j_tt.shape)
            a_j_tt = ttnn.reshape(a_j_tt, a_j_shape[:-1] + (nchannel, nshifts))
        else:
            _, a_j_tt = ttnn_get_ij(a_tt, nb_mode, nbmat_tt, device)

    # Compute avf_sv via einsum "...mag,...mgd->...agd" contraction over M.
    # Uses matmul instead of elementwise mul+sum to avoid intermediate tensor.
    with stage_profile("convsv.contract"):
        A = a_j_tt.shape[0]
        M = a_j_tt.shape[1]

        if nb_mode == 0:
            # Mode 0: full Cartesian product — fall back to elementwise.
            if d2features:
                a_j_tt = ttnn.unsqueeze(a_tt, 1)
                a_exp = ttnn.unsqueeze(a_j_tt, -1)
                g_exp = ttnn.unsqueeze(g_sv_tt, -4)
                prod = ttnn.mul(a_exp, g_exp)
                avf_sv = ttnn.sum(prod, dim=2)
            else:
                a_j_tt = ttnn.unsqueeze(a_tt, 1)
                a_exp = ttnn.unsqueeze(ttnn.unsqueeze(a_j_tt, -1), -1)
                g_exp = ttnn.unsqueeze(g_sv_tt, -4)
                prod = ttnn.mul(a_exp, g_exp)
                avf_sv = ttnn.sum(prod, dim=2)
        else:
            # Mode 1/2: matmul path avoids 96 MB intermediate tensor.
            if d2features:
                C = a_j_tt.shape[2]
                S = a_j_tt.shape[3]
                D = g_sv_tt.shape[-1]
                # a_j: (A, M, C, S) -> (A*S, C, M)
                a_rs = ttnn.permute(a_j_tt, (0, 3, 2, 1))
                a_rs = ttnn.reshape(a_rs, (A * S, C, M))
                # g_sv: (A, M, S, D) -> (A*S, M, D)
                g_rs = ttnn.permute(g_sv_tt, (0, 2, 1, 3))
                g_rs = ttnn.reshape(g_rs, (A * S, M, D))

                out = ttnn.matmul(a_rs, g_rs)  # (A*S, C, D)
                out = ttnn.reshape(out, (A, S, C, D))
                avf_sv = ttnn.permute(out, (0, 2, 1, 3))  # (A, C, S, D)
            else:
                C = a_j_tt.shape[2]
                S = g_sv_tt.shape[-2]
                D = g_sv_tt.shape[-1]
                # a_j: (A, M, C) -> (A, C, M)
                a_rs = ttnn.permute(a_j_tt, (0, 2, 1))
                # g_sv: (A, M, S, D) -> (A, M, S*D)
                g_rs = ttnn.reshape(g_sv_tt, (A, M, S * D))

                out = ttnn.matmul(a_rs, g_rs)  # (A, C, S*D)
                avf_sv = ttnn.reshape(out, (A, C, S, D))

    # Split avf_sv into scalar and vector parts
    avf_s, avf_v = ttnn.split(avf_sv, [1, 3], dim=-1)

    # Flatten scalar part: (..., nchannel, nshifts, 1) -> (..., nchannel*nshifts)
    s_shape = tuple(avf_s.shape)
    avf_s_flat = ttnn.reshape(avf_s, s_shape[:-3] + (s_shape[-3] * s_shape[-2],))

    # Vector projection through agh: einsum("csv,...csd->...cvd")
    # Uses matmul to avoid 27 MB intermediate tensor.
    with stage_profile("convsv.agh_proj"):
        C_v = agh_tt.shape[0]
        S_v = agh_tt.shape[1]
        V = agh_tt.shape[2]
        A_v = avf_v.shape[0]
        # avf_v: (A, C, S, 3) -> (A*C, 3, S)
        avf_rs = ttnn.permute(avf_v, (0, 1, 3, 2))
        avf_rs = ttnn.reshape(avf_rs, (A_v * C_v, 3, S_v))
        # agh: (C, S, V) -> expand to (A*C, S, V)
        agh_exp = ttnn.unsqueeze(agh_tt, 0)  # (1, C, S, V)
        agh_exp = ttnn.repeat(agh_exp, (A_v, 1, 1, 1))  # (A, C, S, V)
        agh_exp = ttnn.reshape(agh_exp, (A_v * C_v, S_v, V))

        out_proj = ttnn.matmul(avf_rs, agh_exp)  # (A*C, 3, V)
        out_proj = ttnn.reshape(out_proj, (A_v, C_v, 3, V))
        avf_v_sum_s = ttnn.permute(out_proj, (0, 1, 3, 2))  # (A, C, V, 3)

        avf_v_sq = ttnn.mul(avf_v_sum_s, avf_v_sum_s)
        avf_v_sum = ttnn.sum(avf_v_sq, dim=-1)

    # Flatten and concat
    v_shape = tuple(avf_v_sum.shape)
    avf_v_flat_out = ttnn.reshape(avf_v_sum, v_shape[:-2] + (v_shape[-2] * v_shape[-1],))
    conv_out = ttnn.concat([avf_s_flat, avf_v_flat_out], dim=-1)
    return conv_out
