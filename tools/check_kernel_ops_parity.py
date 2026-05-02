from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def c_gelu_new(x: torch.Tensor) -> torch.Tensor:
    c = 0.7978845608028654
    return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * x * x * x)))


def c_softmax_rows(x: torch.Tensor) -> torch.Tensor:
    # x: [rows, cols]
    m = x.max(dim=-1, keepdim=True).values
    y = torch.exp(x - m)
    return y / y.sum(dim=-1, keepdim=True)


def c_rope_cos_sin(position_ids: torch.Tensor, head_dim: int, rope_base: float = 10000.0):
    # match moss_rope_cos_sin: inv[i] = 1 / rope_base^(2i/head_dim)
    half = head_dim // 2
    inv = torch.tensor(
        [1.0 / (rope_base ** ((2.0 * i) / float(head_dim))) for i in range(half)],
        dtype=torch.float32,
    )
    # [S, half]
    ang = position_ids.to(torch.float32).unsqueeze(-1) * inv.unsqueeze(0)
    c = torch.cos(ang)
    s = torch.sin(ang)
    # interleave to [S, head_dim]
    cos = torch.empty(position_ids.shape[0], head_dim, dtype=torch.float32)
    sin = torch.empty_like(cos)
    cos[:, 0::2] = c
    cos[:, 1::2] = c
    sin[:, 0::2] = s
    sin[:, 1::2] = s
    return cos, sin


def c_apply_rope_inplace(qk: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # qk: [S, H, Dh], cos/sin: [S, Dh]
    out = qk.clone()
    s, h, d = out.shape
    for si in range(s):
        for hi in range(h):
            v = out[si, hi]
            even = v[0::2].clone()
            odd = v[1::2].clone()
            v[0::2] = even * cos[si, 0::2] - odd * sin[si, 0::2]
            v[1::2] = odd * cos[si, 1::2] + even * sin[si, 1::2]
    return out


def torch_apply_rope(qk: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # qk: [S,H,Dh], cos/sin [S,Dh] -> [S,1,Dh]
    q = qk
    cosu = cos.unsqueeze(1)
    sinu = sin.unsqueeze(1)
    even = q[..., 0::2]
    odd = q[..., 1::2]
    rot = torch.empty_like(q)
    rot[..., 0::2] = -odd
    rot[..., 1::2] = even
    return q * cosu + rot * sinu


def main() -> int:
    torch.manual_seed(0)

    # GELU new parity
    x = torch.randn(4096, dtype=torch.float32)
    d_gelu = (c_gelu_new(x) - F.gelu(x, approximate="tanh")).abs().max().item()
    print(f"[gelu_new] max_abs_diff_vs_torch_tanh={d_gelu:.9g}")

    # softmax parity
    s = torch.randn(32, 257, dtype=torch.float32) * 3.0
    d_soft = (c_softmax_rows(s) - F.softmax(s, dim=-1)).abs().max().item()
    print(f"[softmax_rows] max_abs_diff_vs_torch={d_soft:.9g}")

    # RoPE parity
    S, H, Dh = 211, 12, 64
    pos = torch.arange(S, dtype=torch.long)
    qk = torch.randn(S, H, Dh, dtype=torch.float32)
    cos, sin = c_rope_cos_sin(pos, Dh, rope_base=10000.0)
    c_out = c_apply_rope_inplace(qk, cos, sin)
    t_out = torch_apply_rope(qk, cos, sin)
    d_rope = (c_out - t_out).abs().max().item()
    print(f"[rope] max_abs_diff_c_vs_torch={d_rope:.9g}")

    if d_gelu < 1e-6 and d_soft < 1e-6 and d_rope < 1e-6:
        print("[ok] math kernels match torch formulas (fp32-level).")
    else:
        print("[warn] kernel parity exceeded fp32 tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
