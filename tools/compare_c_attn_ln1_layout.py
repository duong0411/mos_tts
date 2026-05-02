"""
Verify attn.c_attn weight/bias packing vs C++ GEMV layout and LN1 parity.

Checks:
  (1) safetensors mmap tensor bytes == in-memory HF model weight for block0 attn.c_attn
  (2) Torch F.linear(torch.mv semantics) matches row-major GEMV loops (C++ moss_gemv_bf16_nt_bias)
  (3) On first global transformer forward, last-row LN1 stats + c_attn GEMV delta F.linear vs bf16 GEMV simulation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


def moss_style_gemv_bias(W_bf16: torch.Tensor, b_bf16: torch.Tensor, x_f32: torch.Tensor) -> torch.Tensor:
    """
    y[r]= b[r]_as_f32 + sum_c W_bf16[r,c]_as_f32 * x[c] — matches moss_gemv_bf16_nt_bias row loop.
    W: [rows, cols] bf16 on disk/model.
    """
    rows, cols = W_bf16.shape
    out = torch.empty(rows, dtype=torch.float32)
    xf = x_f32.to(dtype=torch.float32).contiguous().view(-1)
    Wf = W_bf16.float()
    bf = b_bf16.float().view(-1)
    assert xf.numel() == cols
    for r in range(rows):
        out[r] = bf[r] + torch.dot(Wf[r], xf).item()
    return out


def manual_layer_norm_eps(
    x: torch.Tensor,
    gamma_bf16: torch.Tensor,
    beta_bf16: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """RMS LN over vector x [D]; gamma/bias converted bf16→fp32 like moss_layernorm_bf16."""
    x = x.to(torch.float64)
    mean = x.mean()
    var = torch.mean((x - mean) ** 2)
    inv = torch.rsqrt(var + eps).to(torch.float32)
    x32 = x.to(torch.float32)
    mean32 = mean.to(torch.float32)
    g = gamma_bf16.float().to(torch.float32)
    bb = beta_bf16.float().to(torch.float32)
    return ((x32 - mean32) * inv) * g + bb


def main() -> int:
    ws = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", default=str(ws / "weight" / "checkpoint"))
    parser.add_argument("--repo", default=str(ws / "MOSS-TTS-Nano"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    ckpt = Path(args.checkpoint).resolve()
    sf_path = ckpt / "pytorch_model.safetensors"
    if not sf_path.is_file():
        print(f"[err] missing {sf_path}", file=sys.stderr)
        return 1

    repo = Path(args.repo).resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    device = torch.device(args.device)

    pt_key_w = "transformer.h.0.attn.c_attn.weight"
    pt_key_b = "transformer.h.0.attn.c_attn.bias"
    pt_key_ln1w = "transformer.h.0.ln_1.weight"
    pt_key_ln1b = "transformer.h.0.ln_1.bias"

    with safe_open(sf_path, framework="pt", device=str(device)) as sf:
        st_w = sf.get_tensor(pt_key_w)
        st_b = sf.get_tensor(pt_key_b)

    from transformers import AutoModelForCausalLM  # after sys.path tweak

    model = AutoModelForCausalLM.from_pretrained(str(ckpt), trust_remote_code=True)
    model.eval()
    model.to(device=device, dtype=torch.bfloat16)
    blk0 = model.transformer.h[0]

    mw = blk0.attn.c_attn.weight.data
    mb = blk0.attn.c_attn.bias.data

    dw = (st_w.cpu().float() - mw.cpu().float()).abs().max().item()
    db = (st_b.cpu().float() - mb.cpu().float()).abs().max().item()
    print(f"[pack] safetensors vs HF model max(abs diff) weight={dw:.9g} bias={db:.9g}")
    print(f"[pack] c_attn.weight shape={tuple(st_w.shape)} bias shape={tuple(st_b.shape)}")

    rnd = torch.randn(mw.shape[1], device=device, dtype=torch.float32) * 0.02
    y_torch = F.linear(rnd.unsqueeze(0), mw.float(), mb.float()).squeeze(0).cpu()
    y_loop = moss_style_gemv_bias(mw, mb, rnd).cpu()
    y_mv = torch.mv(mw.float(), rnd).cpu() + mb.float().cpu()
    print(
        "[gemv_layout] rnd x: loop_vs_torch_linear max(abs)="
        f"{float((y_loop - y_torch).abs().max())} mv_vs_torch {float((y_mv - y_torch).abs().max())}"
    )

    g_cfg = getattr(model.config, "gpt2_config", None)
    ln_eps = float(getattr(g_cfg, "layer_norm_epsilon", 1e-5)) if g_cfg else 1e-5

    ln1w = blk0.ln_1.weight.data
    ln1b = blk0.ln_1.bias.data
    xh = torch.randn(mw.shape[1], device=device, dtype=torch.float32) * 0.03
    y_hf = blk0.ln_1(xh.unsqueeze(0).to(dtype=mw.dtype).to(device)).squeeze(0).float().cpu()
    y_man = manual_layer_norm_eps(xh.cpu(), ln1w.cpu(), ln1b.cpu(), ln_eps).float()
    xh32 = xh.to(torch.float32)
    d = int(xh32.shape[0])
    y_fn32 = F.layer_norm(
        xh32.unsqueeze(0),
        (d,),
        weight=blk0.ln_1.weight.float(),
        bias=blk0.ln_1.bias.float(),
        eps=ln_eps,
    ).squeeze(0)
    print(
        "[ln1] manual(moss-like: fp64 mean/var fp32 affine, bf16→fp32 gamma) vs F.layer_norm(fp32 ALL) max(abs)=",
        float((y_man - y_fn32.cpu()).abs().max()),
        "| vs nn.LayerNorm(bf16 activations) max(abs)=",
        float((y_man - y_hf).abs().max()),
        " ln_eps=", ln_eps,
        " (Cpp runs fp32 residuals + bf16 stored LN weights)",
    )

    y_qkv_torch = F.linear(y_hf.unsqueeze(0).to(device=device), mw.float(), mb.float()).squeeze(0).cpu()
    y_qkv_moss_sim = moss_style_gemv_bias(
        mw, mb, y_hf.to(device=device, dtype=torch.float32)
    )
    print(
        "[c_attn ln1_chain] moss_sim_bf16GEMV_vs_torch_linear ln1_man max(abs)=",
        float((y_qkv_moss_sim.cpu() - y_qkv_torch).abs().max()),
    )

    captured: dict[str, torch.Tensor | int] = {}

    def hook(_m, inp):
        # Block passes self.ln_1(hidden) into attn — inp[0] is already LN1 output.
        x_ln = inp[0]
        lr = int(x_ln.shape[1] - 1)
        xrow = x_ln[0, lr].detach().float().cpu().contiguous()
        W = mw.detach().float().cpu().contiguous()
        bvec = mb.detach().float().cpu().contiguous()
        y_bf16_loop = moss_style_gemv_bias(mw.detach().cpu(), mb.detach().cpu(), xrow)
        y_lin = torch.mv(W, xrow).cpu() + bvec.cpu()
        captured.clear()
        captured["last_row"] = lr
        captured["x_ln_row"] = xrow.clone()
        captured["torch_mv_fp32Weights"] = y_lin.clone()
        captured["moss_style_bf16W"] = y_bf16_loop.clone()

    h = blk0.attn.register_forward_pre_hook(hook, with_kwargs=False)

    with torch.no_grad():
        gconf = model.config.gpt2_config
        d = int(gconf.hidden_size)
        embeds = torch.randn(1, 8, d, device=device, dtype=mw.dtype) * 0.01
        attn_mask = torch.ones(1, 8, dtype=torch.bool, device=device)
        pos = attn_mask.long().cumsum(dim=-1) - 1
        pos = pos.masked_fill(~attn_mask, 0)
        _ = model.transformer(
            inputs_embeds=embeds,
            attention_mask=attn_mask,
            position_ids=pos,
        )

    h.remove()

    mv = captured.get("torch_mv_fp32Weights")
    ms = captured.get("moss_style_bf16W")
    print(
        "[dummy_forward_hooks] moss_sim_bf16GEMV_vs_mv_fp32W max(abs)=",
        float((mv - ms).abs().max()) if mv is not None and ms is not None else "n/a",
        "last_row=", captured.get("last_row"),
    )

    print(
        "[conclusion] If [pack]~0 and moss_sim matches torch_mv: cpp-side layout matches HF; "
        "cpp/python logits gaps are not from attn weight transpose packing."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
