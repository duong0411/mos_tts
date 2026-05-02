from __future__ import annotations

import json
from pathlib import Path

from safetensors import safe_open


def main() -> int:
    ws = Path(__file__).resolve().parents[1]
    ckpt = ws / "weight" / "checkpoint"
    cfg = json.loads((ckpt / "config.json").read_text(encoding="utf-8"))
    ng = int(cfg["gpt2_config"]["n_layer"])
    nl = int(cfg["local_transformer_layers"])

    required: list[str] = []
    suffixes = [
        "ln_1.weight",
        "ln_1.bias",
        "attn.c_attn.weight",
        "attn.c_attn.bias",
        "attn.c_proj.weight",
        "attn.c_proj.bias",
        "ln_2.weight",
        "ln_2.bias",
        "mlp.fc_in.weight",
        "mlp.fc_in.bias",
        "mlp.fc_out.weight",
        "mlp.fc_out.bias",
    ]

    for i in range(ng):
        p = f"transformer.h.{i}."
        required.extend(p + s for s in suffixes)
    for i in range(nl):
        p = f"local_transformer.h.{i}."
        required.extend(p + s for s in suffixes)
    required.extend(
        [
            "transformer.ln_f.weight",
            "transformer.ln_f.bias",
            "local_transformer.ln_f.weight",
            "local_transformer.ln_f.bias",
        ]
    )

    sf_path = ckpt / "pytorch_model.safetensors"
    with safe_open(str(sf_path), framework="pt", device="cpu") as sf:
        keys = set(sf.keys())
    missing = [k for k in required if k not in keys]

    print(
        f"[weights] global_layers={ng} local_layers={nl} "
        f"required_tensors={len(required)} missing={len(missing)}"
    )
    if missing:
        for k in missing[:20]:
            print(f"[missing] {k}")
    else:
        print("[ok] all required global/local layer tensors exist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
