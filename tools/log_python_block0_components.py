import argparse
import importlib
import sys
from pathlib import Path

import torch


def stats_line(tag: str, x: torch.Tensor) -> str:
    v = x.detach().to("cpu", dtype=torch.float32).reshape(-1)
    s = float(v.sum().item())
    l2 = float(torch.linalg.vector_norm(v).item())
    mn = float(v.min().item())
    mx = float(v.max().item())
    return f"[py_comp] {tag} sum={s:.9g} l2={l2:.9g} min={mn:.9g} max={mx:.9g}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Log Python global block0 component stats.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--prompt-audio-path", required=True)
    parser.add_argument("--out-wav", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-frames", type=int, default=96)
    parser.add_argument("--do-sample", type=int, default=0, choices=[0, 1])
    parser.add_argument("--text-temperature", type=float, default=1.0)
    parser.add_argument("--audio-temperature", type=float, default=0.8)
    parser.add_argument("--audio-top-p", type=float, default=0.95)
    parser.add_argument("--audio-top-k", type=int, default=25)
    parser.add_argument("--audio-repetition-penalty", type=float, default=1.2)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import infer as infer_mod  # pylint: disable=import-error

    infer_mod.set_logging()
    parsed = infer_mod.parse_args(
        [
            "--device",
            "cpu",
            "--seed",
            str(args.seed),
            "--max-new-frames",
            str(args.max_new_frames),
            "--do-sample",
            str(args.do_sample),
            "--text-temperature",
            str(args.text_temperature),
            "--audio-temperature",
            str(args.audio_temperature),
            "--audio-top-p",
            str(args.audio_top_p),
            "--audio-top-k",
            str(args.audio_top_k),
            "--audio-repetition-penalty",
            str(args.audio_repetition_penalty),
            "--prompt-audio-path",
            args.prompt_audio_path,
            "--text",
            args.text,
            "--output-audio-path",
            args.out_wav,
        ]
    )

    device = infer_mod.resolve_device(parsed.device)
    dtype = infer_mod.resolve_dtype(parsed.dtype, device)
    if parsed.seed is not None:
        torch.manual_seed(parsed.seed)
    model = infer_mod.load_model(parsed.checkpoint, device=device, dtype=dtype)
    sampling_kwargs = infer_mod.resolve_sampling_kwargs(parsed)

    raw_text = infer_mod.resolve_text(parsed)
    raw_prompt_text = infer_mod.resolve_prompt_text(parsed) or ""
    enable_wetext = bool(parsed.enable_wetext_processing) and not bool(parsed.disable_wetext_processing)
    enable_normalize = bool(parsed.enable_normalize_tts_text) and not bool(parsed.disable_normalize_tts_text)
    text_normalizer_manager = None
    if enable_wetext:
        text_normalizer_manager = infer_mod.WeTextProcessingManager()
        snapshot = text_normalizer_manager.ensure_ready()
        if not snapshot.ready:
            raise RuntimeError(snapshot.error or snapshot.message)

    prepared = infer_mod.prepare_tts_request_texts(
        text=raw_text,
        prompt_text=raw_prompt_text,
        voice="",
        enable_wetext=enable_wetext,
        enable_normalize_tts_text=enable_normalize,
        text_normalizer_manager=text_normalizer_manager,
    )
    text = str(prepared["text"])
    prompt_text = str(prepared["prompt_text"]).strip() or None

    block0 = model.transformer.h[0]
    g_call = {"n": 0}
    comp = {"rope_count": 0, "done": False}

    attn_module = importlib.import_module(block0.attn.__class__.__module__)
    orig_apply_rope = attn_module.apply_rotary_pos_emb

    orig_global_forward = model.transformer.forward
    orig_ln1 = block0.ln_1.forward
    orig_attn_forward = block0.attn.forward
    orig_block0_forward = block0.forward
    orig_c_attn = block0.attn.c_attn.forward
    orig_c_proj = block0.attn.c_proj.forward
    orig_ln2 = block0.ln_2.forward
    orig_mlp_forward = block0.mlp.forward
    orig_fc_in = block0.mlp.fc_in.forward
    orig_fc_out = block0.mlp.fc_out.forward

    def should_log():
        return g_call["n"] == 1 and not comp["done"]

    def wrapped_apply_rope(hidden_states, cos, sin):
        out = orig_apply_rope(hidden_states, cos, sin)
        if should_log():
            comp["rope_count"] += 1
            if comp["rope_count"] == 1:
                print(stats_line("comp_q_post_rope", out[0, -1, :, :]))
            elif comp["rope_count"] == 2:
                print(stats_line("comp_k_post_rope", out[0, -1, :, :]))
        return out

    def wrapped_ln1(x):
        out = orig_ln1(x)
        if should_log():
            print(stats_line("comp_ln1_out", out[0, -1, :]))
        return out

    def wrapped_c_attn(x):
        out = orig_c_attn(x)
        if should_log():
            D = out.shape[-1] // 3
            print(stats_line("comp_q_pre_rope", out[0, -1, :D]))
            print(stats_line("comp_k_pre_rope", out[0, -1, D : 2 * D]))
            print(stats_line("comp_v", out[0, -1, 2 * D :]))
        return out

    def wrapped_c_proj(x):
        out = orig_c_proj(x)
        if should_log():
            print(stats_line("comp_attn_out_pre_proj", x[0, -1, :]))
            print(stats_line("comp_c_proj_out", out[0, -1, :]))
        return out

    def wrapped_attn(hidden_states, **kwargs):
        out, present = orig_attn_forward(hidden_states, **kwargs)
        return out, present

    def wrapped_ln2(x):
        out = orig_ln2(x)
        if should_log():
            print(stats_line("comp_ln2_out", out[0, -1, :]))
        return out

    def wrapped_fc_in(x):
        out = orig_fc_in(x)
        return out

    def wrapped_fc_out(x):
        out = orig_fc_out(x)
        if should_log():
            print(stats_line("comp_mlp_fc_out", out[0, -1, :]))
        return out

    def wrapped_mlp(x):
        z = orig_fc_in(x)
        z = block0.mlp.act(z)
        if should_log():
            print(stats_line("comp_mlp_fc_in_gelu", z[0, -1, :]))
        z = orig_fc_out(z)
        z = block0.mlp.dropout(z)
        return z

    def wrapped_block0_forward(
        hidden_states,
        attention_mask=None,
        position_ids=None,
        packed_metadata=None,
        layer_past=None,
        use_cache=False,
    ):
        if should_log():
            print(stats_line("comp_input", hidden_states[0, -1, :]))
        ln1_out = block0.ln_1(hidden_states)
        attn_output, present = block0.attn(
            ln1_out,
            attention_mask=attention_mask,
            position_ids=position_ids,
            packed_metadata=packed_metadata,
            layer_past=layer_past,
            use_cache=use_cache,
        )
        hs_attn = hidden_states + attn_output
        if should_log():
            print(stats_line("comp_post_attn_res", hs_attn[0, -1, :]))
        ln2_out = block0.ln_2(hs_attn)
        mlp_out = block0.mlp(ln2_out)
        hs_out = hs_attn + mlp_out
        if should_log():
            print(stats_line("comp_post_mlp_res", hs_out[0, -1, :]))
        return hs_out, present

    def wrapped_global_forward(*a, **kw):
        g_call["n"] += 1
        out = orig_global_forward(*a, **kw)
        if g_call["n"] == 1:
            comp["done"] = True
        return out

    attn_module.apply_rotary_pos_emb = wrapped_apply_rope
    block0.ln_1.forward = wrapped_ln1
    block0.attn.c_attn.forward = wrapped_c_attn
    block0.attn.c_proj.forward = wrapped_c_proj
    block0.attn.forward = wrapped_attn
    block0.forward = wrapped_block0_forward
    block0.ln_2.forward = wrapped_ln2
    block0.mlp.fc_in.forward = wrapped_fc_in
    block0.mlp.fc_out.forward = wrapped_fc_out
    block0.mlp.forward = wrapped_mlp
    model.transformer.forward = wrapped_global_forward

    try:
        result = model.inference(
            text=text,
            output_audio_path=parsed.output_audio_path,
            mode=parsed.mode,
            prompt_text=prompt_text,
            prompt_audio_path=parsed.prompt_audio_path,
            reference_audio_path=parsed.reference_audio_path,
            text_tokenizer_path=parsed.text_tokenizer_path,
            audio_tokenizer_type=infer_mod.MOSS_AUDIO_TOKENIZER_TYPE,
            audio_tokenizer_pretrained_name_or_path=parsed.audio_tokenizer_pretrained_name_or_path,
            device=device,
            nq=parsed.nq,
            max_new_frames=parsed.max_new_frames,
            voice_clone_max_text_tokens=parsed.voice_clone_max_text_tokens,
            voice_clone_max_memory_per_sample_gb=parsed.voice_clone_max_memory_per_sample_gb,
            do_sample=bool(parsed.do_sample),
            use_kv_cache=True,
            **sampling_kwargs,
        )
    finally:
        attn_module.apply_rotary_pos_emb = orig_apply_rope
    print(f"python_frames={int(result['audio_token_ids'].shape[0])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
