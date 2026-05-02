import argparse
import sys
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description="Log Python first-step assistant/end logits and dump codes.")
    parser.add_argument("--repo", required=True, help="Absolute path to MOSS-TTS-Nano directory.")
    parser.add_argument("--text", required=True)
    parser.add_argument("--prompt-audio-path", required=True)
    parser.add_argument("--out-wav", required=True)
    parser.add_argument("--out-codes", required=True)
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
    infer_args = infer_mod.parse_args(
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

    device = infer_mod.resolve_device(infer_args.device)
    dtype = infer_mod.resolve_dtype(infer_args.dtype, device)
    if infer_args.seed is not None:
        torch.manual_seed(infer_args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(infer_args.seed)
    model = infer_mod.load_model(infer_args.checkpoint, device=device, dtype=dtype)
    sampling_kwargs = infer_mod.resolve_sampling_kwargs(infer_args)

    first = {"seen": False}
    dbg = {}
    original = model._sample_next_assistant_text_token
    orig_decode_local = model._decode_local_last_hidden_state
    orig_sample_next_token = model._sample_next_token

    def wrapped(self, logits, do_sample, temperature, top_k=None, top_p=None):
        out = original(logits, do_sample, temperature, top_k=top_k, top_p=top_p)
        if not first["seen"]:
            aid = int(self.config.audio_assistant_slot_token_id)
            eid = int(self.config.audio_end_token_id)
            cand = logits[0, [aid, eid]].detach().cpu()
            topv, topi = torch.topk(logits[0].detach().cpu(), k=5)
            first["seen"] = True
            first["assistant_id"] = aid
            first["end_id"] = eid
            first["assistant_logit"] = float(cand[0].item())
            first["end_logit"] = float(cand[1].item())
            first["text_top_ids"] = [int(x) for x in topi.tolist()]
            first["text_top_logits"] = [float(x) for x in topv.tolist()]
            sel_ids = torch.tensor([9, 7, 505], dtype=torch.long, device=logits.device)
            sel_vals = logits[0].index_select(0, sel_ids).detach().cpu()
            first["text_sel_ids"] = [9, 7, 505]
            first["text_sel_logits"] = [float(x) for x in sel_vals.tolist()]
            first["picked"] = int(out[0].item())
            first["do_sample"] = int(bool(do_sample))
            first["text_temperature"] = float(temperature)
        return out

    def wrapped_decode_local(self, local_inputs_embeds):
        out = orig_decode_local(local_inputs_embeds)
        if "local_text_step_stats" not in dbg:
            v = out[0].detach().cpu()
            dbg["local_text_step_stats"] = {
                "sum": float(v.sum().item()),
                "l2": float(torch.linalg.vector_norm(v).item()),
                "min": float(v.min().item()),
                "max": float(v.max().item()),
            }
        return out

    def wrapped_sample_next_token(
        self,
        logits,
        do_sample,
        temperature,
        top_k=None,
        top_p=None,
        previous_token_ids=None,
        repetition_penalty=1.0,
    ):
        out = orig_sample_next_token(
            logits,
            do_sample,
            temperature,
            top_k=top_k,
            top_p=top_p,
            previous_token_ids=previous_token_ids,
            repetition_penalty=repetition_penalty,
        )
        if logits.shape[-1] > 10 and "audio_ch0_top5" not in dbg:
            v, i = torch.topk(logits[0].detach().cpu(), k=5)
            a_sel_ids = torch.tensor([137, 199, 61, 64], dtype=torch.long, device=logits.device)
            a_sel_vals = logits[0].index_select(0, a_sel_ids).detach().cpu()
            dbg["audio_ch0_top5"] = {
                "ids": [int(x) for x in i.tolist()],
                "logits": [float(x) for x in v.tolist()],
                "sel_ids": [137, 199, 61, 64],
                "sel_logits": [float(x) for x in a_sel_vals.tolist()],
            }
        return out

    model._sample_next_assistant_text_token = wrapped.__get__(model, type(model))
    model._decode_local_last_hidden_state = wrapped_decode_local.__get__(model, type(model))
    model._sample_next_token = wrapped_sample_next_token.__get__(model, type(model))

    raw_text = infer_mod.resolve_text(infer_args)
    raw_prompt_text = infer_mod.resolve_prompt_text(infer_args) or ""
    enable_wetext_processing = bool(infer_args.enable_wetext_processing) and not bool(infer_args.disable_wetext_processing)
    enable_normalize_tts_text = bool(infer_args.enable_normalize_tts_text) and not bool(
        infer_args.disable_normalize_tts_text
    )
    text_normalizer_manager = None
    if enable_wetext_processing:
        text_normalizer_manager = infer_mod.WeTextProcessingManager()
        snapshot = text_normalizer_manager.ensure_ready()
        if not snapshot.ready:
            raise RuntimeError(snapshot.error or snapshot.message)
    prepared = infer_mod.prepare_tts_request_texts(
        text=raw_text,
        prompt_text=raw_prompt_text,
        voice="",
        enable_wetext=enable_wetext_processing,
        enable_normalize_tts_text=enable_normalize_tts_text,
        text_normalizer_manager=text_normalizer_manager,
    )
    text = str(prepared["text"])
    prompt_text = str(prepared["prompt_text"]).strip() or None

    result = model.inference(
        text=text,
        output_audio_path=infer_args.output_audio_path,
        mode=infer_args.mode,
        prompt_text=prompt_text,
        prompt_audio_path=infer_args.prompt_audio_path,
        reference_audio_path=infer_args.reference_audio_path,
        text_tokenizer_path=infer_args.text_tokenizer_path,
        audio_tokenizer_type=infer_mod.MOSS_AUDIO_TOKENIZER_TYPE,
        audio_tokenizer_pretrained_name_or_path=infer_args.audio_tokenizer_pretrained_name_or_path,
        device=device,
        nq=infer_args.nq,
        max_new_frames=infer_args.max_new_frames,
        voice_clone_max_text_tokens=infer_args.voice_clone_max_text_tokens,
        voice_clone_max_memory_per_sample_gb=infer_args.voice_clone_max_memory_per_sample_gb,
        do_sample=bool(infer_args.do_sample),
        use_kv_cache=True,
        **sampling_kwargs,
    )

    ids = result["audio_token_ids"].detach().cpu()
    out_codes = Path(args.out_codes)
    out_codes.parent.mkdir(parents=True, exist_ok=True)
    with out_codes.open("w", encoding="utf-8") as f:
        f.write(f"{ids.shape[0]}\n")
        for t in range(int(ids.shape[0])):
            f.write(" ".join(str(int(ids[t, q])) for q in range(int(ids.shape[1]))))
            f.write("\n")

    if first["seen"]:
        print(
            "[py_dbg] frame=0 text_head "
            f"assistant_id={first['assistant_id']} "
            f"end_id={first['end_id']} "
            f"assistant_logit={first['assistant_logit']:.9g} "
            f"end_logit={first['end_logit']:.9g} "
            f"picked={first['picked']} "
            f"do_sample={first['do_sample']} "
            f"text_temperature={first['text_temperature']:.6g}"
        )
        print(
            "[py_dbg] frame=0 text_head_top5 "
            + " ".join(
                f"top{idx+1}_id={first['text_top_ids'][idx]} top{idx+1}_logit={first['text_top_logits'][idx]:.9g}"
                for idx in range(5)
            )
        )
        print(
            "[py_dbg] frame=0 text_head_selected "
            + " ".join(
                f"id={first['text_sel_ids'][idx]} logit={first['text_sel_logits'][idx]:.9g}"
                for idx in range(len(first["text_sel_ids"]))
            )
        )
    if "local_text_step_stats" in dbg:
        s = dbg["local_text_step_stats"]
        print(
            "[py_dbg] frame=0 local_hidden_text_step "
            f"sum={s['sum']:.9g} l2={s['l2']:.9g} min={s['min']:.9g} max={s['max']:.9g}"
        )
    if "audio_ch0_top5" in dbg:
        a = dbg["audio_ch0_top5"]
        print(
            "[py_dbg] frame=0 audio_head_ch0_top5 "
            + " ".join(
                f"top{idx+1}_id={a['ids'][idx]} top{idx+1}_logit={a['logits'][idx]:.9g}"
                for idx in range(5)
            )
        )
        print(
            "[py_dbg] frame=0 audio_head_ch0_selected "
            + " ".join(
                f"id={a['sel_ids'][idx]} logit={a['sel_logits'][idx]:.9g}"
                for idx in range(len(a["sel_ids"]))
            )
        )
    print(f"python_frames={ids.shape[0]} n_q={ids.shape[1]} dump={out_codes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
