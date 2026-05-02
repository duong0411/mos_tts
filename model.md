# MOSS-TTS-Nano full layout (updated from new `weight/`)

Tai lieu nay mo ta cau truc model hien tai sau khi ban download lai:

- `weight/checkpoint/` -> TTS Nano 100M (global/local transformer)
- `weight/audio_tokenizer/` -> MOSS Audio Tokenizer (model rieng cho WAV <-> codes)

Code C/C++ infer da duoc cap nhat de nhan `--model-dir` tro thang vao thu muc `weight/` nay.

## 1) Cau truc folder

```
weight/
  checkpoint/
    config.json
    tokenizer.model
    pytorch_model.safetensors
    ...
  audio_tokenizer/
    config.json
    model.safetensors.index.json
    modeling_moss_audio_tokenizer.py
    ...
```

Luu y:
- TTS safetensors nam trong `checkpoint/pytorch_model.safetensors`
- Audio tokenizer la model rieng, duoc tham chieu boi `checkpoint/config.json`.

## 2) TTS model (checkpoint)

Tu `checkpoint/config.json`:
- `n_embd = 768`
- `n_head = 12`
- `n_layer = 12`
- `n_inner = 3072`
- `n_vq = 16`
- `audio_vocab_size = 1024`
- `vocab_size = 16384`
- `local_transformer_layers = 1`
- RoPE `rope_base = 10000`

Token control:
- `pad_token_id = 3`
- `im_start_token_id = 4`
- `im_end_token_id = 5`
- `audio_start_token_id = 6`
- `audio_end_token_id = 7`
- `audio_user_slot_token_id = 8`
- `audio_assistant_slot_token_id = 9`

Joint row width:
- `n_vq + 1 = 17`

Tensor map (TTS):
- `transformer.*` (global stack)
- `local_transformer.*` (local stack)
- `transformer.wte.weight`
- `text_lm_head.weight`
- `audio_embeddings.0..15.weight`
- `audio_lm_heads.0..15.weight`

## 3) Audio tokenizer model

Tu `audio_tokenizer/config.json`:
- model type: `moss-audio-tokenizer`
- sample rate: `48000`
- channels: `2` (`number_channels`)
- downsample rate: `3840`
- quantizer: `num_quantizers = 16`, `codebook_size = 1024`

Index `model.safetensors.index.json` cho thay shards expected:
- `model-00001-of-00001.safetensors`

Neu file shard nay chua co trong `audio_tokenizer/`, phan encode WAV se khong chay duoc.

## 4) Luong infer C/C++ (hien tai)

1. Prompt text:
   - SentencePiece + template (C++): `moss_sp_prompt.cc`
2. TTS inference:
   - global transformer -> local transformer -> text/audio heads
   - kernel tinh toan trong `moss_kernel.c` (SIMD/BLAS path)
3. Prompt audio:
   - `--prompt-audio-codes <txt>` (codes da co san)
   - hoac script helper Python `moss_encode_prompt_wav.py` de doi WAV -> codes

## 5) Cap nhat code da lam cho layout moi

- `moss_config.c`:
  - thu `model_dir/config.json`, neu khong co se fallback `model_dir/checkpoint/config.json`.
- `moss_weights.c`:
  - thu `model_dir/pytorch_model.safetensors`, neu khong co fallback `model_dir/checkpoint/pytorch_model.safetensors`.
- `moss_sp_prompt.cc`:
  - thu tokenizer tai `model_dir/tokenizer.model`, fallback `model_dir/checkpoint/tokenizer.model`.
- `moss_encode_prompt_wav.py`:
  - doc config tu root hoac `checkpoint/`
  - uu tien tokenizer local tai `model_dir/audio_tokenizer`.

## 6) Kernel hotspots (SIMD + BLAS)

Trong `moss_kernel.c`:
- GEMV BF16->F32: QKV / proj / MLP / heads
- vector ops: add/scale
- layernorm / gelu / softmax / rope

Neu co BLAS (`MOSS_USE_CBLAS`):
- `cblas_sdot`, `cblas_saxpy`, `cblas_sscal`

Fallback scalar van duoc giu cho tinh on dinh.
