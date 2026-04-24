# Plan: Infer MOSS-TTS-Nano bằng C/C++ + SIMD + Safetensors

Tài liệu này bám **thư mục weights / code tham chiếu** tại  
`/media/hdd1/duongpv/VibeVoice/MOSS-TTS-Nano/weight/`  
và binary / thư viện infer trong **`../cpp/`** (cùng repo).

Checkpoint gốc trên Hub: [OpenMOSS-Team/MOSS-TTS-Nano-100M](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M/tree/main).

---

## 1. Nội dung folder `weight/` (đã rà soát)

| File | Vai trò cho infer C/C++ |
|------|-------------------------|
| `config.json` | Nguồn sự thật: `gpt2_config` (global), `n_vq`, `vocab_size`, `audio_*_token_id`, `im_start`/`im_end`, `pad_token_id`, `audio_tokenizer_*`, `local_transformer_layers`, … |
| `tokenizer.model` | SentencePiece — bắt buộc để tokenize text giống Python (`MossTTSNanoSentencePieceTokenizer`). |
| `tokenizer_config.json`, `special_tokens_map.json` | Metadata tokenizer / special tokens. |
| `prompting.py` | Ghép prompt: `build_prompt_token_ids`, prefix user/assistant, template `<user_inst>`… |
| `modeling_moss_tts_nano.py` | Forward + **`generate` / `_iter_generation_events`**: luồng global→local→sample→append row. |
| `gpt2_decoder.py` | Kiến trúc block GPT-2 (RoPE, attn, MLP) dùng cho `transformer` và `local_transformer`. |
| `configuration_moss_tts_nano.py` | Class config Python (đồng bộ với `config.json`). |
| `pytorch_model.safetensors` (hoặc `.bin`) | **Phải có** trong `weight/` cùng `config.json` — toàn bộ trọng số TTS (BF16 trên Hub). |

**Lưu ý:** Audio codec **không** nằm trong file trên; `config.json` trỏ tới  
`audio_tokenizer_pretrained_name_or_path`: **MOSS-Audio-Tokenizer-Nano** (tải riêng / ONNX riêng) để `codes → waveform` (sample rate 48000 theo config).

---

## 2. Spec model cần khóa (từ `config.json`)

- **Global transformer** (`transformer.*`): GPT-2 style, `n_layer=12`, `n_head=12`, `n_embd=768`, `n_inner=3072`, `activation_function=gelu_new`, `position_embedding_type=rope`, `rope_base=10000`, `vocab_size=16384`.
- **Hàng joint** `input_ids`: shape `[B, T, n_vq + 1]` với `n_vq = 16` → **17 cột**; cột 0 = text; 16 cột sau = audio code hoặc pad **`audio_pad_token_id = 1024`**.
- **Local transformer** (`local_transformer.*`): `local_transformer_layers = 1`; context theo chiều dài local = `1 + n_vq` bước embed trong một vòng decode frame (xem `_iter_generation_events` trong `modeling_moss_tts_nano.py`).
- **Token điều khiển generate** (ví dụ): `audio_assistant_slot_token_id`, `audio_end_token_id`, `im_start_token_id`, `im_end_token_id`, `pad_token_id` — đọc đúng số trong `config.json`, không hard-code sai.

---

## 3. Map tensor Safetensors → runtime C

Weights đặt tên theo PyTorch state dict (đã audit trong `moss_weight_audit.json` ở `cpp/`):

- `transformer.wte.weight` — `[vocab_size, n_embd]`
- `transformer.h.{i}.*` — attn `c_attn` / `c_proj`, `ln_1` / `ln_2`, mlp `fc_in` / `fc_out`
- `transformer.ln_f.weight` / `transformer.ln_f.bias`
- `text_lm_head.weight` — tied với `wte` trong PyTorch; file safetensors có thể là bản copy (OK).
- `audio_embeddings.{k}.weight`, `audio_lm_heads.{k}.weight` — mỗi kênh `k ∈ [0,15]`, codebook 1024.
- `local_transformer.h.0.*`, `local_transformer.ln_f.*`

**SIMD:** tập trung vào các op lặp nhiều — matvec/GEMM (QKV, proj, MLP), RoPE, softmax attention, GELU. Chuẩn bị **multi-backend**: `generic` (scalar), **NEON** (ARM64), **AVX2+FMA** (x64), runtime chọn theo CPU (giống hướng `qwen3-tts`).

---

## 4. Luồng infer đúng (phải khớp Python)

Tham chiếu trực tiếp `MossTTSNanoForCausalLM._iter_generation_events`:

1. **Tokenize** text user + **build prompt** (`build_prompt_token_ids` trong `prompting.py`) → chuỗi token text.
2. **Ghép tensor** `[1, T, 17]`: các cột audio = pad cho đến khi bắt đầu sinh frame.
3. **Vòng theo frame** (tối đa `max_new_frames`):
   - `_build_inputs_embeds` → forward **global** `transformer` với **KV cache** (chỉ feed thêm hàng mới mỗi bước nếu `use_kv_cache`).
   - Lấy hidden **bước thời gian cuối** → đưa vào **local** (chuỗi embed dài dần trong frame).
   - Sample **một** token text assistant (`text_lm_head`); kiểm tra continue vs `audio_end_token_id` / slot.
   - Lần lượt **16** bước: mỗi bước sample token audio kênh `k` từ `audio_lm_heads[k]`, nhét embed kênh `k` vào local cho bước sau (repetition penalty / top-k / top-p giống Python nếu cần parity).
   - Ghép **hàng mới** `[1, 1, 17]`, append vào `input_ids`, cập nhật `attention_mask`.
4. **Dừng** khi hết continue hoặc đủ frame.
5. **Codec:** tensor `[B, F, 16]` → waveform (module riêng, weights MOSS-Audio-Tokenizer-Nano).

---

## 5. Lộ trình triển khai (theo phase)

### Phase 0 — Chuẩn parity

- Script Python nhỏ: cùng `weight/`, cố định `seed`, dump ra file:
  - `input_ids` sau prompt (shape, dtype int32),
  - sau frame 0: logits / token text / 16 token audio,
  - (tuỳ chọn) hidden trung gian.
- Test C: so khớp từng phase với golden (float tolerance cho BF16).

### Phase 1 — Loader + config

- Đọc `config.json` (JSON tối thiểu hoặc thư viện nhẹ) → struct `moss_model_spec_t`.
- mmap `pytorch_model.safetensors` (đã có `safetensors.c` trong `cpp/`).

### Phase 2 — Tokenizer

- Link **SentencePiece** C++ (`#include <sentencepiece_processor.h>`) hoặc port tối thiểu: load `tokenizer.model`, `Encode` giống `encode_text` trong `prompting.py`.

### Phase 3 — Prompt builder

- Port logic `build_prompt_prefix` / `build_prompt_suffix` / `build_prompt_token_ids` từ `prompting.py` + ID từ `config.json`.

### Phase 4 — Global transformer

- 12 layer: LN → QKV (fused `c_attn`) → RoPE → softmax attn → `c_proj` → residual → LN → MLP `fc_in` → GELU → `fc_out` → residual.
- KV cache: lưu K/V theo layout HF, append theo độ dài chuỗi.
- **SIMD:** kernel matvec BF16→F32 + AVX2/NEON cho phần nóng.

### Phase 5 — Local transformer + heads + sampling

- 1 layer, forward lặp theo độ dài local trong mỗi frame (theo code Python).
- `text_lm_head` + `audio_lm_heads[k]`; greedy trước, sau đó top-k/top-p/rep penalty như `_sample_next_token`.

### Phase 6 — Codec + CLI

- WAV 48 kHz: tích hợp decoder MOSS-Audio-Tokenizer-Nano (ONNX Runtime C, hoặc subprocess tạm thời).
- CLI `moss_tts`: `--model-dir ../weight`, `--text`, `--out` / `--output`, `--backend`, `--frames`, tham số sample.

### Phase 7 — Tối ưu & bench

- So sánh `generic` vs SIMD (sai số chấp nhận được).
- `bench.sh`: RTF / frame/s.

---

## 6. Trạng thái code hiện tại trong `cpp/`

- `moss_tts.c` hiện tại là **stub**: không chạy full `transformer` / `local_transformer`, không tokenizer/prompt đúng, codec là sóng giả — **không thể** trùng output PyTorch cho tới khi xong Phase 2–6.

---

## 7. Rủi ro & giảm thiểu

| Rủi ro | Cách xử lý |
|--------|------------|
| Sai tên / layout tensor | Dùng `moss_weight_audit.json`, assert shape khi load. |
| BF16 vs F32 | Chuẩn hoá nội bộ F32 activations, weight BF16 từ mmap. |
| FlashAttention trong config | Implement **eager** / math attention tương đương (sdpa thường). |
| Codec ngoài repo | Tách interface `moss_codec_decode`; phase 1 chỉ xuất `.codes` raw. |

---

## 8. Lệnh build / chạy tham chiếu

```bash
cd /media/hdd1/duongpv/VibeVoice/MOSS-TTS-Nano/cpp
make
./moss_tts --model-dir ../weight --text "..." --out out.wav --backend auto
```

Sau khi implement đúng pipeline: đổi `../weight` trỏ tới bất kỳ bản snapshot nào có đủ `config.json` + `tokenizer.model` + `pytorch_model.safetensors`.

---

*Tài liệu được sinh để đồng bộ với nội dung folder `weight/` tại thời điểm viết; khi config hoặc tên tensor trên Hub thay đổi, cần cập nhật Phase 1 và bảng map tensor.*
