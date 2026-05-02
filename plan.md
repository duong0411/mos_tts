# Plan: Infer MOSS-TTS-Nano bằng C/C++ (không Torch) — LLM + Audio tokenizer + BLAS

Tài liệu mô tả **pipeline inference thực tế** trong repo này: đọc weight Hugging Face–style (**safetensors**), chạy **global/local GPT‑2 stacks** và **decoder/encoder Moss Audio Tokenizer** hoàn toàn trong C/C++ + SentencePiece. **PyTorch/ONNX không dùng ở runtime.**

**BLAS** (OpenBLAS khi biên dịch với `-DMOSS_USE_CBLAS`) chỉ đóng vai trò **tăng tốc nhân ma trận / dot** (GEMM/GEMV/SDOT); **toàn bộ kiến trúc layer** (attention, softmax, RoPE, GELU, patch, nucleus sampling…) do code tự viết.

---

## 1. Hai khối weight (thư mục `--model-dir`, ví dụ `weight/`)

| Khối | Đường dẫn | File chính | Vai trò |
|------|-----------|------------|---------|
| **LLM (TTS causal)** | `checkpoint/` hoặc gốc `model-dir` | `config.json`, `tokenizer.model`, `pytorch_model.safetensors` | Nhận joint grid `[T, 1+n_vq]`, sinh token text (assistant/end) + 16 RVQ/frame. |
| **Audio tokenizer / codec** | `audio_tokenizer/` | `config.json`, `model-00001-of-00001.safetensors` | Encode WAV→codes (prompt), decode codes→PCM (sau infer). |

Cả hai được **mmap** qua reader safetensors tối thiểu (`safetensors.c`), không deserialize PyTorch.

---

## 2. Source map (C/C++)

| Module | File | Chức năng |
|--------|------|-----------|
| Config | `moss_config.c` | Parse `gpt2_config` + id token audio/text từ JSON. |
| LLM weights | `moss_weights.c` | Gắn pointer BF16 vào các tensor `transformer.*`, `local_transformer.*`, `audio_embeddings.*`, `audio_lm_heads.*`, `text_lm_head`, v.v. |
| **GPT forward** | **`moss_gpt2.c`** | Một stack: LN1 → `c_attn` (QKV fused) → **RoPE** → attention nhân quả → `c_proj` → residual → LN2 → MLP (GELU) → residual → `ln_f`. |
| **Kernel** | **`moss_kernel.c`** | BF16→F32, **GEMV** (`moss_gemv_bf16_nt*`) — có nhánh **`cblas_sdot`** / mở rộng khi có CBLAS; LayerNorm, GELU, softmax hàng, RoPE. |
| **Audio codec** | **`moss_audio_tok.c`** | QuantizerResidual (weight-norm mats), encoder/decoder **transformer modules** trong f32 (`transformer_module_tm`: LN, QKV linear, causal window attention + RoPE, FFN…), reshape patch; chỗ dense dùng **SGEMM** (`moss_at_batch_mm_nt`) nếu `MOSS_USE_CBLAS`. |
| Prompt / joint | `moss_sp_prompt.cc`, `moss_tts.c` | SentencePiece + template hoặc **voice_clone** sections; dựng `joint`, vòng sinh frame (global → local → sample). |
| CLI | `main.c` | `moss_tts_load`, `moss_tts_generate_codes`, `moss_audio_tok_decode_codes`, ghi WAV. |

SentencePiece **C++**: `moss_sp_prompt.cc` (+ link `-lsentencepiece`).

---

## 3. Luồng inference (đồng bộ ý `modeling_moss_tts_nano._iter_generation_events`)

1. **Load**  
   - `moss_config_load(model_dir)`  
   - `moss_weights_load` → mmap `checkpoint/pytorch_model.safetensors` (fallback đường dẫn trong code).  
   - `moss_audio_tok_load` → mmap `audio_tokenizer/*.safetensors`.

2. **Chuẩn bị joint**  
   - Không có reference audio: `moss_build_prompt_token_ids` + hàng `audio_start` (+ pad các cột VQ).  
   - Voice clone: `moss_build_voice_clone_sections` + hàng prompt audio (`audio_user_slot` + codes từ `moss_audio_tok_encode_wav_file`).  

3. **Vòng sinh frame** (`moss_tts_generate_codes`)  
   - **Global:** `moss_build_input_embeds` (text WTE / audio_emb theo slot) → `moss_gpt2_forward(&wb.global, …)`.  
   - **Local:** Hidden bước cuối toàn cục → chèn WTE của token text đã sample → lặp `n_vq` lần: `moss_gpt2_forward(&wb.local, …)` → logits `audio_head[k]` → sampling (temperature / top-k / top-p / repetition penalty như HF).  
   - Text điều khiển: **assistant slot vs audio_end** (head `text_lm_head` hai lớp ứng viên hoặc greedy).  
   - Append một hàng joint mới `[assistant_slot | 16 code]`, `S++`.  

4. **Dừng**  
   Theo `audio_end` (tuỳ `min_frames`, `fill_to_max`) hoặc `max_new_frames`.

5. **Decode âm thanh**  
   - `moss_audio_tok_decode_codes` đọc `quantizer.*` + chuỗi `decoder.*`/`encoder.*` trong weight codec → PCM float stereo interleaved → `moss_write_wav16`.

**KV cache:** bản C hiện **forward lại global với chiều dài đầy đủ `S`** mỗi frame (đúng toán học causal, chi phí \(O(S^2)\) mỗi bước); đây là trade-off đơn giản, không đổi weight.

---

## 4. BLAS dùng ở đâu — “chỉ hỗ trợ nhân ma trận”

| Vị trí | Không có CBLAS | Có `MOSS_USE_CBLAS` |
|--------|----------------|---------------------|
| `moss_kernel.c` GEMV BF16→F32 | Vòng `for` thuần | `cblas_sdot` trên hàng đã promote F32 |
| `moss_audio_tok.c` linear theo batch thời gian | Fallback matvec tay | **`cblas_sgemm`** (RowMajor NT) trong `moss_at_batch_mm_nt` |

Attention (softmax trên cửa sổ, tích có trọng số V), RoPE, GELU layer-norm trong stack GPT và trong audio tokenizer — **đều tự viết**, không có “attention BLAS”.

---

## 5. Chuẩn hóa văn bản vs `infer.py`

`infer.py` chạy `prepare_tts_request_texts` + WeText; C++ chỉ có SentencePiece trên **`--text`**. Để trùng chuỗi token như infer: dùng `tools/prepare_infer_text_for_cpp.py` và **`--text-file`** (đã hỗ trợ trong `main.c`).

---

## 6. Biên dịch & chạy tham khảo (WSL / Linux)

```bash
cd /path/to/mos_tts
make clean && make -j4   # Makefile: có thể kèm -DMOSS_USE_CBLAS -lopenblas

./moss_tts \
  --model-dir weight \
  --text-file normalized.txt \
  --out out.wav \
  --frames 128 \
  --do-sample 1 \
  --seed 0
```

Điều kiện runtime: **`weight/audio_tokenizer/`** và **`weight/checkpoint/pytorch_model.safetensors`** đầy đủ (main có thể bắt buộc audio tokenizer mmap OK trước khi infer).

---

## 7. Hướng mở rộng / parity

- KV cache incremental cho global (giảm độ phức tạp độ dài joint).  
- SIMD riêng (NEON/AVX) cho GEMV BF16 và attention (theo kiểu `qwen3-tts`).  
- So golden: dump logits/codes một bước so với Python (`tools/*.py`).  
- Nếu Hub đổi tên shard audio (`model-xxxx-of-yyyy`): cập nhật `moss_audio_tok_load`.

---

*Tài liệu phản ánh layout hiện tại trong repo workspace; không dựa Torch cho forward; BLAS chỉ là tầng nhân.*
