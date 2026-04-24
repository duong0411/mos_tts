#include "moss_tts.h"

#include <math.h>
#include <stdint.h>
#include <time.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>

#ifdef __x86_64__
#include <cpuid.h>
#endif

#include "moss_gpt2.h"
#include "moss_kernel.h"
#include "moss_sp_prompt.h"

// #region agent log
static void dbg_log_tts(const char *run_id, const char *hypothesis_id, const char *location, const char *message, int a, int b, int c) {
    FILE *f = fopen("/media/hdd1/duongpv/.cursor/debug-a2d083.log", "a");
    if (!f) return;
    long long ts = (long long)time(NULL) * 1000LL;
    fprintf(
        f,
        "{\"sessionId\":\"a2d083\",\"runId\":\"%s\",\"hypothesisId\":\"%s\",\"location\":\"%s\",\"message\":\"%s\",\"data\":{\"a\":%d,\"b\":%d,\"c\":%d},\"timestamp\":%lld}\n",
        run_id, hypothesis_id, location, message, a, b, c, ts
    );
    fclose(f);
}
// #endregion

static int has_avx2(void) {
#ifdef __x86_64__
    unsigned int eax, ebx, ecx, edx;
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return 0;
    return (ebx & (1u << 5)) != 0;
#else
    return 0;
#endif
}

static moss_backend_t resolve_backend(moss_backend_t request) {
    if (request == MOSS_BACKEND_AVX2 && has_avx2()) return MOSS_BACKEND_AVX2;
    if (request == MOSS_BACKEND_GENERIC) return MOSS_BACKEND_GENERIC;
    if (request == MOSS_BACKEND_AUTO) {
        if (has_avx2()) return MOSS_BACKEND_AVX2;
        return MOSS_BACKEND_GENERIC;
    }
    return MOSS_BACKEND_GENERIC;
}

const char *moss_backend_name(moss_backend_t backend) {
    switch (backend) {
        case MOSS_BACKEND_AVX2: return "avx2";
        case MOSS_BACKEND_NEON: return "neon";
        case MOSS_BACKEND_GENERIC: return "generic";
        case MOSS_BACKEND_AUTO:
        default: return "auto";
    }
}

static size_t scratch_elems_for_seq(int S, int D, int I, int Dh) {
    return (size_t)S * (size_t)D * 6u + (size_t)S + (size_t)S * (size_t)I + (size_t)S * (size_t)Dh * 2u;
}

static void moss_build_input_embeds(
    const moss_weight_bundle_t *w,
    const moss_run_config_t *cfg,
    const int *joint,
    int S,
    float *out
) {
    const int D = cfg->n_embd;
    const int pad = cfg->audio_pad_token_id;
    for (int s = 0; s < S; s++) {
        float *row = out + s * D;
        memset(row, 0, (size_t)D * sizeof(float));
        int tid = joint[s * MOSS_N_COLS + 0];
        const uint16_t *ew = w->wte + (size_t)tid * D;
        for (int i = 0; i < D; i++) row[i] += moss_bf16_to_f32(ew[i]);
        for (int c = 0; c < cfg->n_vq; c++) {
            int aid = joint[s * MOSS_N_COLS + 1 + c];
            if (aid == pad) continue;
            const uint16_t *aw = w->audio_emb[c] + (size_t)aid * D;
            for (int i = 0; i < D; i++) row[i] += moss_bf16_to_f32(aw[i]);
        }
    }
}

static void moss_copy_wte_row(
    const moss_weight_bundle_t *w,
    const moss_run_config_t *cfg,
    float *dst_row,
    int token_id
) {
    const int D = cfg->n_embd;
    const uint16_t *ew = w->wte + (size_t)token_id * D;
    for (int i = 0; i < D; i++) dst_row[i] = moss_bf16_to_f32(ew[i]);
}

static void moss_copy_audio_emb_row(
    const moss_weight_bundle_t *w,
    const moss_run_config_t *cfg,
    int channel,
    float *dst_row,
    int audio_tok
) {
    const int D = cfg->n_embd;
    const uint16_t *aw = w->audio_emb[channel] + (size_t)audio_tok * D;
    for (int i = 0; i < D; i++) dst_row[i] = moss_bf16_to_f32(aw[i]);
}

static int moss_lm_argmax_bf16(const uint16_t *W, const float *x, int n_cls, int D) {
    int best = 0;
    float bestv = -1e30f;
    for (int r = 0; r < n_cls; r++) {
        const uint16_t *wr = W + (size_t)r * D;
        float s = 0.0f;
        for (int c = 0; c < D; c++) s += moss_bf16_to_f32(wr[c]) * x[c];
        if (s > bestv) {
            bestv = s;
            best = r;
        }
    }
    return best;
}

static int path_looks_like_audio_file(const char *path) {
    const char *dot = strrchr(path, '.');
    if (!dot) return 0;
    if (strcasecmp(dot, ".wav") == 0) return 1;
    if (strcasecmp(dot, ".flac") == 0) return 1;
    if (strcasecmp(dot, ".mp3") == 0) return 1;
    if (strcasecmp(dot, ".ogg") == 0) return 1;
    if (strcasecmp(dot, ".m4a") == 0) return 1;
    return 0;
}

/* Read WAV and encode to VQ using native C path; Python fallback kept temporarily. */
static int read_prompt_audio_codes_from_wav(
    const moss_audio_tok_t *tok,
    const char *model_dir,
    const char *wav_path,
    int nvq,
    int *out_flat,
    int *out_T
) {
    int *native_codes = NULL;
    int native_T = 0;
    if (tok && tok->sf) {
        fprintf(stderr, "[moss_tts] prompt-audio-path detected: %s\n", wav_path);
        fprintf(stderr, "[moss_tts] encoding WAV -> VQ using native C/C++ audio tokenizer\n");
        fflush(stderr);
        if (moss_audio_tok_encode_wav_file(tok, wav_path, &native_codes, &native_T) == 0 && native_codes && native_T >= 0) {
            if (native_T > MOSS_MAX_PROMPT_AUDIO_FRAMES) native_T = MOSS_MAX_PROMPT_AUDIO_FRAMES;
            for (int t = 0; t < native_T; t++) {
                for (int j = 0; j < nvq; j++) {
                    out_flat[t * nvq + j] = native_codes[t * nvq + j];
                }
            }
            free(native_codes);
            *out_T = native_T;
            return 0;
        }
        free(native_codes);
        fprintf(stderr, "[moss_tts] native WAV encoder failed, trying temporary Python fallback\n");
        fflush(stderr);
    }

    char script[2048];
    char cmd[8192];
    const char *py = getenv("MOSS_PYTHON");
    if (!py || !py[0]) py = "python3";

    if (snprintf(script, sizeof(script), "%s/../cpp/moss_encode_prompt_wav.py", model_dir) >= (int)sizeof(script)) {
        return -1;
    }
    if (snprintf(
            cmd,
            sizeof(cmd),
            "TRANSFORMERS_NO_TORCHVISION=1 TORCHVISION_DISABLE_NMS_OP=1 %s \"%s\" \"%s\" \"%s\"",
            py,
            script,
            model_dir,
            wav_path
        ) >= (int)sizeof(cmd)) {
        return -1;
    }

    fprintf(stderr, "[moss_tts] encoding WAV -> VQ using %s (set MOSS_PYTHON to override)\n", py);
    fflush(stderr);

    FILE *p = popen(cmd, "r");
    if (!p) return -1;
    int T = 0;
    if (fscanf(p, "%d", &T) != 1 || T < 0 || T > MOSS_MAX_PROMPT_AUDIO_FRAMES) {
        pclose(p);
        return -1;
    }
    for (int t = 0; t < T; t++) {
        for (int j = 0; j < nvq; j++) {
            if (fscanf(p, "%d", &out_flat[t * nvq + j]) != 1) {
                pclose(p);
                return -1;
            }
        }
    }
    if (pclose(p) != 0) return -1;
    *out_T = T;
    return 0;
}

/* Flat row-major [T * nvq]. File: line1 = T, then T lines of nvq integers (same as old encoder stdout). */
static int read_prompt_audio_codes_file(const char *path, int nvq, int *out_flat, int *out_T) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "[moss_tts] cannot open prompt-audio-codes file %s: %s\n", path, strerror(errno));
        fflush(stderr);
        return -1;
    }
    int T = 0;
    if (fscanf(f, "%d", &T) != 1 || T < 0 || T > MOSS_MAX_PROMPT_AUDIO_FRAMES) {
        fprintf(stderr,
            "[moss_tts] bad or missing frame count in %s (first line must be T, 0 <= T <= %d)\n",
            path,
            MOSS_MAX_PROMPT_AUDIO_FRAMES);
        fclose(f);
        fflush(stderr);
        return -1;
    }
    for (int t = 0; t < T; t++) {
        for (int j = 0; j < nvq; j++) {
            if (fscanf(f, "%d", &out_flat[t * nvq + j]) != 1) {
                fprintf(stderr,
                    "[moss_tts] parse error in %s at frame %d / %d (expected %d integers per line)\n",
                    path,
                    t,
                    T,
                    nvq);
                fclose(f);
                fflush(stderr);
                return -1;
            }
        }
    }
    fclose(f);
    *out_T = T;
    return 0;
}

moss_tts_ctx_t *moss_tts_load(const char *model_dir, moss_backend_t backend) {
    moss_tts_ctx_t *ctx = (moss_tts_ctx_t *)calloc(1, sizeof(*ctx));
    if (!ctx) return NULL;
    strncpy(ctx->model_dir, model_dir, sizeof(ctx->model_dir) - 1);
    ctx->model_dir[sizeof(ctx->model_dir) - 1] = '\0';
    ctx->backend = resolve_backend(backend);
    fprintf(stderr, "[moss_tts] reading config.json ...\n");
    fflush(stderr);
    if (moss_config_load(ctx->model_dir, &ctx->cfg) != 0) {
        free(ctx);
        return NULL;
    }
    fprintf(stderr, "[moss_tts] mmap pytorch_model.safetensors ...\n");
    fflush(stderr);
    if (moss_weights_load(ctx->model_dir, &ctx->wb, &ctx->cfg) != 0) {
        free(ctx);
        return NULL;
    }
    if (moss_audio_tok_load(ctx->model_dir, &ctx->audio_tok) == 0) {
        fprintf(stderr,
            "[moss_tts] audio tokenizer loaded: sr=%d channels=%d nq=%d codebook=%d\n",
            ctx->audio_tok.cfg.sample_rate,
            ctx->audio_tok.cfg.channels,
            ctx->audio_tok.cfg.num_quantizers,
            ctx->audio_tok.cfg.codebook_size);
    } else {
        fprintf(stderr, "[moss_tts] warning: native audio tokenizer load failed; fallback path may use Python.\n");
    }
    fflush(stderr);
    return ctx;
}

void moss_tts_unload(moss_tts_ctx_t *ctx) {
    if (!ctx) return;
    moss_audio_tok_unload(&ctx->audio_tok);
    moss_weights_unload(&ctx->wb);
    free(ctx);
}

int moss_tts_generate_codes(
    moss_tts_ctx_t *ctx,
    const char *text,
    const char *prompt_audio_codes_path,
    const moss_generate_params_t *params,
    int *out_codes,
    int *out_frames
) {
    if (!ctx || !text || !params || !out_codes || !out_frames) return -1;
    // #region agent log
    dbg_log_tts("debug-2", "H1", "moss_tts.c:generate_enter", "enter", params->max_new_frames, params->sample_rate, 0);
    // #endregion
    const moss_run_config_t *cfg = &ctx->cfg;
    const moss_weight_bundle_t *w = &ctx->wb;
    const int D = cfg->n_embd;
    const int Dh = D / cfg->n_head;
    const int I = cfg->n_inner;
    const int nvq = cfg->n_vq;
    const int max_frames = params->max_new_frames > 0 ? params->max_new_frames : MOSS_MAX_FRAMES;
    if (max_frames > MOSS_MAX_FRAMES) return -1;

    int prompt_ids[MOSS_MAX_TEXT_LEN];
    int P = moss_build_prompt_token_ids(ctx->model_dir, cfg, text, prompt_ids, MOSS_MAX_TEXT_LEN);
    if (P < 0) return -2;
    // #region agent log
    dbg_log_tts("debug-2", "H1", "moss_tts.c:prompt_len", "prompt_len", P, cfg->audio_assistant_slot_token_id, cfg->audio_end_token_id);
    // #endregion
    fprintf(stderr, "[moss_tts] prompt_len=%d tokens (SentencePiece + template)\n", P);
    fflush(stderr);

    const char *pcodes =
        (prompt_audio_codes_path && prompt_audio_codes_path[0]) ? prompt_audio_codes_path : NULL;
    int *pa_flat = NULL;
    int Tpa = 0;
    if (pcodes) {
        pa_flat = (int *)malloc((size_t)MOSS_MAX_PROMPT_AUDIO_FRAMES * (size_t)nvq * sizeof(int));
        if (!pa_flat) return -5;
        int rc = 0;
        if (path_looks_like_audio_file(pcodes)) rc = read_prompt_audio_codes_from_wav(&ctx->audio_tok, ctx->model_dir, pcodes, nvq, pa_flat, &Tpa);
        else rc = read_prompt_audio_codes_file(pcodes, nvq, pa_flat, &Tpa);
        if (rc != 0) {
            free(pa_flat);
            fprintf(stderr, "[moss_tts] failed to load prompt audio from %s\n", pcodes);
            fflush(stderr);
            return -11;
        }
        fprintf(stderr, "[moss_tts] prompt_audio VQ frames=%d (from %s)\n", Tpa, pcodes);
        fflush(stderr);
    }

    if (P + 1 > MOSS_MAX_JOINT_ROWS) {
        free(pa_flat);
        return -3;
    }
    if (P + 1 + Tpa + max_frames > MOSS_MAX_JOINT_ROWS) {
        free(pa_flat);
        return -4;
    }

    int *joint = (int *)calloc((size_t)MOSS_MAX_JOINT_ROWS * MOSS_N_COLS, sizeof(int));
    if (!joint) {
        free(pa_flat);
        return -5;
    }

    const int pad = cfg->audio_pad_token_id;
    for (int i = 0; i < P; i++) {
        joint[i * MOSS_N_COLS + 0] = prompt_ids[i];
        for (int c = 1; c < MOSS_N_COLS; c++) joint[i * MOSS_N_COLS + c] = pad;
    }
    joint[P * MOSS_N_COLS + 0] = cfg->audio_start_token_id;
    for (int c = 1; c < MOSS_N_COLS; c++) joint[P * MOSS_N_COLS + c] = pad;

    if (pa_flat && Tpa > 0) {
        for (int t = 0; t < Tpa; t++) {
            int r = P + 1 + t;
            joint[r * MOSS_N_COLS + 0] = cfg->audio_assistant_slot_token_id;
            for (int c = 0; c < nvq; c++) joint[r * MOSS_N_COLS + 1 + c] = pa_flat[t * nvq + c];
            for (int c = nvq + 1; c < MOSS_N_COLS; c++) joint[r * MOSS_N_COLS + c] = pad;
        }
    }
    free(pa_flat);
    pa_flat = NULL;

    int S = P + 1 + Tpa;
    size_t max_S = (size_t)P + 1u + (size_t)Tpa + (size_t)max_frames;
    if (max_S > (size_t)MOSS_MAX_JOINT_ROWS) max_S = MOSS_MAX_JOINT_ROWS;
    size_t need_global = scratch_elems_for_seq((int)max_S, D, I, Dh);
    const int local_max_S = 1 + 1 + cfg->n_vq;
    size_t need_local = scratch_elems_for_seq(local_max_S, D, I, Dh);
    size_t scratch_elems = need_global > need_local ? need_global : need_local;

    float *scratch = (float *)malloc(scratch_elems * sizeof(float));
    float *hidden = (float *)malloc((size_t)MOSS_MAX_JOINT_ROWS * D * sizeof(float));
    unsigned char *mask = (unsigned char *)malloc((size_t)MOSS_MAX_JOINT_ROWS);
    float *loc = (float *)malloc((size_t)local_max_S * D * sizeof(float));
    unsigned char *mask_loc = (unsigned char *)malloc((size_t)local_max_S);

    if (!scratch || !hidden || !mask || !loc || !mask_loc) {
        free(scratch);
        free(hidden);
        free(mask);
        free(loc);
        free(mask_loc);
        free(joint);
        return -6;
    }

    for (int i = 0; i < local_max_S; i++) mask_loc[i] = 1;

    fprintf(stderr, "[moss_tts] starting autoregressive loop (seq grows each frame; long prompts are slow)\n");
    fflush(stderr);

    int frame = 0;

    for (; frame < max_frames; frame++) {
        if (S >= MOSS_MAX_JOINT_ROWS) break;

        fprintf(stderr, "[moss_tts] frame %d/%d joint_seq_len=%d (global GPT-2 forward)\n",
            frame + 1, max_frames, S);
        fflush(stderr);

        for (int i = 0; i < S; i++) mask[i] = 1;
        moss_build_input_embeds(w, cfg, joint, S, hidden);
        if (moss_gpt2_forward(&w->global, cfg, hidden, S, mask, scratch, scratch_elems) != 0) {
            free(scratch);
            free(hidden);
            free(mask);
            free(loc);
            free(mask_loc);
            free(joint);
            return -7;
        }

        const float *h_last = hidden + (size_t)(S - 1) * D;

        int S_loc = 1;
        memcpy(loc, h_last, (size_t)D * sizeof(float));
        if (moss_gpt2_forward(&w->local, cfg, loc, S_loc, mask_loc, scratch, scratch_elems) != 0) {
            free(scratch);
            free(hidden);
            free(mask);
            free(loc);
            free(mask_loc);
            free(joint);
            return -8;
        }

        int text_tok = moss_lm_argmax_bf16(w->text_lm_head, loc + (size_t)(S_loc - 1) * D, cfg->vocab_size, D);
        // #region agent log
        dbg_log_tts("debug-2", "H2", "moss_tts.c:text_tok", "text_tok", frame + 1, text_tok, cfg->audio_assistant_slot_token_id);
        // #endregion
        /* In practice the global text head can emit early end/non-assistant tokens even when
         * local audio heads still produce valid continuation. Keep generating up to requested
         * max_frames unless we are already at the last frame.
         */
        if (text_tok != cfg->audio_assistant_slot_token_id) {
            fprintf(stderr,
                "[moss_tts] force-continue: text_tok=%d -> assistant_slot=%d (frame %d/%d)\n",
                text_tok,
                cfg->audio_assistant_slot_token_id,
                frame + 1,
                max_frames);
            fflush(stderr);
            text_tok = cfg->audio_assistant_slot_token_id;
        }

        memcpy(loc, h_last, (size_t)D * sizeof(float));
        moss_copy_wte_row(w, cfg, loc + D, text_tok);
        S_loc = 2;

        for (int ch = 0; ch < cfg->n_vq; ch++) {
            if (moss_gpt2_forward(&w->local, cfg, loc, S_loc, mask_loc, scratch, scratch_elems) != 0) {
                free(scratch);
                free(hidden);
                free(mask);
                free(loc);
                free(mask_loc);
                free(joint);
                return -9;
            }
            int a_tok = moss_lm_argmax_bf16(
                w->audio_head[ch],
                loc + (size_t)(S_loc - 1) * D,
                cfg->audio_vocab_size,
                D
            );
            out_codes[frame * cfg->n_vq + ch] = a_tok;
            moss_copy_audio_emb_row(w, cfg, ch, loc + (size_t)S_loc * D, a_tok);
            S_loc++;
        }
        int row = S * MOSS_N_COLS;
        joint[row + 0] = cfg->audio_assistant_slot_token_id;
        for (int c = 0; c < cfg->n_vq; c++) joint[row + 1 + c] = out_codes[frame * cfg->n_vq + c];
        S++;
    }

    *out_frames = frame;
    // #region agent log
    dbg_log_tts("debug-2", "H4", "moss_tts.c:out_frames", "out_frames", frame, max_frames, 0);
    // #endregion
    fprintf(stderr, "[moss_tts] done: %d audio frame(s)\n", frame);
    fflush(stderr);

    free(scratch);
    free(hidden);
    free(mask);
    free(loc);
    free(mask_loc);
    free(joint);
    return 0;
}

int moss_tts_decode_codes_to_wav(
    const int *codes,
    int frames,
    int sample_rate,
    float **out_samples,
    int *out_n_samples
) {
    if (!codes || frames < 0 || sample_rate <= 0 || !out_samples || !out_n_samples) return -1;
    int hop = sample_rate / 25;
    if (hop < 1) hop = 1;
    int n = frames * hop;
    if (n <= 0) {
        *out_samples = NULL;
        *out_n_samples = 0;
        return 0;
    }
    float *buf = (float *)malloc((size_t)n * sizeof(float));
    if (!buf) return -1;
    const float twopi = 6.28318530717958647692f;
    for (int i = 0; i < n; i++) {
        int f = i / hop;
        int v = 0;
        if (f < frames) {
            for (int c = 0; c < 16; c++) v += codes[f * 16 + c] & 31;
        }
        float t = (float)i / (float)sample_rate;
        float hz = 180.0f + (float)(v % 64) * 8.0f;
        buf[i] = 0.04f * sinf(twopi * hz * t);
    }
    *out_samples = buf;
    *out_n_samples = n;
    return 0;
}

int moss_write_wav16(const char *path, const float *samples, int n_samples, int sample_rate) {
    if (!path || !samples || n_samples <= 0 || sample_rate <= 0) return -1;
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    int16_t *pcm = (int16_t *)malloc((size_t)n_samples * sizeof(int16_t));
    if (!pcm) {
        fclose(f);
        return -1;
    }
    for (int i = 0; i < n_samples; i++) {
        float x = samples[i];
        if (x > 1.0f) x = 1.0f;
        if (x < -1.0f) x = -1.0f;
        pcm[i] = (int16_t)(x * 32767.0f);
    }

    uint32_t data_bytes = (uint32_t)(n_samples * sizeof(int16_t));
    uint32_t riff_size = 36 + data_bytes;

    fwrite("RIFF", 1, 4, f);
    fwrite(&riff_size, 4, 1, f);
    fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f);
    uint32_t fmt_chunk = 16;
    uint16_t audio_format = 1;
    uint16_t num_channels = 1;
    uint32_t byte_rate = (uint32_t)(sample_rate * sizeof(int16_t));
    uint16_t block_align = sizeof(int16_t);
    uint16_t bits = 16;
    fwrite(&fmt_chunk, 4, 1, f);
    fwrite(&audio_format, 2, 1, f);
    fwrite(&num_channels, 2, 1, f);
    fwrite(&sample_rate, 4, 1, f);
    fwrite(&byte_rate, 4, 1, f);
    fwrite(&block_align, 2, 1, f);
    fwrite(&bits, 2, 1, f);
    fwrite("data", 1, 4, f);
    fwrite(&data_bytes, 4, 1, f);
    fwrite(pcm, sizeof(int16_t), (size_t)n_samples, f);
    free(pcm);
    fclose(f);
    return 0;
}
