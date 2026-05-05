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
#include <sys/stat.h>

#ifdef __x86_64__
#include <cpuid.h>
#endif

#include "moss_gpt2.h"
#include "moss_kernel.h"
#include "moss_sp_prompt.h"

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

static double moss_rng01(uint64_t *state);

static int cmp_float_desc(const void *a, const void *b) {
    const float fa = *(const float *)a;
    const float fb = *(const float *)b;
    return (fa < fb) - (fa > fb);
}

static int moss_sample_audio_token_bf16(
    const uint16_t *W,
    const float *x,
    int n_cls,
    int D,
    int do_sample,
    float temperature,
    int top_k,
    float top_p,
    float repetition_penalty,
    const int *history,
    int history_len,
    uint64_t *rng
) {
    int best = 0;
    float bestv = -1e30f;
    float *logits = (float *)malloc((size_t)n_cls * sizeof(float));
    if (!logits) return moss_lm_argmax_bf16(W, x, n_cls, D);

    for (int r = 0; r < n_cls; r++) {
        const uint16_t *wr = W + (size_t)r * D;
        float s = 0.0f;
        for (int c = 0; c < D; c++) s += moss_bf16_to_f32(wr[c]) * x[c];
        logits[r] = s;
        if (s > bestv) {
            bestv = s;
            best = r;
        }
    }

    /* Match HF moss_tts_nano._sample_next_token: repetition penalty applies before greedy argmax too. */
    if (repetition_penalty > 1.0f && history && history_len > 0) {
        unsigned char *seen = (unsigned char *)calloc((size_t)n_cls, 1);
        if (seen) {
            for (int i = 0; i < history_len; i++) {
                int id = history[i];
                if (id >= 0 && id < n_cls) seen[id] = 1;
            }
            for (int i = 0; i < n_cls; i++) {
                if (!seen[i]) continue;
                float v = logits[i];
                logits[i] = (v < 0.0f) ? (v * repetition_penalty) : (v / repetition_penalty);
            }
            free(seen);
        }
    }

    if (!do_sample) {
        int pick = best;
        float mx = logits[0];
        for (int r = 1; r < n_cls; r++) {
            if (logits[r] > mx) {
                mx = logits[r];
                pick = r;
            }
        }
        free(logits);
        return pick;
    }

    float T = temperature;
    if (T < 1e-6f) T = 1e-6f;
    for (int i = 0; i < n_cls; i++) logits[i] /= T;

    int k = top_k;
    if (k < 1 || k > n_cls) k = n_cls;
    float *sorted = (float *)malloc((size_t)n_cls * sizeof(float));
    if (!sorted) {
        free(logits);
        return best;
    }
    memcpy(sorted, logits, (size_t)n_cls * sizeof(float));
    qsort(sorted, (size_t)n_cls, sizeof(float), cmp_float_desc);
    float kth = sorted[k - 1];
    free(sorted);

    float m = -1e30f;
    for (int i = 0; i < n_cls; i++) {
        if (logits[i] < kth) continue;
        if (logits[i] > m) m = logits[i];
    }
    if (m <= -1e29f) {
        free(logits);
        return best;
    }

    float *probs = (float *)malloc((size_t)n_cls * sizeof(float));
    if (!probs) {
        free(logits);
        return best;
    }
    float sum = 0.0f;
    for (int i = 0; i < n_cls; i++) {
        if (logits[i] < kth) {
            probs[i] = 0.0f;
            continue;
        }
        probs[i] = expf(logits[i] - m);
        sum += probs[i];
    }
    if (sum <= 0.0f) {
        free(probs);
        free(logits);
        return best;
    }
    for (int i = 0; i < n_cls; i++) probs[i] /= sum;

    float p = top_p;
    if (p > 0.0f && p < 1.0f) {
        float *tmp = (float *)malloc((size_t)n_cls * sizeof(float));
        if (tmp) {
            memcpy(tmp, probs, (size_t)n_cls * sizeof(float));
            qsort(tmp, (size_t)n_cls, sizeof(float), cmp_float_desc);
            float c = 0.0f;
            float cut = tmp[n_cls - 1];
            for (int i = 0; i < n_cls; i++) {
                c += tmp[i];
                cut = tmp[i];
                if (c >= p) break;
            }
            free(tmp);
            float s2 = 0.0f;
            for (int i = 0; i < n_cls; i++) {
                if (probs[i] < cut) probs[i] = 0.0f;
                s2 += probs[i];
            }
            if (s2 > 0.0f) {
                for (int i = 0; i < n_cls; i++) probs[i] /= s2;
            }
        }
    }

    double u = moss_rng01(rng);
    double acc = 0.0;
    int picked = best;
    for (int i = 0; i < n_cls; i++) {
        acc += probs[i];
        if (u <= acc) {
            picked = i;
            break;
        }
    }
    free(probs);
    free(logits);
    return picked;
}

static uint64_t moss_rng_u64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 2685821657736338717ULL;
}

static double moss_rng01(uint64_t *state) {
    return (moss_rng_u64(state) >> 11) * (1.0 / 9007199254740992.0);
}

/* Sample index 0|1 given two scaled logits — same topology as moss_sample_audio_token_bf16 tail (top-k on logits, top-p on probs). */
static int moss_sample_two_logits_float(float la, float lb, int top_k, float top_p, uint64_t *rng, int fallback_idx) {
    float logits[2] = { la, lb };
    const int n_cls = 2;
    int k = top_k > 0 ? top_k : n_cls;
    if (k > n_cls) k = n_cls;
    float sorted_pair[2];
    sorted_pair[0] = la >= lb ? la : lb;
    sorted_pair[1] = la >= lb ? lb : la;
    float kth = sorted_pair[k - 1];

    float m = -1e30f;
    for (int i = 0; i < n_cls; i++) {
        if (logits[i] < kth) continue;
        if (logits[i] > m) m = logits[i];
    }
    if (m <= -1e29f) return fallback_idx;

    float probs[2];
    float sum = 0.0f;
    for (int i = 0; i < n_cls; i++) {
        if (logits[i] < kth) {
            probs[i] = 0.0f;
            continue;
        }
        probs[i] = expf(logits[i] - m);
        sum += probs[i];
    }
    if (sum <= 0.0f) return fallback_idx;
    probs[0] /= sum;
    probs[1] /= sum;

    if (top_p > 0.0f && top_p < 1.0f) {
        float tmp0 = probs[0], tmp1 = probs[1];
        float hi = tmp0 >= tmp1 ? tmp0 : tmp1;
        float lo = tmp0 >= tmp1 ? tmp1 : tmp0;
        float td[2] = { hi, lo };
        float c = 0.0f;
        float cut = td[n_cls - 1];
        for (int i = 0; i < n_cls; i++) {
            c += td[i];
            cut = td[i];
            if (c >= top_p) break;
        }
        float s2 = 0.0f;
        if (probs[0] < cut) probs[0] = 0.0f;
        if (probs[1] < cut) probs[1] = 0.0f;
        s2 = probs[0] + probs[1];
        if (s2 > 0.0f) {
            probs[0] /= s2;
            probs[1] /= s2;
        }
    }

    double u = moss_rng01(rng);
    double acc = probs[0];
    if (u <= acc) return 0;
    return 1;
}

/* Python _sample_next_assistant_text_token: logits restricted to assistant vs end only. */
static int moss_text_assistant_or_end_pick_bf16(
    const uint16_t *text_lm_head,
    const float *x,
    int D,
    int assistant_id,
    int end_id,
    int do_sample,
    float text_temperature,
    int text_top_k,
    float text_top_p,
    float end_token_logit_bias,
    uint64_t *rng,
    float *out_assistant_logit,
    float *out_end_logit
) {
    float s_ass = 0.0f, s_end = 0.0f;
    const uint16_t *wa = text_lm_head + (size_t)assistant_id * D;
    const uint16_t *we = text_lm_head + (size_t)end_id * D;
    for (int i = 0; i < D; i++) {
        float xi = x[i];
        s_ass += moss_bf16_to_f32(wa[i]) * xi;
        s_end += moss_bf16_to_f32(we[i]) * xi;
    }
    /* Optional runtime knob to encourage earlier stopping without changing model weights. */
    s_end += end_token_logit_bias;
    if (out_assistant_logit) *out_assistant_logit = s_ass;
    if (out_end_logit) *out_end_logit = s_end;
    if (!do_sample) return (s_ass >= s_end) ? assistant_id : end_id;
    float T = text_temperature;
    if (T < 1e-6f) T = 1e-6f;
    float la = s_ass / T;
    float lb = s_end / T;
    int pick = moss_sample_two_logits_float(la, lb, text_top_k, text_top_p, rng, (s_ass >= s_end) ? 0 : 1);
    return pick == 0 ? assistant_id : end_id;
}

static void moss_debug_log_vec_stats(const char *tag, const float *x, int D) {
    if (!tag || !x || D <= 0) return;
    double sum = 0.0;
    double sq = 0.0;
    float mn = x[0], mx = x[0];
    for (int i = 0; i < D; i++) {
        float v = x[i];
        sum += (double)v;
        sq += (double)v * (double)v;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    double l2 = sqrt(sq);
    fprintf(stderr, "[moss_dbg] %s sum=%.9g l2=%.9g min=%.9g max=%.9g\n", tag, sum, l2, (double)mn, (double)mx);
}

static void moss_debug_log_topk_bf16(
    const char *tag,
    const uint16_t *W,
    const float *x,
    int n_cls,
    int D,
    int k
) {
    if (!tag || !W || !x || n_cls <= 0 || D <= 0 || k <= 0) return;
    if (k > n_cls) k = n_cls;
    int *best_i = (int *)malloc((size_t)k * sizeof(int));
    float *best_v = (float *)malloc((size_t)k * sizeof(float));
    if (!best_i || !best_v) {
        free(best_i);
        free(best_v);
        return;
    }
    for (int i = 0; i < k; i++) {
        best_i[i] = -1;
        best_v[i] = -1e30f;
    }
    for (int r = 0; r < n_cls; r++) {
        const uint16_t *wr = W + (size_t)r * D;
        float s = 0.0f;
        for (int c = 0; c < D; c++) s += moss_bf16_to_f32(wr[c]) * x[c];
        int pos = -1;
        for (int j = 0; j < k; j++) {
            if (s > best_v[j]) {
                pos = j;
                break;
            }
        }
        if (pos >= 0) {
            for (int t = k - 1; t > pos; t--) {
                best_v[t] = best_v[t - 1];
                best_i[t] = best_i[t - 1];
            }
            best_v[pos] = s;
            best_i[pos] = r;
        }
    }
    fprintf(stderr, "[moss_dbg] %s", tag);
    for (int i = 0; i < k; i++) {
        fprintf(stderr, " top%d_id=%d top%d_logit=%.9g", i + 1, best_i[i], i + 1, (double)best_v[i]);
    }
    fputc('\n', stderr);
    free(best_i);
    free(best_v);
}

static void moss_debug_log_selected_logits_bf16(
    const char *tag,
    const uint16_t *W,
    const float *x,
    int n_cls,
    int D,
    const int *ids,
    int n_ids
) {
    if (!tag || !W || !x || !ids || n_ids <= 0) return;
    fprintf(stderr, "[moss_dbg] %s", tag);
    for (int i = 0; i < n_ids; i++) {
        int id = ids[i];
        if (id < 0 || id >= n_cls) {
            fprintf(stderr, " id=%d logit=NaN", id);
            continue;
        }
        const uint16_t *wr = W + (size_t)id * D;
        float s = 0.0f;
        for (int c = 0; c < D; c++) s += moss_bf16_to_f32(wr[c]) * x[c];
        fprintf(stderr, " id=%d logit=%.9g", id, (double)s);
    }
    fputc('\n', stderr);
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

static void moss_debug_dump_int_row(const char *tag, int row_idx, const int *row, int n_cols) {
    if (!tag || !row || n_cols <= 0) return;
    fprintf(stderr, "[moss_input] %s row=%d", tag, row_idx);
    for (int i = 0; i < n_cols; i++) fprintf(stderr, " %d", row[i]);
    fprintf(stderr, "\n");
}

/* WAV -> VQ: native moss_audio_tok only (no Python encoder). */
static int read_prompt_audio_codes_from_wav(
    const moss_audio_tok_t *tok,
    const char *model_dir,
    const char *wav_path,
    int nvq,
    int *out_flat,
    int *out_T
) {
    if (!tok || !tok->sf) {
        fprintf(stderr,
            "[moss_tts] --prompt-audio-path needs a loaded native audio tokenizer under %s/audio_tokenizer\n",
            model_dir && model_dir[0] ? model_dir : "(model-dir)");
        fflush(stderr);
        return -1;
    }

    fprintf(stderr, "[moss_tts] prompt-audio-path: %s\n", wav_path);
    fprintf(stderr, "[moss_tts] encoding WAV -> VQ (native C/C++ only)\n");
    fflush(stderr);

    int *native_codes = NULL;
    int native_T = 0;
    if (moss_audio_tok_encode_wav_file(tok, wav_path, &native_codes, &native_T) != 0 || !native_codes || native_T < 0) {
        free(native_codes);
        fprintf(stderr,
            "[moss_tts] native WAV->VQ encode failed (use PCM WAV readable by the decoder, or ffmpeg for other "
            "formats if built with decode support; check audio_tokenizer weights).\n");
        fflush(stderr);
        return -1;
    }

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
        fprintf(stderr,
            "[moss_tts] warning: native audio tokenizer load failed; WAV prompt encoding and decode need "
            "weight/audio_tokenizer (no Python encoder).\n");
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

static uint64_t moss_hash_seed_to_rng(unsigned long long seed) {
    uint64_t z = (uint64_t)seed;
    z ^= z >> 33;
    z *= 0xff51afd7ed558ccdULL;
    z ^= z >> 33;
    z *= 0xc4ceb9fe1a85ec53ULL;
    z ^= z >> 33;
    return z ? z : 1ull;
}

int moss_tts_generate_codes(
    moss_tts_ctx_t *ctx,
    const char *text,
    const char *prompt_audio_codes_path,
    moss_generate_params_t *params,
    int *out_codes,
    int *out_frames
) {
    if (!ctx || !text || !params || !out_codes || !out_frames) return -1;
    const moss_run_config_t *cfg = &ctx->cfg;
    const moss_weight_bundle_t *w = &ctx->wb;
    const int D = cfg->n_embd;
    const int Dh = D / cfg->n_head;
    const int I = cfg->n_inner;
    const int nvq = cfg->n_vq;
    const int max_frames =
        params->max_new_frames > 0 ? params->max_new_frames : MOSS_DEFAULT_MAX_NEW_FRAMES;
    int dbg_input_pipeline = 0;
    {
        const char *dbg = getenv("MOSS_DEBUG_INPUT_PIPELINE");
        if (dbg && dbg[0] && strcmp(dbg, "0") != 0) dbg_input_pipeline = 1;
    }
    /* min_frames: suppress audio_end sampling until this many frame steps; fill_to_max bumps to max_frames */
    int min_frames_eff = params->min_frames > 0 ? params->min_frames : 0;
    if (params->fill_to_max && max_frames > 0) min_frames_eff = max_frames;
    if (max_frames > MOSS_MAX_FRAMES) return -1;

    int prompt_ids[MOSS_MAX_TEXT_LEN];
    int prefix_ids[MOSS_MAX_TEXT_LEN];
    int suffix_ids[MOSS_MAX_TEXT_LEN];
    int P = 0;
    int P_pre = 0;
    int P_suf = 0;

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
        if (dbg_input_pipeline) {
            int tmax = Tpa < 3 ? Tpa : 3;
            for (int t = 0; t < tmax; t++) {
                fprintf(stderr, "[moss_input] prompt_audio_codes t=%d", t);
                for (int j = 0; j < nvq; j++) fprintf(stderr, " %d", pa_flat[t * nvq + j]);
                fprintf(stderr, "\n");
            }
            fflush(stderr);
        }
    }
    if (pcodes && Tpa > 0) {
        if (moss_build_voice_clone_sections(
                ctx->model_dir,
                cfg,
                text,
                prefix_ids,
                MOSS_MAX_TEXT_LEN,
                suffix_ids,
                MOSS_MAX_TEXT_LEN,
                &P_pre,
                &P_suf
            ) != 0) {
            free(pa_flat);
            return -2;
        }
        P = P_pre + P_suf;
        fprintf(stderr, "[moss_tts] prompt_len=%d tokens (SentencePiece + voice_clone text sections)\n", P);
        fflush(stderr);
        if (dbg_input_pipeline) {
            fprintf(stderr, "[moss_input] voice_clone_prefix_len=%d", P_pre);
            for (int i = 0; i < P_pre; i++) fprintf(stderr, " %d", prefix_ids[i]);
            fprintf(stderr, "\n");
            fprintf(stderr, "[moss_input] voice_clone_suffix_len=%d", P_suf);
            for (int i = 0; i < P_suf; i++) fprintf(stderr, " %d", suffix_ids[i]);
            fprintf(stderr, "\n");
            fflush(stderr);
        }
    } else {
        P = moss_build_prompt_token_ids(ctx->model_dir, cfg, text, prompt_ids, MOSS_MAX_TEXT_LEN);
        if (P < 0) {
            free(pa_flat);
            return -2;
        }
        fprintf(stderr, "[moss_tts] prompt_len=%d tokens (SentencePiece + template)\n", P);
        fflush(stderr);
        if (dbg_input_pipeline) {
            fprintf(stderr, "[moss_input] prompt_ids_len=%d", P);
            for (int i = 0; i < P; i++) fprintf(stderr, " %d", prompt_ids[i]);
            fprintf(stderr, "\n");
            fflush(stderr);
        }
    }

    if ((!pcodes || Tpa <= 0) && P + 1 > MOSS_MAX_JOINT_ROWS) {
        free(pa_flat);
        return -3;
    }
    if ((pcodes && Tpa > 0 && (P + Tpa + max_frames > MOSS_MAX_JOINT_ROWS)) ||
        ((!pcodes || Tpa <= 0) && (P + 1 + Tpa + max_frames > MOSS_MAX_JOINT_ROWS))) {
        free(pa_flat);
        return -4;
    }

    int *joint = (int *)calloc((size_t)MOSS_MAX_JOINT_ROWS * MOSS_N_COLS, sizeof(int));
    if (!joint) {
        free(pa_flat);
        return -5;
    }

    const int pad = cfg->audio_pad_token_id;
    int has_prompt_audio = (pa_flat && Tpa > 0) ? 1 : 0;
    if (has_prompt_audio) {
        int r = 0;
        for (int i = 0; i < P_pre; i++, r++) {
            joint[r * MOSS_N_COLS + 0] = prefix_ids[i];
            for (int c = 1; c < MOSS_N_COLS; c++) joint[r * MOSS_N_COLS + c] = pad;
        }
        for (int t = 0; t < Tpa; t++, r++) {
            joint[r * MOSS_N_COLS + 0] = cfg->audio_user_slot_token_id;
            for (int c = 0; c < nvq; c++) joint[r * MOSS_N_COLS + 1 + c] = pa_flat[t * nvq + c];
            for (int c = nvq + 1; c < MOSS_N_COLS; c++) joint[r * MOSS_N_COLS + c] = pad;
        }
        for (int i = 0; i < P_suf; i++, r++) {
            joint[r * MOSS_N_COLS + 0] = suffix_ids[i];
            for (int c = 1; c < MOSS_N_COLS; c++) joint[r * MOSS_N_COLS + c] = pad;
        }
    } else {
        for (int i = 0; i < P; i++) {
            joint[i * MOSS_N_COLS + 0] = prompt_ids[i];
            for (int c = 1; c < MOSS_N_COLS; c++) joint[i * MOSS_N_COLS + c] = pad;
        }
        joint[P * MOSS_N_COLS + 0] = cfg->audio_start_token_id;
        for (int c = 1; c < MOSS_N_COLS; c++) joint[P * MOSS_N_COLS + c] = pad;
    }
    free(pa_flat);
    pa_flat = NULL;

    int S = has_prompt_audio ? (P_pre + Tpa + P_suf) : (P + 1 + Tpa);
    if (dbg_input_pipeline) {
        fprintf(stderr, "[moss_input] initial_joint_rows=%d row_width=%d\n", S, MOSS_N_COLS);
        if (S > 0) moss_debug_dump_int_row("joint", 0, joint + 0 * MOSS_N_COLS, MOSS_N_COLS);
        if (S > 1) moss_debug_dump_int_row("joint", 1, joint + 1 * MOSS_N_COLS, MOSS_N_COLS);
        if (S > 2) moss_debug_dump_int_row("joint", 2, joint + 2 * MOSS_N_COLS, MOSS_N_COLS);
        if (S > 3) moss_debug_dump_int_row("joint", S - 3, joint + (S - 3) * MOSS_N_COLS, MOSS_N_COLS);
        if (S > 2) moss_debug_dump_int_row("joint", S - 2, joint + (S - 2) * MOSS_N_COLS, MOSS_N_COLS);
        if (S > 1) moss_debug_dump_int_row("joint", S - 1, joint + (S - 1) * MOSS_N_COLS, MOSS_N_COLS);
        fflush(stderr);
    }
    size_t base_S = (size_t)(has_prompt_audio ? (P_pre + Tpa + P_suf) : (P + 1 + Tpa));
    size_t max_S = base_S + (size_t)max_frames;
    if (max_S > (size_t)MOSS_MAX_JOINT_ROWS) max_S = MOSS_MAX_JOINT_ROWS;
    size_t need_global = scratch_elems_for_seq((int)max_S, D, I, Dh);
    const int local_max_S = 1 + 1 + cfg->n_vq;
    size_t need_local = scratch_elems_for_seq(local_max_S, D, I, Dh);
    size_t scratch_elems = need_global > need_local ? need_global : need_local;

    float *scratch = (float *)malloc(scratch_elems * sizeof(float));
    float *hidden = (float *)malloc((size_t)MOSS_MAX_JOINT_ROWS * D * sizeof(float));
    unsigned char *mask = (unsigned char *)malloc((size_t)MOSS_MAX_JOINT_ROWS);
    float *loc = (float *)malloc((size_t)local_max_S * D * sizeof(float));
    float *loc_run = (float *)malloc((size_t)local_max_S * D * sizeof(float));
    unsigned char *mask_loc = (unsigned char *)malloc((size_t)local_max_S);

    if (!scratch || !hidden || !mask || !loc || !loc_run || !mask_loc) {
        free(scratch);
        free(hidden);
        free(mask);
        free(loc);
        free(loc_run);
        free(mask_loc);
        free(joint);
        return -6;
    }

    for (int i = 0; i < local_max_S; i++) mask_loc[i] = 1;

    fprintf(stderr, "[moss_tts] starting autoregressive loop (seq grows each frame; long prompts are slow)\n");
    fflush(stderr);

    uint64_t rng;
    if (params->rng_state != 0ull) {
        rng = (uint64_t)params->rng_state;
    } else if (params->rng_seed != 0ull) {
        rng = moss_hash_seed_to_rng(params->rng_seed);
    } else {
        rng = (uint64_t)time(NULL);
        rng ^= (uint64_t)(uintptr_t)out_codes * 1315423911ULL;
        rng ^= (uint64_t)(uintptr_t)joint * 1181783497276652981ULL;
        if (rng == 0) rng = 1;
    }

    int frame = 0;
    int stopped_by_audio_end = 0;
    int debug_first_step = 0;
    {
        const char *dbg = getenv("MOSS_DEBUG_FIRST_STEP");
        if (dbg && dbg[0] && strcmp(dbg, "0") != 0) debug_first_step = 1;
    }
    int log_every_ar_frame = 0;
    {
        const char *vf = getenv("MOSS_VERBOSE_FRAMES");
        if (vf && vf[0] && strcmp(vf, "0") != 0) log_every_ar_frame = 1;
    }

    for (; frame < max_frames; frame++) {
        if (S >= MOSS_MAX_JOINT_ROWS) break;

        if (log_every_ar_frame || frame == 0 || ((frame + 1) % 25) == 0) {
            fprintf(stderr, "[moss_tts] frame %d/%d joint_seq_len=%d (global GPT-2 forward)\n",
                frame + 1, max_frames, S);
        }

        for (int i = 0; i < S; i++) mask[i] = 1;
        moss_build_input_embeds(w, cfg, joint, S, hidden);
        if (moss_gpt2_forward(&w->global, cfg, hidden, S, mask, scratch, scratch_elems, "global") != 0) {
            free(scratch);
            free(hidden);
            free(mask);
            free(loc);
            free(loc_run);
            free(mask_loc);
            free(joint);
            return -7;
        }

        const float *h_last = hidden + (size_t)(S - 1) * D;

        int S_loc = 1;
        memcpy(loc, h_last, (size_t)D * sizeof(float));
        if (moss_gpt2_forward(&w->local, cfg, loc, S_loc, mask_loc, scratch, scratch_elems, "local") != 0) {
            free(scratch);
            free(hidden);
            free(mask);
            free(loc);
            free(loc_run);
            free(mask_loc);
            free(joint);
            return -8;
        }

        float text_temp = params->text_temperature;
        if (text_temp <= 0.0f) text_temp = params->temperature > 0.0f ? params->temperature : 1.0f;
        if (debug_first_step && frame == 0) {
            const int text_dbg_ids[] = {9, 7, 505};
            moss_debug_log_vec_stats("frame=0 global_last_hidden", h_last, D);
            moss_debug_log_vec_stats("frame=0 local_hidden_text_step", loc + (size_t)(S_loc - 1) * D, D);
            moss_debug_log_topk_bf16(
                "frame=0 text_head_top5",
                w->text_lm_head,
                loc + (size_t)(S_loc - 1) * D,
                cfg->vocab_size,
                D,
                5
            );
            moss_debug_log_selected_logits_bf16(
                "frame=0 text_head_selected",
                w->text_lm_head,
                loc + (size_t)(S_loc - 1) * D,
                cfg->vocab_size,
                D,
                text_dbg_ids,
                (int)(sizeof(text_dbg_ids) / sizeof(text_dbg_ids[0]))
            );
        }
        float dbg_ass_logit = 0.0f;
        float dbg_end_logit = 0.0f;
        int text_tok = moss_text_assistant_or_end_pick_bf16(
            w->text_lm_head,
            loc + (size_t)(S_loc - 1) * D,
            D,
            cfg->audio_assistant_slot_token_id,
            cfg->audio_end_token_id,
            params->do_sample != 0,
            text_temp,
            params->text_top_k,
            params->text_top_p,
            params->end_token_logit_bias,
            &rng,
            (debug_first_step && frame == 0) ? &dbg_ass_logit : NULL,
            (debug_first_step && frame == 0) ? &dbg_end_logit : NULL
        );
        if (debug_first_step && frame == 0) {
            fprintf(stderr,
                "[moss_dbg] frame=0 text_head assistant_id=%d end_id=%d assistant_logit=%.9g end_logit=%.9g picked=%d do_sample=%d text_temperature=%.6g\n",
                cfg->audio_assistant_slot_token_id,
                cfg->audio_end_token_id,
                (double)dbg_ass_logit,
                (double)dbg_end_logit,
                text_tok,
                params->do_sample != 0,
                (double)text_temp);
            fflush(stderr);
        }
        /* Same stop as modeling_moss_tts_nano._iter_generation_events: end token ends this step with no new RVQ row. */
        if (text_tok == cfg->audio_end_token_id) {
            if (frame < min_frames_eff) {
                text_tok = cfg->audio_assistant_slot_token_id;
                fprintf(stderr,
                    "[moss_tts] ignore early end at frame %d (< min_frames=%d), continue\n",
                    frame + 1, min_frames_eff);
                fflush(stderr);
            } else {
                fprintf(stderr,
                    "[moss_tts] stop: text head chose audio_end_token_id=%d (1-based step %d, do_sample=%d)\n",
                    cfg->audio_end_token_id,
                    frame + 1,
                    params->do_sample != 0);
                fflush(stderr);
                stopped_by_audio_end = 1;
                break;
            }
        }

        memcpy(loc, h_last, (size_t)D * sizeof(float));
        moss_copy_wte_row(w, cfg, loc + D, text_tok);
        S_loc = 2;

        for (int ch = 0; ch < cfg->n_vq; ch++) {
            /* Keep local input embeddings immutable across channels, matching Python local_inputs_embeds behavior.
             * moss_gpt2_forward mutates `hidden` in-place, so run on a working copy. */
            memcpy(loc_run, loc, (size_t)S_loc * (size_t)D * sizeof(float));
            if (moss_gpt2_forward(&w->local, cfg, loc_run, S_loc, mask_loc, scratch, scratch_elems, "local") != 0) {
                free(scratch);
                free(hidden);
                free(mask);
                free(loc);
                free(loc_run);
                free(mask_loc);
                free(joint);
                return -9;
            }
            int *ch_hist = NULL;
            if (frame > 0) {
                ch_hist = (int *)malloc((size_t)frame * sizeof(int));
                if (ch_hist) {
                    for (int t = 0; t < frame; t++) ch_hist[t] = out_codes[t * cfg->n_vq + ch];
                }
            }
            int a_tok = moss_sample_audio_token_bf16(
                w->audio_head[ch],
                loc_run + (size_t)(S_loc - 1) * D,
                cfg->audio_vocab_size,
                D,
                params->do_sample != 0,
                params->audio_temperature,
                params->audio_top_k,
                params->audio_top_p,
                params->audio_repetition_penalty,
                ch_hist,
                frame,
                &rng
            );
            if (debug_first_step && frame == 0 && ch == 0) {
                const int audio_dbg_ids[] = {137, 199, 61, 64};
                moss_debug_log_vec_stats("frame=0 local_hidden_audio_ch0", loc + (size_t)(S_loc - 1) * D, D);
                moss_debug_log_topk_bf16(
                    "frame=0 audio_head_ch0_top5",
                    w->audio_head[ch],
                    loc_run + (size_t)(S_loc - 1) * D,
                    cfg->audio_vocab_size,
                    D,
                    5
                );
                moss_debug_log_selected_logits_bf16(
                    "frame=0 audio_head_ch0_selected",
                    w->audio_head[ch],
                    loc_run + (size_t)(S_loc - 1) * D,
                    cfg->audio_vocab_size,
                    D,
                    audio_dbg_ids,
                    (int)(sizeof(audio_dbg_ids) / sizeof(audio_dbg_ids[0]))
                );
            }
            free(ch_hist);
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
    params->rng_state = (unsigned long long)rng;
    if (stopped_by_audio_end) {
        fprintf(stderr, "[moss_tts] done: %d audio frame(s), stop_reason=audio_end\n", frame);
    } else if (frame >= max_frames) {
        fprintf(stderr, "[moss_tts] done: %d audio frame(s), stop_reason=max_new_frames_cap(%d)\n", frame, max_frames);
    } else {
        fprintf(stderr, "[moss_tts] done: %d audio frame(s), stop_reason=loop_exit\n", frame);
    }
    fflush(stderr);

    free(scratch);
    free(hidden);
    free(mask);
    free(loc);
    free(loc_run);
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

/* Create parent directories for a file path (mkdir -p style). Ignores if no '/' in path. */
static int moss_mkdirs_for_file(const char *path) {
    char buf[4096];
    size_t n = strlen(path);
    if (n == 0 || n >= sizeof(buf)) return -1;
    memcpy(buf, path, n + 1u);
    char *slash = strrchr(buf, '/');
    if (!slash || slash == buf) return 0;
    *slash = '\0';
    if (buf[0] == '\0') return 0;
    for (char *p = buf + 1; *p; p++) {
        if (*p != '/') continue;
        *p = '\0';
        if (mkdir(buf, (mode_t)0755) != 0 && errno != EEXIST) return -1;
        *p = '/';
    }
    if (mkdir(buf, (mode_t)0755) != 0 && errno != EEXIST) return -1;
    return 0;
}

int moss_write_wav16(const char *path, const float *samples, int n_samples, int sample_rate, int n_channels) {
    if (!path || !samples || n_samples <= 0 || sample_rate <= 0 || n_channels <= 0) return -1;
    if (n_samples % n_channels != 0) return -1;
    if (moss_mkdirs_for_file(path) != 0) return -1;
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
    uint16_t num_channels = (uint16_t)n_channels;
    uint32_t byte_rate = (uint32_t)(sample_rate * num_channels * sizeof(int16_t));
    uint16_t block_align = (uint16_t)(num_channels * sizeof(int16_t));
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
