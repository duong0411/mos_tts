#include "moss_audio_tok.h"
#include "moss_kernel.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef MOSS_USE_CBLAS
#include <cblas.h>
/* Row-major X[T,in], W[out,in] -> Y[t,o] = sum_i X[t,i]*W[o,i]  (one SGEMM). */
static void moss_at_batch_mm_nt(const float *X, const float *W, float *Y, int T, int out_dim, int in_dim) {
    if (T <= 0 || out_dim <= 0 || in_dim <= 0) return;
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasTrans,
        T,
        out_dim,
        in_dim,
        1.0f,
        X,
        in_dim,
        W,
        in_dim,
        0.0f,
        Y,
        out_dim
    );
}
#endif

static int parse_json_int_in(const char *start, const char *end, const char *key, int *out) {
    char pat[128];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = start;
    while (p && p < end) {
        p = strstr(p, pat);
        if (!p || p >= end) return -1;
        p = strchr(p + strlen(pat), ':');
        if (!p || p >= end) return -1;
        p++;
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
        *out = atoi(p);
        return 0;
    }
    return -1;
}

static int has_required_tensor(safetensors_file_t *sf, const char *name) {
    for (int i = 0; i < sf->num_tensors; i++) {
        if (strcmp(sf->tensors[i].name, name) == 0) return 1;
    }
    return 0;
}

static const safetensor_t *find_tensor(const safetensors_file_t *sf, const char *name) {
    if (!sf || !name) return NULL;
    for (int i = 0; i < sf->num_tensors; i++) {
        if (strcmp(sf->tensors[i].name, name) == 0) return &sf->tensors[i];
    }
    return NULL;
}

static int get_f32_tensor(const safetensors_file_t *sf, const char *name, const float **ptr, int64_t *numel) {
    const safetensor_t *t = find_tensor(sf, name);
    if (!t || t->dtype != DTYPE_F32) return -1;
    const void *p = safetensors_data(sf, t);
    if (!p) return -1;
    *ptr = (const float *)p;
    *numel = safetensor_numel(t);
    return 0;
}

static void matvec_bias(float *y, const float *W, const float *b, const float *x, int rows, int cols) {
#ifdef MOSS_USE_CBLAS
    if (rows > 0 && cols > 0) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans, rows, cols, 1.0f, W, cols, x, 1, 0.0f, y, 1);
        if (b) {
            for (int r = 0; r < rows; r++) y[r] += b[r];
        }
        return;
    }
#endif
    for (int r = 0; r < rows; r++) {
        const float *wr = W + (size_t)r * cols;
        float s = b ? b[r] : 0.0f;
        for (int c = 0; c < cols; c++) s += wr[c] * x[c];
        y[r] = s;
    }
}

static void build_weightnorm_matrix(float *W_eff, const float *g, const float *v, int out_dim, int in_dim) {
    for (int o = 0; o < out_dim; o++) {
        const float *vo = v + (size_t)o * in_dim;
        float n2 = 0.0f;
        for (int i = 0; i < in_dim; i++) n2 += vo[i] * vo[i];
        float scale = g[o] / sqrtf(n2 + 1e-12f);
        float *wo = W_eff + (size_t)o * in_dim;
        for (int i = 0; i < in_dim; i++) wo[i] = vo[i] * scale;
    }
}

#ifndef MOSS_USE_CBLAS
static void linear_f32(float *y, const float *W, const float *x, int out_dim, int in_dim) {
    for (int o = 0; o < out_dim; o++) {
        const float *wo = W + (size_t)o * in_dim;
        float s = 0.0f;
        for (int i = 0; i < in_dim; i++) s += wo[i] * x[i];
        y[o] = s;
    }
}
#endif

static void layernorm_row_f32(
    float *out,
    const float *x,
    const float *w,
    const float *b,
    int n
) {
    float mean = 0.0f;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= (float)n;
    float var = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= (float)n;
    float inv = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < n; i++) out[i] = (x[i] - mean) * inv * w[i] + b[i];
}

/* Input/Output are time-major [T, D]. */
static int patched_decode_tm(const float *in, int T, int Dh, int patch, float **out, int *outT, int *outD) {
    if (!in || T <= 0 || Dh <= 0 || patch <= 0 || (Dh % patch) != 0) return -1;
    int D = Dh / patch;
    int TT = T * patch;
    float *y = (float *)malloc((size_t)TT * (size_t)D * sizeof(float));
    if (!y) return -1;
    for (int l = 0; l < T; l++) {
        for (int j = 0; j < patch; j++) {
            int t2 = l * patch + j;
            for (int d = 0; d < D; d++) {
                y[(size_t)t2 * D + d] = in[(size_t)l * Dh + d * patch + j];
            }
        }
    }
    *out = y;
    *outT = TT;
    *outD = D;
    return 0;
}

/* Input/Output are time-major [T, D]. */
static int patched_encode_tm(const float *in, int T, int D, int patch, float **out, int *outT, int *outD) {
    if (!in || T <= 0 || D <= 0 || patch <= 0 || (T % patch) != 0) return -1;
    int TT = T / patch;
    int DD = D * patch;
    float *y = (float *)malloc((size_t)TT * (size_t)DD * sizeof(float));
    if (!y) return -1;
    for (int l = 0; l < TT; l++) {
        for (int d = 0; d < D; d++) {
            for (int j = 0; j < patch; j++) {
                y[(size_t)l * DD + d * patch + j] = in[(size_t)(l * patch + j) * D + d];
            }
        }
    }
    *out = y;
    *outT = TT;
    *outD = DD;
    return 0;
}

static int transformer_module_tm(
    const moss_audio_tok_t *tok,
    const char *prefix,
    int module_id,
    int num_layers,
    int in_dim,
    int out_dim,
    int context_tokens,
    const float *in,
    int T,
    float **out
) {
    /* FFN inner dim; name must not be "I" — OpenBLAS cblas.h defines I as _Complex_I */
    const int D = 256, H = 4, Dh = 64, dff = 1024;
    int64_t n = 0;
    char k[256];
    const float *w_in = NULL, *w_out = NULL;

    snprintf(k, sizeof(k), "%s.%d.input_proj.weight", prefix, module_id);
    if (get_f32_tensor(tok->sf, k, &w_in, &n) != 0 || n != (int64_t)D * in_dim) return -1;
    snprintf(k, sizeof(k), "%s.%d.output_proj.weight", prefix, module_id);
    if (get_f32_tensor(tok->sf, k, &w_out, &n) != 0 || n != (int64_t)out_dim * D) return -1;

    float *x = (float *)malloc((size_t)T * D * sizeof(float));
    float *ln = (float *)malloc((size_t)T * D * sizeof(float));
    float *q = (float *)malloc((size_t)T * D * sizeof(float));
    float *kbuf = (float *)malloc((size_t)T * D * sizeof(float));
    float *v = (float *)malloc((size_t)T * D * sizeof(float));
    float *attn = (float *)malloc((size_t)T * D * sizeof(float));
    float *qkv = (float *)malloc((size_t)T * (3 * D) * sizeof(float));
    float *ffh = (float *)malloc((size_t)T * dff * sizeof(float));
    float *tmp = (float *)malloc((size_t)T * D * sizeof(float));
    float *cos = (float *)malloc((size_t)T * Dh * sizeof(float));
    float *sin = (float *)malloc((size_t)T * Dh * sizeof(float));
    int *pos = (int *)malloc((size_t)T * sizeof(int));
    if (!x || !ln || !q || !kbuf || !v || !attn || !qkv || !ffh || !tmp || !cos || !sin || !pos) goto fail;

    for (int t = 0; t < T; t++) pos[t] = t;
#ifdef MOSS_USE_CBLAS
    moss_at_batch_mm_nt(in, w_in, x, T, D, in_dim);
#else
    for (int t = 0; t < T; t++) linear_f32(x + (size_t)t * D, w_in, in + (size_t)t * in_dim, D, in_dim);
#endif

    for (int li = 0; li < num_layers; li++) {
        const float *ln1_w = NULL, *ln1_b = NULL, *ln2_w = NULL, *ln2_b = NULL;
        const float *attn_in = NULL, *attn_out = NULL, *ff1 = NULL, *ff2 = NULL, *ls1 = NULL, *ls2 = NULL;
        snprintf(k, sizeof(k), "%s.%d.transformer.layers.%d.norm1.weight", prefix, module_id, li);
        if (get_f32_tensor(tok->sf, k, &ln1_w, &n) != 0 || n != D) goto fail;
        snprintf(k, sizeof(k), "%s.%d.transformer.layers.%d.norm1.bias", prefix, module_id, li);
        if (get_f32_tensor(tok->sf, k, &ln1_b, &n) != 0 || n != D) goto fail;
        snprintf(k, sizeof(k), "%s.%d.transformer.layers.%d.norm2.weight", prefix, module_id, li);
        if (get_f32_tensor(tok->sf, k, &ln2_w, &n) != 0 || n != D) goto fail;
        snprintf(k, sizeof(k), "%s.%d.transformer.layers.%d.norm2.bias", prefix, module_id, li);
        if (get_f32_tensor(tok->sf, k, &ln2_b, &n) != 0 || n != D) goto fail;
        snprintf(k, sizeof(k), "%s.%d.transformer.layers.%d.self_attn.in_proj.weight", prefix, module_id, li);
        if (get_f32_tensor(tok->sf, k, &attn_in, &n) != 0 || n != 3 * D * D) goto fail;
        snprintf(k, sizeof(k), "%s.%d.transformer.layers.%d.self_attn.out_proj.weight", prefix, module_id, li);
        if (get_f32_tensor(tok->sf, k, &attn_out, &n) != 0 || n != D * D) goto fail;
        snprintf(k, sizeof(k), "%s.%d.transformer.layers.%d.ffn.0.weight", prefix, module_id, li);
        if (get_f32_tensor(tok->sf, k, &ff1, &n) != 0 || n != dff * D) goto fail;
        snprintf(k, sizeof(k), "%s.%d.transformer.layers.%d.ffn.2.weight", prefix, module_id, li);
        if (get_f32_tensor(tok->sf, k, &ff2, &n) != 0 || n != D * dff) goto fail;
        snprintf(k, sizeof(k), "%s.%d.transformer.layers.%d.layer_scale_1.scale", prefix, module_id, li);
        if (get_f32_tensor(tok->sf, k, &ls1, &n) != 0 || n != D) goto fail;
        snprintf(k, sizeof(k), "%s.%d.transformer.layers.%d.layer_scale_2.scale", prefix, module_id, li);
        if (get_f32_tensor(tok->sf, k, &ls2, &n) != 0 || n != D) goto fail;

        for (int t = 0; t < T; t++) layernorm_row_f32(ln + (size_t)t * D, x + (size_t)t * D, ln1_w, ln1_b, D);
#ifdef MOSS_USE_CBLAS
        moss_at_batch_mm_nt(ln, attn_in, qkv, T, 3 * D, D);
#else
        for (int t = 0; t < T; t++) {
            linear_f32(qkv + (size_t)t * (3 * D), attn_in, ln + (size_t)t * D, 3 * D, D);
        }
#endif
        for (int t = 0; t < T; t++) {
            memcpy(q + (size_t)t * D, qkv + (size_t)t * (3 * D), (size_t)D * sizeof(float));
            memcpy(kbuf + (size_t)t * D, qkv + (size_t)t * (3 * D) + D, (size_t)D * sizeof(float));
            memcpy(v + (size_t)t * D, qkv + (size_t)t * (3 * D) + 2 * D, (size_t)D * sizeof(float));
        }
        moss_rope_cos_sin(cos, sin, pos, T, Dh, 10000.0f);
        moss_apply_rope_inplace(q, cos, sin, T, H, Dh);
        moss_apply_rope_inplace(kbuf, cos, sin, T, H, Dh);

        memset(attn, 0, (size_t)T * D * sizeof(float));
        float scale = 1.0f / sqrtf((float)Dh);
        float *scores = (float *)malloc((size_t)T * sizeof(float));
        if (!scores) goto fail;
        for (int h = 0; h < H; h++) {
            for (int i = 0; i < T; i++) {
                int j0 = 0;
                if (context_tokens > 0) {
                    j0 = i - context_tokens + 1;
                    if (j0 < 0) j0 = 0;
                }
                int win = i - j0 + 1;
                for (int j = 0; j < win; j++) {
                    const float *qi = q + (size_t)i * D + h * Dh;
                    const float *kj = kbuf + (size_t)(j0 + j) * D + h * Dh;
                    float d = 0.0f;
                    for (int c = 0; c < Dh; c++) d += qi[c] * kj[c];
                    scores[j] = d * scale;
                }
                moss_softmax_rows(scores, 1, win);
                float *oi = attn + (size_t)i * D + h * Dh;
                for (int c = 0; c < Dh; c++) oi[c] = 0.0f;
                for (int j = 0; j < win; j++) {
                    const float *vj = v + (size_t)(j0 + j) * D + h * Dh;
                    float p = scores[j];
                    for (int c = 0; c < Dh; c++) oi[c] += p * vj[c];
                }
            }
        }
        free(scores);

#ifdef MOSS_USE_CBLAS
        moss_at_batch_mm_nt(attn, attn_out, tmp, T, D, D);
#else
        for (int t = 0; t < T; t++) {
            linear_f32(tmp + (size_t)t * D, attn_out, attn + (size_t)t * D, D, D);
        }
#endif
        for (int t = 0; t < T; t++) {
            for (int i = 0; i < D; i++) x[(size_t)t * D + i] += tmp[(size_t)t * D + i] * ls1[i];
        }

        for (int t = 0; t < T; t++) layernorm_row_f32(ln + (size_t)t * D, x + (size_t)t * D, ln2_w, ln2_b, D);
#ifdef MOSS_USE_CBLAS
        moss_at_batch_mm_nt(ln, ff1, ffh, T, dff, D);
#else
        for (int t = 0; t < T; t++) {
            linear_f32(ffh + (size_t)t * dff, ff1, ln + (size_t)t * D, dff, D);
        }
#endif
        for (int t = 0; t < T; t++) {
            moss_gelu_new_inplace(ffh + (size_t)t * dff, dff);
        }
#ifdef MOSS_USE_CBLAS
        moss_at_batch_mm_nt(ffh, ff2, tmp, T, D, dff);
#else
        for (int t = 0; t < T; t++) {
            linear_f32(tmp + (size_t)t * D, ff2, ffh + (size_t)t * dff, D, dff);
        }
#endif
        for (int t = 0; t < T; t++) {
            for (int i = 0; i < D; i++) x[(size_t)t * D + i] += tmp[(size_t)t * D + i] * ls2[i];
        }
    }

    {
        float *y = (float *)malloc((size_t)T * (size_t)out_dim * sizeof(float));
        if (!y) goto fail;
#ifdef MOSS_USE_CBLAS
        moss_at_batch_mm_nt(x, w_out, y, T, out_dim, D);
#else
        for (int t = 0; t < T; t++) linear_f32(y + (size_t)t * out_dim, w_out, x + (size_t)t * D, out_dim, D);
#endif
        *out = y;
    }
    free(x); free(ln); free(q); free(kbuf); free(v); free(attn); free(qkv); free(ffh); free(tmp); free(cos); free(sin); free(pos);
    return 0;
fail:
    free(x); free(ln); free(q); free(kbuf); free(v); free(attn); free(qkv); free(ffh); free(tmp); free(cos); free(sin); free(pos);
    return -1;
}

int moss_audio_tok_load(const char *model_dir, moss_audio_tok_t *out) {
    if (!model_dir || !out) return -1;
    memset(out, 0, sizeof(*out));

    char cfg_path[1024];
    snprintf(cfg_path, sizeof(cfg_path), "%s/audio_tokenizer/config.json", model_dir);
    FILE *f = fopen(cfg_path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) {
        fclose(f);
        return -1;
    }
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) {
        free(buf);
        fclose(f);
        return -1;
    }
    fclose(f);
    buf[n] = '\0';

    const char *end = buf + n;
    out->cfg.sample_rate = 48000;
    out->cfg.downsample_rate = 3840;
    out->cfg.channels = 2;
    out->cfg.num_quantizers = 16;
    out->cfg.codebook_size = 1024;
    parse_json_int_in(buf, end, "sample_rate", &out->cfg.sample_rate);
    parse_json_int_in(buf, end, "downsample_rate", &out->cfg.downsample_rate);
    parse_json_int_in(buf, end, "number_channels", &out->cfg.channels);
    parse_json_int_in(buf, end, "codebook_size", &out->cfg.codebook_size);
    parse_json_int_in(buf, end, "num_quantizers", &out->cfg.num_quantizers);
    free(buf);

    snprintf(out->model_path, sizeof(out->model_path), "%s/audio_tokenizer/model-00001-of-00001.safetensors", model_dir);
    out->sf = safetensors_open(out->model_path);
    if (!out->sf) return -1;

    if (!has_required_tensor(out->sf, "encoder.1.input_proj.weight") ||
        !has_required_tensor(out->sf, "quantizer.quantizers.0.codebook.weight") ||
        !has_required_tensor(out->sf, "decoder.7.output_proj.weight")) {
        moss_audio_tok_unload(out);
        return -1;
    }
    return 0;
}

void moss_audio_tok_unload(moss_audio_tok_t *tok) {
    if (!tok) return;
    if (tok->sf) {
        safetensors_close(tok->sf);
        tok->sf = NULL;
    }
    memset(tok, 0, sizeof(*tok));
}

int moss_audio_tok_is_loaded(const moss_audio_tok_t *tok) {
    return tok && tok->sf != NULL;
}

int moss_audio_tok_encode_wav(
    const moss_audio_tok_t *tok,
    const float *pcm,
    int n_samples,
    int n_channels,
    int **out_codes,
    int *out_frames
) {
    (void)tok;
    if (!tok || !pcm || n_samples <= 0 || n_channels <= 0 || !out_codes || !out_frames) return -1;
    float *x0 = NULL, *x1 = NULL, *x2 = NULL, *x3 = NULL, *x4 = NULL, *x5 = NULL, *x6 = NULL, *x7 = NULL, *x8 = NULL;
    float *z768_tm = NULL;
    int err_step = 0;

    const int nq = tok->cfg.num_quantizers > 0 ? tok->cfg.num_quantizers : 16;
    const int hop = tok->cfg.downsample_rate > 0 ? tok->cfg.downsample_rate : 3840;
    const int cb = tok->cfg.codebook_size > 0 ? tok->cfg.codebook_size : 1024;
    /* Match Python length path:
       - channel interleave doubles temporal length for stereo
       - encoder length tracking uses floor divisions (no ceil)
       => final frame count is floor(n_samples / downsample_rate). */
    const int frames = n_samples / hop;
    if (frames <= 0) {
        return -1;
    }
    int *codes = (int *)malloc((size_t)frames * (size_t)nq * sizeof(int));
    if (!codes) return -1;

    int64_t _n = 0;

    const float *q_in_g = NULL, *q_in_v = NULL, *q_in_b = NULL;
    if (get_f32_tensor(tok->sf, "quantizer.input_proj.parametrizations.weight.original0", &q_in_g, &_n) != 0 || _n != 512) {
        free(codes);
        return -1;
    }
    if (get_f32_tensor(tok->sf, "quantizer.input_proj.parametrizations.weight.original1", &q_in_v, &_n) != 0 || _n != 512 * 768) {
        free(codes);
        return -1;
    }
    if (get_f32_tensor(tok->sf, "quantizer.input_proj.bias", &q_in_b, &_n) != 0 || _n != 512) {
        free(codes);
        return -1;
    }
    float *q_in_w = (float *)malloc((size_t)512 * 768 * sizeof(float));
    if (!q_in_w) {
        free(codes);
        return -1;
    }
    build_weightnorm_matrix(q_in_w, q_in_g, q_in_v, 512, 768);

    float *q_q_in_w[16] = {0};
    const float *q_q_in_b[16] = {0};
    const float *q_q_codebook[16] = {0};
    float *q_q_out_w[16] = {0};
    const float *q_q_out_b[16] = {0};
    for (int q = 0; q < nq; q++) {
        char k_g[256], k_v[256], k_b[256], k_cb[128], k_og[256], k_ov[256], k_ob[256];
        snprintf(k_g, sizeof(k_g), "quantizer.quantizers.%d.in_proj.parametrizations.weight.original0", q);
        snprintf(k_v, sizeof(k_v), "quantizer.quantizers.%d.in_proj.parametrizations.weight.original1", q);
        snprintf(k_b, sizeof(k_b), "quantizer.quantizers.%d.in_proj.bias", q);
        snprintf(k_cb, sizeof(k_cb), "quantizer.quantizers.%d.codebook.weight", q);
        snprintf(k_og, sizeof(k_og), "quantizer.quantizers.%d.out_proj.parametrizations.weight.original0", q);
        snprintf(k_ov, sizeof(k_ov), "quantizer.quantizers.%d.out_proj.parametrizations.weight.original1", q);
        snprintf(k_ob, sizeof(k_ob), "quantizer.quantizers.%d.out_proj.bias", q);

        const float *ig = NULL, *iv = NULL, *ib = NULL, *cbw = NULL, *og = NULL, *ov = NULL, *ob = NULL;
        if (get_f32_tensor(tok->sf, k_g, &ig, &_n) != 0 || _n != 8) { err_step = 11; goto fail; }
        if (get_f32_tensor(tok->sf, k_v, &iv, &_n) != 0 || _n != 8 * 512) { err_step = 12; goto fail; }
        if (get_f32_tensor(tok->sf, k_b, &ib, &_n) != 0 || _n != 8) { err_step = 13; goto fail; }
        if (get_f32_tensor(tok->sf, k_cb, &cbw, &_n) != 0 || _n != cb * 8) { err_step = 14; goto fail; }
        if (get_f32_tensor(tok->sf, k_og, &og, &_n) != 0 || _n != 512) { err_step = 15; goto fail; }
        if (get_f32_tensor(tok->sf, k_ov, &ov, &_n) != 0 || _n != 512 * 8) { err_step = 16; goto fail; }
        if (get_f32_tensor(tok->sf, k_ob, &ob, &_n) != 0 || _n != 512) { err_step = 17; goto fail; }

        q_q_in_w[q] = (float *)malloc((size_t)8 * 512 * sizeof(float));
        q_q_out_w[q] = (float *)malloc((size_t)512 * 8 * sizeof(float));
        if (!q_q_in_w[q] || !q_q_out_w[q]) { err_step = 18; goto fail; }
        build_weightnorm_matrix(q_q_in_w[q], ig, iv, 8, 512);
        build_weightnorm_matrix(q_q_out_w[q], og, ov, 512, 8);
        q_q_in_b[q] = ib;
        q_q_codebook[q] = cbw;
        q_q_out_b[q] = ob;
    }

    float z768[768];
    float z512[512];
    float e8[8];
    float zq512[512];

    int ch = tok->cfg.channels > 0 ? tok->cfg.channels : 2;
    int n_interleave = n_samples * ch;
    int full_patch = 240 * 2 * 2 * 2 * 4; /* 7680 after channel interleave */
    int rem = n_interleave % full_patch;
    if (rem != 0) n_interleave += (full_patch - rem);
    x0 = (float *)calloc((size_t)n_interleave, sizeof(float));
    if (!x0) { err_step = 21; goto fail; }
    for (int s = 0; s < n_samples; s++) {
        for (int c = 0; c < ch; c++) {
            int sc = (n_channels == 1) ? 0 : (c < n_channels ? c : n_channels - 1);
            x0[s * ch + c] = pcm[(size_t)s * n_channels + sc];
        }
    }

    int T0 = n_interleave, D0 = 1;
    int T1 = 0, D1 = 0, T2 = 0, D2 = 0, T3 = 0, D3 = 0, T4 = 0, D4 = 0;
    int T5 = 0, D5 = 0, T6 = 0, D6 = 0, T7 = 0, D7 = 0, T8 = 0, D8 = 0;
    if (patched_encode_tm(x0, T0, D0, 240, &x1, &T1, &D1) != 0) { err_step = 22; goto fail; }
    if (transformer_module_tm(tok, "encoder", 1, 4, D1, 384, 1600, x1, T1, &x2) != 0) { err_step = 23; goto fail; }
    T2 = T1; D2 = 384;
    if (patched_encode_tm(x2, T2, D2, 2, &x3, &T3, &D3) != 0) { err_step = 24; goto fail; }
    if (transformer_module_tm(tok, "encoder", 3, 2, D3, 384, 1200, x3, T3, &x4) != 0) { err_step = 25; goto fail; }
    T4 = T3; D4 = 384;
    if (patched_encode_tm(x4, T4, D4, 2, &x5, &T5, &D5) != 0) { err_step = 26; goto fail; }
    if (transformer_module_tm(tok, "encoder", 5, 2, D5, 384, 800, x5, T5, &x6) != 0) { err_step = 27; goto fail; }
    T6 = T5; D6 = 384;
    if (patched_encode_tm(x6, T6, D6, 2, &x7, &T7, &D7) != 0) { err_step = 28; goto fail; }
    if (transformer_module_tm(tok, "encoder", 7, 4, D7, 192, 500, x7, T7, &x8) != 0) { err_step = 29; goto fail; }
    T8 = T7; D8 = 192;
    int Tf = 0, Df = 0;
    if (patched_encode_tm(x8, T8, D8, 4, &z768_tm, &Tf, &Df) != 0) { err_step = 30; goto fail; }
    if (Df != 768) { err_step = 31; goto fail; }
    if (Tf < frames) { err_step = 32; goto fail; }

    for (int t = 0; t < frames; t++) {
        memcpy(z768, z768_tm + (size_t)t * 768, 768 * sizeof(float));
        matvec_bias(z512, q_in_w, q_in_b, z768, 512, 768);

        for (int q = 0; q < nq; q++) {
            matvec_bias(e8, q_q_in_w[q], q_q_in_b[q], z512, 8, 512);
            float en = 0.0f;
            for (int i = 0; i < 8; i++) en += e8[i] * e8[i];
            en = sqrtf(en + 1e-12f);
            for (int i = 0; i < 8; i++) e8[i] /= en;

            int best = 0;
            float best_dot = -1e30f;
            for (int k = 0; k < cb; k++) {
                const float *cw = q_q_codebook[q] + (size_t)k * 8;
                float cn = 0.0f, dot = 0.0f;
                for (int d = 0; d < 8; d++) {
                    cn += cw[d] * cw[d];
                    dot += e8[d] * cw[d];
                }
                dot /= sqrtf(cn + 1e-12f);
                if (dot > best_dot) {
                    best_dot = dot;
                    best = k;
                }
            }
            codes[(size_t)t * nq + q] = best;

            const float *cbv = q_q_codebook[q] + (size_t)best * 8;
            matvec_bias(zq512, q_q_out_w[q], q_q_out_b[q], cbv, 512, 8);
            for (int i = 0; i < 512; i++) z512[i] -= zq512[i];
        }
    }

    for (int q = 0; q < nq; q++) {
        free(q_q_in_w[q]);
        free(q_q_out_w[q]);
    }
    free(z768_tm);
    free(x8); free(x7); free(x6); free(x5); free(x4); free(x3); free(x2); free(x1); free(x0);
    free(q_in_w);
    *out_codes = codes;
    *out_frames = frames;
    return 0;

fail:
    fprintf(stderr, "[moss_audio_tok] encode_wav failed at step=%d\n", err_step);
    fflush(stderr);
    free(z768_tm);
    free(x8); free(x7); free(x6); free(x5); free(x4); free(x3); free(x2); free(x1); free(x0);
    for (int q = 0; q < nq; q++) {
        free(q_q_in_w[q]);
        free(q_q_out_w[q]);
    }
    free(q_in_w);
    free(codes);
    return -1;
}

static int read_le16(FILE *f, unsigned short *out) {
    unsigned char b[2];
    if (fread(b, 1, 2, f) != 2) return -1;
    *out = (unsigned short)(b[0] | (b[1] << 8));
    return 0;
}

static int read_le32(FILE *f, unsigned int *out) {
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4) return -1;
    *out = (unsigned int)(b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24));
    return 0;
}

static unsigned long moss_aud_tmp_salt(void) {
    uintptr_t u = (uintptr_t)&fopen;
    return (unsigned long)(time(NULL) ^ ((unsigned long)(u >> 4) ^ (unsigned long)(u << 17)));
}

static int moss_aud_tmp_pcm_wav_path(char *dst, size_t dst_sz, unsigned long salt) {
#ifdef _WIN32
    static const char def_dir[] = ".";
#else
    static const char def_dir[] = "/tmp";
#endif
#ifdef _WIN32
    static const char sep = '\\';
#else
    static const char sep = '/';
#endif

    const char *d = getenv("MOSS_TMPDIR");
    if (!d || !d[0]) d = getenv("TMPDIR");
    if (!d || !d[0]) d = getenv("TEMP");
    if (!d || !d[0]) d = getenv("TMP");
    if (!d || !d[0]) d = def_dir;

    size_t dn = strlen(d);
    if (dst_sz <= dn + 80) return -1;
    memcpy(dst, d, dn);
    dst[dn] = '\0';
    if (dn > 0 && dst[dn - 1] != '/' && dst[dn - 1] != '\\') {
        dst[dn++] = sep;
        dst[dn] = '\0';
    }
    snprintf(dst + dn, dst_sz - dn, "moss_aud_%lx.wav", (unsigned long)salt);
    return 0;
}

static int moss_run_ffmpeg_to_pcm_wav(
    const char *ffmpeg_exe,
    const char *in_path,
    const char *out_path,
    int dst_sr,
    int dst_ch
) {
    char cmd[9216];
    int nw = snprintf(
        cmd,
        sizeof(cmd),
        "\"%s\" -nostdin -hide_banner -loglevel error -y -i \"%s\" "
        "-ac %d -ar %d -sample_fmt s16 -c:a pcm_s16le \"%s\"",
        ffmpeg_exe,
        in_path,
        dst_ch,
        dst_sr,
        out_path);
    if (nw <= 0 || (size_t)nw >= sizeof(cmd)) return -1;
    int st = system(cmd);
    if (st != 0)
        return -1;
    return 0;
}

/*
 * Assuming f is at BOF and starts with valid RIFF…WAVE PCM (fmt parsed here).
 */
static int moss_aud_read_riff_wave_and_encode(
    const moss_audio_tok_t *tok,
    FILE *f,
    int dst_sr,
    int dst_ch,
    int **out_codes,
    int *out_frames
) {
    char riff[4], wave[4];
    unsigned int riff_size_unused = 0;
    if (fread(riff, 1, 4, f) != 4 || read_le32(f, &riff_size_unused) != 0 || fread(wave, 1, 4, f) != 4)
        return -1;
    if (memcmp(riff, "RIFF", 4) != 0 || memcmp(wave, "WAVE", 4) != 0)
        return -1;

    unsigned short audio_format = 1, channels = 1, bits = 16;
    unsigned int sample_rate = 0, data_size = 0;
    long data_pos = -1;
    while (!feof(f)) {
        char ck[4];
        unsigned int csz = 0;
        if (fread(ck, 1, 4, f) != 4) break;
        if (read_le32(f, &csz) != 0) break;
        if (memcmp(ck, "fmt ", 4) == 0) {
            read_le16(f, &audio_format);
            read_le16(f, &channels);
            read_le32(f, &sample_rate);
            unsigned int byte_rate = 0;
            unsigned short align = 0;
            read_le32(f, &byte_rate);
            read_le16(f, &align);
            read_le16(f, &bits);
            if (csz > 16) fseek(f, (long)(csz - 16), SEEK_CUR);
        } else if (memcmp(ck, "data", 4) == 0) {
            data_pos = ftell(f);
            data_size = csz;
            fseek(f, (long)csz, SEEK_CUR);
        } else {
            fseek(f, (long)csz, SEEK_CUR);
        }
    }

    if (data_pos < 0 || data_size == 0 || sample_rate == 0)
        return -1;
    if (!(audio_format == 1 || audio_format == 3))
        return -1;
    if (!(bits == 16 || bits == 32))
        return -1;

    fseek(f, data_pos, SEEK_SET);
    int src_channels = (int)channels;
    int src_samples = (int)(data_size / (unsigned int)(src_channels * (bits / 8)));
    float *src = (float *)malloc((size_t)src_samples * (size_t)src_channels * sizeof(float));
    if (!src)
        return -1;

    if (bits == 16) {
        for (int i = 0; i < src_samples * src_channels; i++) {
            short v = 0;
            if (fread(&v, 2, 1, f) != 1) {
                free(src);
                return -1;
            }
            src[i] = (float)v / 32768.0f;
        }
    } else {
        for (int i = 0; i < src_samples * src_channels; i++) {
            float v = 0.0f;
            if (fread(&v, 4, 1, f) != 1) {
                free(src);
                return -1;
            }
            src[i] = v;
        }
    }

    int dst_samples = (int)((int64_t)src_samples * dst_sr / (int)sample_rate);
    if (dst_samples < 1) dst_samples = 1;
    float *dst = (float *)malloc((size_t)dst_samples * (size_t)dst_ch * sizeof(float));
    if (!dst) {
        free(src);
        return -1;
    }

    for (int t = 0; t < dst_samples; t++) {
        float pos = ((float)t * (float)sample_rate) / (float)dst_sr;
        int i0 = (int)pos;
        int i1 = i0 + 1;
        if (i0 < 0) i0 = 0;
        if (i0 >= src_samples) i0 = src_samples - 1;
        if (i1 >= src_samples) i1 = src_samples - 1;
        float a = pos - (float)i0;
        for (int c = 0; c < dst_ch; c++) {
            int sc = (src_channels == 1) ? 0 : (c < src_channels ? c : src_channels - 1);
            float s0 = src[(size_t)i0 * src_channels + sc];
            float s1 = src[(size_t)i1 * src_channels + sc];
            dst[(size_t)t * dst_ch + c] = s0 + (s1 - s0) * a;
        }
    }
    free(src);

    int rc = moss_audio_tok_encode_wav(tok, dst, dst_samples, dst_ch, out_codes, out_frames);
    free(dst);
    return rc;
}

int moss_audio_tok_encode_wav_file(
    const moss_audio_tok_t *tok,
    const char *wav_path,
    int **out_codes,
    int *out_frames
) {
    if (!tok || !wav_path || !out_codes || !out_frames) return -1;
    const int dst_sr = tok->cfg.sample_rate > 0 ? tok->cfg.sample_rate : 48000;
    const int dst_ch = tok->cfg.channels > 0 ? tok->cfg.channels : 2;

    FILE *f = fopen(wav_path, "rb");
    if (!f) return -1;

    unsigned char fh[16];
    size_t nh = fread(fh, 1, 12, f);
    if (nh >= 12 && memcmp(fh, "RIFF", 4) == 0 && memcmp(fh + 8, "WAVE", 4) == 0) {
        rewind(f);
        int rc = moss_aud_read_riff_wave_and_encode(tok, f, dst_sr, dst_ch, out_codes, out_frames);
        fclose(f);
        return rc;
    }
    fclose(f);

    const char *no_ff = getenv("MOSS_DISABLE_FFMPEG");
    if (no_ff && no_ff[0] != '\0') {
        fprintf(stderr,
                "[moss_audio_tok] not a RIFF/WAVE PCM file (e.g. FLAC-in-.wav): %s.\n",
                wav_path);
        fprintf(stderr,
                "  Use PCM WAV, or unset MOSS_DISABLE_FFMPEG and install ffmpeg for auto-transcode.\n");
        return -1;
    }

    const char *ffmpeg_exe = getenv("MOSS_FFMPEG");
    if (!ffmpeg_exe || !ffmpeg_exe[0]) ffmpeg_exe = "ffmpeg";

    char tmp_path[768];
    if (moss_aud_tmp_pcm_wav_path(tmp_path, sizeof(tmp_path), moss_aud_tmp_salt()) != 0) return -1;

    fprintf(stderr, "[moss_audio_tok] transcoding via ffmpeg → %dkHz %dch PCM WAV (%s …)\n",
            dst_sr / 1000,
            dst_ch,
            ffmpeg_exe);

    if (moss_run_ffmpeg_to_pcm_wav(ffmpeg_exe, wav_path, tmp_path, dst_sr, dst_ch) != 0) {
        fprintf(stderr,
                "[moss_audio_tok] ffmpeg failed (need ffmpeg in PATH, or MOSS_FFMPEG to its binary).\n");
        remove(tmp_path);
        return -1;
    }

    FILE *tf = fopen(tmp_path, "rb");
    if (!tf) {
        remove(tmp_path);
        return -1;
    }
    nh = fread(fh, 1, 12, tf);
    if (nh < 12 || memcmp(fh, "RIFF", 4) != 0 || memcmp(fh + 8, "WAVE", 4) != 0) {
        fclose(tf);
        remove(tmp_path);
        fprintf(stderr, "[moss_audio_tok] ffmpeg produced non-WAV output\n");
        return -1;
    }
    rewind(tf);
    int rc = moss_aud_read_riff_wave_and_encode(tok, tf, dst_sr, dst_ch, out_codes, out_frames);
    fclose(tf);
    remove(tmp_path);
    return rc;
}

int moss_audio_tok_decode_codes(
    const moss_audio_tok_t *tok,
    const int *codes,
    int frames,
    int sample_rate,
    float **out_samples,
    int *out_n_samples,
    int *out_n_channels
) {
    if (!out_samples || !out_n_samples || !out_n_channels) return -1;
    *out_samples = NULL;
    *out_n_samples = 0;
    *out_n_channels = 1;
    if (!tok || !tok->sf || !codes || frames <= 0 || sample_rate <= 0) return -1;

    const int nq = tok->cfg.num_quantizers > 0 ? tok->cfg.num_quantizers : 16;
    const int cb = tok->cfg.codebook_size > 0 ? tok->cfg.codebook_size : 1024;
    const int model_sr = tok->cfg.sample_rate > 0 ? tok->cfg.sample_rate : 48000;
    float *z768_tm = NULL;
    float *x1 = NULL, *x2 = NULL, *x3 = NULL, *x4 = NULL, *x5 = NULL, *x6 = NULL, *x7 = NULL, *x8 = NULL;
    float *wav_interleave = NULL;

    const float *q_out_g = NULL, *q_out_v = NULL, *q_out_b = NULL;
    int64_t n = 0;
    if (get_f32_tensor(tok->sf, "quantizer.output_proj.parametrizations.weight.original0", &q_out_g, &n) != 0 || n != 768) return -1;
    if (get_f32_tensor(tok->sf, "quantizer.output_proj.parametrizations.weight.original1", &q_out_v, &n) != 0 || n != 768 * 512) return -1;
    if (get_f32_tensor(tok->sf, "quantizer.output_proj.bias", &q_out_b, &n) != 0 || n != 768) return -1;

    float *q_out_w = (float *)malloc((size_t)768 * 512 * sizeof(float));
    if (!q_out_w) return -1;
    build_weightnorm_matrix(q_out_w, q_out_g, q_out_v, 768, 512);

    float *q_q_out_w[16] = {0};
    const float *q_q_out_b[16] = {0};
    const float *q_q_codebook[16] = {0};
    for (int q = 0; q < nq; q++) {
        char k_cb[128], k_og[256], k_ov[256], k_ob[256];
        snprintf(k_cb, sizeof(k_cb), "quantizer.quantizers.%d.codebook.weight", q);
        snprintf(k_og, sizeof(k_og), "quantizer.quantizers.%d.out_proj.parametrizations.weight.original0", q);
        snprintf(k_ov, sizeof(k_ov), "quantizer.quantizers.%d.out_proj.parametrizations.weight.original1", q);
        snprintf(k_ob, sizeof(k_ob), "quantizer.quantizers.%d.out_proj.bias", q);

        const float *cbw = NULL, *og = NULL, *ov = NULL, *ob = NULL;
        if (get_f32_tensor(tok->sf, k_cb, &cbw, &n) != 0 || n != cb * 8) goto fail;
        if (get_f32_tensor(tok->sf, k_og, &og, &n) != 0 || n != 512) goto fail;
        if (get_f32_tensor(tok->sf, k_ov, &ov, &n) != 0 || n != 512 * 8) goto fail;
        if (get_f32_tensor(tok->sf, k_ob, &ob, &n) != 0 || n != 512) goto fail;
        q_q_out_w[q] = (float *)malloc((size_t)512 * 8 * sizeof(float));
        if (!q_q_out_w[q]) goto fail;
        build_weightnorm_matrix(q_q_out_w[q], og, ov, 512, 8);
        q_q_out_b[q] = ob;
        q_q_codebook[q] = cbw;
    }

    z768_tm = (float *)malloc((size_t)frames * 768 * sizeof(float));
    if (!z768_tm) goto fail;
    float z512[512], t512[512], z768[768];
    for (int t = 0; t < frames; t++) {
        memset(z512, 0, sizeof(z512));
        for (int q = 0; q < nq; q++) {
            int id = codes[t * nq + q];
            if (id < 0) id = 0;
            if (id >= cb) id = cb - 1;
            const float *cbv = q_q_codebook[q] + (size_t)id * 8;
            matvec_bias(t512, q_q_out_w[q], q_q_out_b[q], cbv, 512, 8);
            for (int i = 0; i < 512; i++) z512[i] += t512[i];
        }

        matvec_bias(z768, q_out_w, q_out_b, z512, 768, 512);
        memcpy(z768_tm + (size_t)t * 768, z768, 768 * sizeof(float));
    }

    float *x0 = NULL;
    int T0 = frames, D0 = 768;
    int T1 = 0, D1 = 0, T2 = 0, D2 = 0, T3 = 0, D3 = 0, T4 = 0, D4 = 0;
    int T5 = 0, D5 = 0, T6 = 0, D6 = 0, T7 = 0, D7 = 0, T8 = 0, D8 = 0;
    x0 = z768_tm;
    if (patched_decode_tm(x0, T0, D0, 4, &x1, &T1, &D1) != 0) goto fail;
    if (transformer_module_tm(tok, "decoder", 1, 4, D1, 768, 500, x1, T1, &x2) != 0) goto fail;
    T2 = T1; D2 = 768;
    if (patched_decode_tm(x2, T2, D2, 2, &x3, &T3, &D3) != 0) goto fail;
    if (transformer_module_tm(tok, "decoder", 3, 2, D3, 768, 400, x3, T3, &x4) != 0) goto fail;
    T4 = T3; D4 = 768;
    if (patched_decode_tm(x4, T4, D4, 2, &x5, &T5, &D5) != 0) goto fail;
    if (transformer_module_tm(tok, "decoder", 5, 2, D5, 768, 300, x5, T5, &x6) != 0) goto fail;
    T6 = T5; D6 = 768;
    if (patched_decode_tm(x6, T6, D6, 2, &x7, &T7, &D7) != 0) goto fail;
    if (transformer_module_tm(tok, "decoder", 7, 4, D7, 240, 200, x7, T7, &x8) != 0) goto fail;
    T8 = T7; D8 = 240;

    int Ti = 0, Di = 0;
    if (patched_decode_tm(x8, T8, D8, 240, &wav_interleave, &Ti, &Di) != 0) goto fail;
    if (Di != 1) goto fail;

    int ch = tok->cfg.channels > 0 ? tok->cfg.channels : 2;
    if (ch < 1) ch = 1;
    int in_frames = Ti / ch;
    if (in_frames < 1) goto fail;
    int out_frames = (sample_rate == model_sr) ? in_frames : (int)((int64_t)in_frames * sample_rate / model_sr);
    if (out_frames < 1) out_frames = 1;
    int out_n = out_frames * ch;
    float *out = (float *)malloc((size_t)out_n * sizeof(float));
    if (!out) goto fail;
    if (sample_rate == model_sr) {
        memcpy(out, wav_interleave, (size_t)out_n * sizeof(float));
    } else {
        for (int i = 0; i < out_frames; i++) {
            float pos = ((float)i * (float)model_sr) / (float)sample_rate;
            int i0 = (int)pos;
            int i1 = i0 + 1;
            if (i0 < 0) i0 = 0;
            if (i0 >= in_frames) i0 = in_frames - 1;
            if (i1 >= in_frames) i1 = in_frames - 1;
            float a = pos - (float)i0;
            for (int c = 0; c < ch; c++) {
                float s0 = wav_interleave[(size_t)i0 * ch + c];
                float s1 = wav_interleave[(size_t)i1 * ch + c];
                out[(size_t)i * ch + c] = s0 + (s1 - s0) * a;
            }
        }
    }
    free(x1); free(x2); free(x3); free(x4); free(x5); free(x6); free(x7); free(x8);
    free(wav_interleave);
    free(z768_tm);

    for (int q = 0; q < nq; q++) free(q_q_out_w[q]);
    free(q_out_w);
    *out_samples = out;
    *out_n_samples = out_n;
    *out_n_channels = ch;
    return 0;

fail:
    free(z768_tm);
    free(x1); free(x2); free(x3); free(x4); free(x5); free(x6); free(x7); free(x8);
    free(wav_interleave);
    for (int q = 0; q < nq; q++) free(q_q_out_w[q]);
    free(q_out_w);
    return -1;
}
