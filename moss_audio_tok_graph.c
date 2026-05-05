#include "moss_audio_tok_internal.h"
#include "moss_kernel.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const safetensor_t *moss_audio_tok_find_tensor(const safetensors_file_t *sf, const char *name) {
    if (!sf || !name) return NULL;
    for (int i = 0; i < sf->num_tensors; i++) {
        if (strcmp(sf->tensors[i].name, name) == 0) return &sf->tensors[i];
    }
    return NULL;
}

int moss_audio_tok_get_f32_tensor(const safetensors_file_t *sf, const char *name, const float **ptr, int64_t *numel) {
    const safetensor_t *t = moss_audio_tok_find_tensor(sf, name);
    if (!t || t->dtype != DTYPE_F32) return -1;
    const void *p = safetensors_data(sf, t);
    if (!p) return -1;
    *ptr = (const float *)p;
    *numel = safetensor_numel(t);
    return 0;
}

static void layernorm_row_f32(float *out, const float *x, const float *w, const float *b, int n) {
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

int moss_audio_tok_patched_decode_tm(const float *in, int T, int Dh, int patch, float **out, int *outT, int *outD) {
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

int moss_audio_tok_patched_encode_tm(const float *in, int T, int D, int patch, float **out, int *outT, int *outD) {
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

int moss_audio_tok_transformer_module_tm(
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
    const int D = 256, H = 4, Dh = 64, dff = 1024;
    int64_t n = 0;
    char k[256];
    const float *w_in = NULL, *w_out = NULL;

    snprintf(k, sizeof(k), "%s.%d.input_proj.weight", prefix, module_id);
    if (moss_audio_tok_get_f32_tensor(tok->sf, k, &w_in, &n) != 0 || n != (int64_t)D * in_dim) return -1;
    snprintf(k, sizeof(k), "%s.%d.output_proj.weight", prefix, module_id);
    if (moss_audio_tok_get_f32_tensor(tok->sf, k, &w_out, &n) != 0 || n != (int64_t)out_dim * D) return -1;

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
    moss_batch_mm_nt_f32(in, w_in, x, T, D, in_dim);

    for (int li = 0; li < num_layers; li++) {
        const float *ln1_w = NULL, *ln1_b = NULL, *ln2_w = NULL, *ln2_b = NULL;
        const float *attn_in = NULL, *attn_out = NULL, *ff1 = NULL, *ff2 = NULL, *ls1 = NULL, *ls2 = NULL;
        snprintf(k, sizeof(k), "%s.%d.transformer.layers.%d.norm1.weight", prefix, module_id, li);
        if (moss_audio_tok_get_f32_tensor(tok->sf, k, &ln1_w, &n) != 0 || n != D) goto fail;
        snprintf(k, sizeof(k), "%s.%d.transformer.layers.%d.norm1.bias", prefix, module_id, li);
        if (moss_audio_tok_get_f32_tensor(tok->sf, k, &ln1_b, &n) != 0 || n != D) goto fail;
        snprintf(k, sizeof(k), "%s.%d.transformer.layers.%d.norm2.weight", prefix, module_id, li);
        if (moss_audio_tok_get_f32_tensor(tok->sf, k, &ln2_w, &n) != 0 || n != D) goto fail;
        snprintf(k, sizeof(k), "%s.%d.transformer.layers.%d.norm2.bias", prefix, module_id, li);
        if (moss_audio_tok_get_f32_tensor(tok->sf, k, &ln2_b, &n) != 0 || n != D) goto fail;
        snprintf(k, sizeof(k), "%s.%d.transformer.layers.%d.self_attn.in_proj.weight", prefix, module_id, li);
        if (moss_audio_tok_get_f32_tensor(tok->sf, k, &attn_in, &n) != 0 || n != 3 * D * D) goto fail;
        snprintf(k, sizeof(k), "%s.%d.transformer.layers.%d.self_attn.out_proj.weight", prefix, module_id, li);
        if (moss_audio_tok_get_f32_tensor(tok->sf, k, &attn_out, &n) != 0 || n != D * D) goto fail;
        snprintf(k, sizeof(k), "%s.%d.transformer.layers.%d.ffn.0.weight", prefix, module_id, li);
        if (moss_audio_tok_get_f32_tensor(tok->sf, k, &ff1, &n) != 0 || n != dff * D) goto fail;
        snprintf(k, sizeof(k), "%s.%d.transformer.layers.%d.ffn.2.weight", prefix, module_id, li);
        if (moss_audio_tok_get_f32_tensor(tok->sf, k, &ff2, &n) != 0 || n != D * dff) goto fail;
        snprintf(k, sizeof(k), "%s.%d.transformer.layers.%d.layer_scale_1.scale", prefix, module_id, li);
        if (moss_audio_tok_get_f32_tensor(tok->sf, k, &ls1, &n) != 0 || n != D) goto fail;
        snprintf(k, sizeof(k), "%s.%d.transformer.layers.%d.layer_scale_2.scale", prefix, module_id, li);
        if (moss_audio_tok_get_f32_tensor(tok->sf, k, &ls2, &n) != 0 || n != D) goto fail;

        for (int t = 0; t < T; t++) layernorm_row_f32(ln + (size_t)t * D, x + (size_t)t * D, ln1_w, ln1_b, D);
        moss_batch_mm_nt_f32(ln, attn_in, qkv, T, 3 * D, D);
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

        moss_batch_mm_nt_f32(attn, attn_out, tmp, T, D, D);
        for (int t = 0; t < T; t++) {
            for (int i = 0; i < D; i++) x[(size_t)t * D + i] += tmp[(size_t)t * D + i] * ls1[i];
        }

        for (int t = 0; t < T; t++) layernorm_row_f32(ln + (size_t)t * D, x + (size_t)t * D, ln2_w, ln2_b, D);
        moss_batch_mm_nt_f32(ln, ff1, ffh, T, dff, D);
        for (int t = 0; t < T; t++) {
            moss_gelu_new_inplace(ffh + (size_t)t * dff, dff);
        }
        moss_batch_mm_nt_f32(ffh, ff2, tmp, T, D, dff);
        for (int t = 0; t < T; t++) {
            for (int i = 0; i < D; i++) x[(size_t)t * D + i] += tmp[(size_t)t * D + i] * ls2[i];
        }
    }

    {
        float *y = (float *)malloc((size_t)T * (size_t)out_dim * sizeof(float));
        if (!y) goto fail;
        moss_batch_mm_nt_f32(x, w_out, y, T, out_dim, D);
        *out = y;
    }
    free(x); free(ln); free(q); free(kbuf); free(v); free(attn); free(qkv); free(ffh); free(tmp); free(cos); free(sin); free(pos);
    return 0;
fail:
    free(x); free(ln); free(q); free(kbuf); free(v); free(attn); free(qkv); free(ffh); free(tmp); free(cos); free(sin); free(pos);
    return -1;
}
