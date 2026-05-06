#include "moss_gpt2.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "moss_kernel.h"

static void zero_row(float *row, int D) {
    memset(row, 0, (size_t)D * sizeof(float));
}

static void apply_attn_mask_scale_rows(float *hidden, const unsigned char *mask, int S, int D) {
    for (int s = 0; s < S; s++) {
        if (!mask[s]) zero_row(hidden + s * D, D);
    }
}

static void log_row_stats(
    const char *stack,
    int call_idx,
    int layer_idx,
    const char *tag,
    const float *row,
    int dim
) {
    if (!tag || !row || dim <= 0) return;
    const char *sk = (stack && stack[0]) ? stack : "gpt2";
    double sum = 0.0;
    double sq = 0.0;
    float mn = row[0], mx = row[0];
    for (int i = 0; i < dim; i++) {
        float v = row[i];
        sum += (double)v;
        sq += (double)v * (double)v;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    fprintf(stderr,
        "[moss_layer] stack=%s call=%d layer=%d dim=%d stage=%s sum=%.9g l2=%.9g min=%.9g max=%.9g\n",
        sk,
        call_idx,
        layer_idx,
        dim,
        tag,
        sum,
        sqrt(sq),
        (double)mn,
        (double)mx);
    fflush(stderr);
}

static void moss_debug_log_attn_scores_pre_softmax(
    int call_idx,
    int layer_idx,
    int head,
    int query_row,
    int seq_len,
    const float *rowscores,
    const unsigned char *attn_mask
) {
    if (!rowscores || !attn_mask || seq_len <= 0 || query_row < 0) return;
    double sum = 0.0;
    double sq = 0.0;
    int nvalid = 0;
    float mn = 0.0f, mx = 0.0f;
    int argmax_j = -1;
    float argmax_v = -1e30f;
    int first = 1;
    for (int j = 0; j < seq_len; j++) {
        if (!attn_mask[j] || j > query_row) continue;
        float v = rowscores[j];
        if (first) {
            mn = mx = v;
            first = 0;
        } else {
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        sum += (double)v;
        sq += (double)v * (double)v;
        nvalid++;
        if (v > argmax_v) {
            argmax_v = v;
            argmax_j = j;
        }
    }
    fprintf(stderr,
        "[moss_attn] call=%d layer=%d head=%d q_row=%d pre_softmax nvalid=%d sum=%.9g l2=%.9g min=%.9g max=%.9g argmax_j=%d\n",
        call_idx,
        layer_idx,
        head,
        query_row,
        nvalid,
        sum,
        (nvalid > 0 ? sqrt(sq) : 0.0),
        (double)(first ? 0.0f : mn),
        (double)(first ? 0.0f : mx),
        argmax_j);
    fflush(stderr);
}

int moss_gpt2_forward(
    const moss_gpt2_stack_t *stk,
    const moss_run_config_t *cfg,
    float *hidden,
    int S,
    const unsigned char *attn_mask,
    float *scratch,
    size_t scratch_elems,
    const char *dbg_stack_id
) {
    static int call_counter = 0;
    const int D = cfg->n_embd;
    const int H = cfg->n_head;
    const int Dh = D / H;
    const int I = cfg->n_inner;
    const int nL = stk->n_layer;
    if (Dh * H != D) return -1;

    size_t need = (size_t)S * (size_t)D * 6u + (size_t)S + (size_t)S * (size_t)I + (size_t)S * (size_t)Dh * 2u;
    if (scratch_elems < need) return -2;

    float *x_ln = scratch;
    float *q = x_ln + (size_t)S * D;
    float *k = q + (size_t)S * D;
    float *v = k + (size_t)S * D;
    float *attn_out = v + (size_t)S * D;
    float *rowscores = attn_out + (size_t)S * D;
    float *mlp_h = rowscores + (size_t)S;
    float *cos = mlp_h + (size_t)S * I;
    float *sin = cos + (size_t)S * Dh;

    int *pos_ids = (int *)malloc((size_t)S * sizeof(int));
    if (!pos_ids) return -3;
    {
        int cum = 0;
        for (int i = 0; i < S; i++) {
            cum += attn_mask[i] ? 1 : 0;
            pos_ids[i] = (attn_mask[i] ? (cum - 1) : 0);
        }
    }

    float *rowqkv = (float *)malloc((size_t)(3 * D) * sizeof(float));
    if (!rowqkv) {
        free(pos_ids);
        return -4;
    }

    apply_attn_mask_scale_rows(hidden, attn_mask, S, D);

    int call_idx = ++call_counter;
    const char *stack = (dbg_stack_id && dbg_stack_id[0]) ? dbg_stack_id : "gpt2";
    int dbg_enabled = 0;
    int dbg_max_calls = 1;
    int dbg_verbose = 0;
    {
        const char *e = getenv("MOSS_DEBUG_LAYER_STATS");
        if (e && e[0] && strcmp(e, "0") != 0) dbg_enabled = 1;
        const char *m = getenv("MOSS_DEBUG_LAYER_CALLS");
        if (m && m[0]) {
            int v = atoi(m);
            if (v > 0) dbg_max_calls = v;
        }
        const char *v = getenv("MOSS_DEBUG_LAYER_STATS_VERBOSE");
        if (v && v[0] && strcmp(v, "0") != 0) dbg_verbose = 1;
    }
    int dbg_this_call = dbg_enabled && call_idx <= dbg_max_calls;
    int dbg_block0_components = 0;
    {
        const char *e = getenv("MOSS_DEBUG_BLOCK0_COMPONENTS");
        if (e && e[0] && strcmp(e, "0") != 0) dbg_block0_components = 1;
    }
    int dbg_attn_layer0 = 0;
    {
        const char *e = getenv("MOSS_DEBUG_ATTN_LAYER0");
        if (e && e[0] && strcmp(e, "0") != 0) dbg_attn_layer0 = 1;
    }
    int last_row = S - 1;
    while (last_row > 0 && !attn_mask[last_row]) last_row--;
    if (dbg_attn_layer0 && call_idx == 1 && last_row >= 0) {
        fprintf(stderr, "[moss_attn] call=%d pos_id[last_row=%d]=%d\n", call_idx, last_row, pos_ids[last_row]);
        fflush(stderr);
    }
    if (dbg_this_call) {
        fprintf(stderr,
            "[moss_layer] stack=%s begin call=%d n_layer=%d S=%d D=%d I=%d last_row=%d\n",
            stack,
            call_idx,
            nL,
            S,
            D,
            I,
            last_row);
        fflush(stderr);
        log_row_stats(stack, call_idx, -1, "input", hidden + last_row * D, D);
    }

    for (int li = 0; li < nL; li++) {
        const moss_gpt2_layer_w_t *w = &stk->layers[li];
        for (int s = 0; s < S; s++) {
            moss_layernorm_bf16(
                x_ln + s * D,
                hidden + s * D,
                w->ln1w,
                w->ln1b,
                D,
                cfg->layer_norm_epsilon
            );
        }
        if (dbg_block0_components && call_idx == 1 && li == 0) {
            log_row_stats(stack, call_idx, li, "comp_ln1_out", x_ln + last_row * D, D);
        }
        if (dbg_this_call && dbg_verbose && last_row >= 0) {
            log_row_stats(stack, call_idx, li, "ln1_out", x_ln + last_row * D, D);
        }

        for (int s = 0; s < S; s++) {
            moss_gemv_bf16_nt_bias(rowqkv, w->c_attn_w, w->c_attn_b, x_ln + s * D, 3 * D, D);
            memcpy(q + s * D, rowqkv, (size_t)D * sizeof(float));
            memcpy(k + s * D, rowqkv + D, (size_t)D * sizeof(float));
            memcpy(v + s * D, rowqkv + 2 * D, (size_t)D * sizeof(float));
        }
        if (dbg_block0_components && call_idx == 1 && li == 0) {
            log_row_stats(stack, call_idx, li, "comp_q_pre_rope", q + last_row * D, D);
            log_row_stats(stack, call_idx, li, "comp_k_pre_rope", k + last_row * D, D);
            log_row_stats(stack, call_idx, li, "comp_v", v + last_row * D, D);
        }

        moss_rope_cos_sin(cos, sin, pos_ids, S, Dh, cfg->rope_base);
        moss_apply_rope_inplace(q, cos, sin, S, H, Dh);
        moss_apply_rope_inplace(k, cos, sin, S, H, Dh);
        if (dbg_block0_components && call_idx == 1 && li == 0) {
            log_row_stats(stack, call_idx, li, "comp_q_post_rope", q + last_row * D, D);
            log_row_stats(stack, call_idx, li, "comp_k_post_rope", k + last_row * D, D);
        }

        const float scale = 1.0f / sqrtf((float)Dh);
        memset(attn_out, 0, (size_t)S * D * sizeof(float));

        for (int h = 0; h < H; h++) {
            for (int i = 0; i < S; i++) {
                if (!attn_mask[i]) continue;
                for (int j = 0; j < S; j++) {
                    if (!attn_mask[j] || j > i) rowscores[j] = -1e9f;
                    else {
                        const float *qi = q + i * D + h * Dh;
                        const float *kj = k + j * D + h * Dh;
                        float dot = 0.0f;
                        for (int d = 0; d < Dh; d++) dot += qi[d] * kj[d];
                        rowscores[j] = dot * scale;
                    }
                }
                if (dbg_attn_layer0 && call_idx == 1 && li == 0 && h == 0 && i == last_row) {
                    moss_debug_log_attn_scores_pre_softmax(call_idx, li, h, i, S, rowscores, attn_mask);
                }
                moss_softmax_rows(rowscores, 1, S);
                float *oi = attn_out + i * D + h * Dh;
                memset(oi, 0, (size_t)Dh * sizeof(float));
                for (int j = 0; j < S; j++) {
                    float p = rowscores[j];
                    if (p <= 0.0f) continue;
                    const float *vj = v + j * D + h * Dh;
                    for (int d = 0; d < Dh; d++) oi[d] += p * vj[d];
                }
            }
        }
        if (dbg_block0_components && call_idx == 1 && li == 0) {
            log_row_stats(stack, call_idx, li, "comp_attn_out_pre_proj", attn_out + last_row * D, D);
        }
        if (dbg_this_call && dbg_verbose && last_row >= 0) {
            log_row_stats(stack, call_idx, li, "attn_out_pre_proj", attn_out + last_row * D, D);
        }

        memset(mlp_h, 0, (size_t)S * D * sizeof(float));
        for (int s = 0; s < S; s++) {
            moss_gemv_bf16_nt_bias(mlp_h + s * D, w->c_proj_w, w->c_proj_b, attn_out + s * D, D, D);
            moss_vec_add(hidden + s * D, mlp_h + s * D, D);
        }
        if (dbg_block0_components && call_idx == 1 && li == 0) {
            log_row_stats(stack, call_idx, li, "comp_c_proj_out", mlp_h + last_row * D, D);
            log_row_stats(stack, call_idx, li, "comp_post_attn_res", hidden + last_row * D, D);
        }
        apply_attn_mask_scale_rows(hidden, attn_mask, S, D);
        if (dbg_this_call) log_row_stats(stack, call_idx, li, "post_attn_res", hidden + last_row * D, D);

        for (int s = 0; s < S; s++) {
            moss_layernorm_bf16(
                x_ln + s * D,
                hidden + s * D,
                w->ln2w,
                w->ln2b,
                D,
                cfg->layer_norm_epsilon
            );
        }
        if (dbg_block0_components && call_idx == 1 && li == 0) {
            log_row_stats(stack, call_idx, li, "comp_ln2_out", x_ln + last_row * D, D);
        }
        if (dbg_this_call && dbg_verbose && last_row >= 0) {
            log_row_stats(stack, call_idx, li, "ln2_out", x_ln + last_row * D, D);
        }

        for (int s = 0; s < S; s++) {
            moss_gemv_bf16_nt_bias(mlp_h + s * I, w->fc_in_w, w->fc_in_b, x_ln + s * D, I, D);
            moss_gelu_new_inplace(mlp_h + s * I, I);
        }
        if (dbg_block0_components && call_idx == 1 && li == 0) {
            log_row_stats(stack, call_idx, li, "comp_mlp_fc_in_gelu", mlp_h + last_row * I, I);
        }
        if (dbg_this_call && dbg_verbose && last_row >= 0) {
            log_row_stats(stack, call_idx, li, "mlp_fc_in_gelu", mlp_h + last_row * I, I);
        }
        memset(x_ln, 0, (size_t)S * D * sizeof(float));
        for (int s = 0; s < S; s++) {
            moss_gemv_bf16_nt_bias(x_ln + s * D, w->fc_out_w, w->fc_out_b, mlp_h + s * I, D, I);
            moss_vec_add(hidden + s * D, x_ln + s * D, D);
        }
        if (dbg_block0_components && call_idx == 1 && li == 0) {
            log_row_stats(stack, call_idx, li, "comp_mlp_fc_out", x_ln + last_row * D, D);
            log_row_stats(stack, call_idx, li, "comp_post_mlp_res", hidden + last_row * D, D);
        }
        if (dbg_this_call && dbg_verbose && last_row >= 0) {
            log_row_stats(stack, call_idx, li, "mlp_fc_out_pre_add", x_ln + last_row * D, D);
        }
        apply_attn_mask_scale_rows(hidden, attn_mask, S, D);
        if (dbg_this_call) log_row_stats(stack, call_idx, li, "post_mlp_res", hidden + last_row * D, D);
    }

    for (int s = 0; s < S; s++) {
        moss_layernorm_bf16(
            x_ln + s * D,
            hidden + s * D,
            stk->ln_f_w,
            stk->ln_f_b,
            D,
            cfg->layer_norm_epsilon
        );
        memcpy(hidden + s * D, x_ln + s * D, (size_t)D * sizeof(float));
    }
    apply_attn_mask_scale_rows(hidden, attn_mask, S, D);
    if (dbg_this_call) log_row_stats(stack, call_idx, nL, "output_ln_f", hidden + last_row * D, D);

    free(rowqkv);
    free(pos_ids);
    return 0;
}

int moss_gpt2_kv_cache_init(
    moss_gpt2_kv_cache_t *cache,
    int n_layer,
    int max_seq,
    int D
) {
    if (!cache || n_layer <= 0 || max_seq <= 0 || D <= 0) return -1;
    memset(cache, 0, sizeof(*cache));
    size_t n = (size_t)n_layer * (size_t)max_seq * (size_t)D;
    cache->k_cache = (float *)malloc(n * sizeof(float));
    cache->v_cache = (float *)malloc(n * sizeof(float));
    if (!cache->k_cache || !cache->v_cache) {
        free(cache->k_cache);
        free(cache->v_cache);
        memset(cache, 0, sizeof(*cache));
        return -2;
    }
    cache->n_layer = n_layer;
    cache->max_seq = max_seq;
    cache->D = D;
    cache->cur_seq = 0;
    return 0;
}

void moss_gpt2_kv_cache_reset(moss_gpt2_kv_cache_t *cache) {
    if (!cache) return;
    cache->cur_seq = 0;
}

void moss_gpt2_kv_cache_free(moss_gpt2_kv_cache_t *cache) {
    if (!cache) return;
    free(cache->k_cache);
    free(cache->v_cache);
    memset(cache, 0, sizeof(*cache));
}

int moss_gpt2_forward_step(
    const moss_gpt2_stack_t *stk,
    const moss_run_config_t *cfg,
    const float *input_embed,
    moss_gpt2_kv_cache_t *cache,
    float *out_hidden
) {
    if (!stk || !cfg || !input_embed || !cache || !out_hidden) return -1;
    const int D = cfg->n_embd;
    const int H = cfg->n_head;
    const int Dh = D / H;
    const int I = cfg->n_inner;
    if (Dh * H != D) return -2;
    if (cache->D != D || cache->n_layer != stk->n_layer || cache->cur_seq >= cache->max_seq) return -3;

    const int pos = cache->cur_seq;
    float *x = (float *)malloc((size_t)D * sizeof(float));
    float *x_ln = (float *)malloc((size_t)D * sizeof(float));
    float *q = (float *)malloc((size_t)D * sizeof(float));
    float *k = (float *)malloc((size_t)D * sizeof(float));
    float *v = (float *)malloc((size_t)D * sizeof(float));
    float *attn_out = (float *)malloc((size_t)D * sizeof(float));
    float *rowscores = (float *)malloc((size_t)(pos + 1) * sizeof(float));
    float *mlp_h = (float *)malloc((size_t)I * sizeof(float));
    float *proj = (float *)malloc((size_t)D * sizeof(float));
    float *cos = (float *)malloc((size_t)Dh * sizeof(float));
    float *sin = (float *)malloc((size_t)Dh * sizeof(float));
    int pos_id[1] = {pos};
    if (!x || !x_ln || !q || !k || !v || !attn_out || !rowscores || !mlp_h || !proj || !cos || !sin) {
        free(x); free(x_ln); free(q); free(k); free(v); free(attn_out);
        free(rowscores); free(mlp_h); free(proj); free(cos); free(sin);
        return -4;
    }
    memcpy(x, input_embed, (size_t)D * sizeof(float));

    for (int li = 0; li < stk->n_layer; li++) {
        const moss_gpt2_layer_w_t *w = &stk->layers[li];
        moss_layernorm_bf16(x_ln, x, w->ln1w, w->ln1b, D, cfg->layer_norm_epsilon);
        float rowqkv[3 * 2048];
        if (3 * D > (int)(sizeof(rowqkv) / sizeof(rowqkv[0]))) {
            free(x); free(x_ln); free(q); free(k); free(v); free(attn_out);
            free(rowscores); free(mlp_h); free(proj); free(cos); free(sin);
            return -5;
        }
        moss_gemv_bf16_nt_bias(rowqkv, w->c_attn_w, w->c_attn_b, x_ln, 3 * D, D);
        memcpy(q, rowqkv, (size_t)D * sizeof(float));
        memcpy(k, rowqkv + D, (size_t)D * sizeof(float));
        memcpy(v, rowqkv + 2 * D, (size_t)D * sizeof(float));
        moss_rope_cos_sin(cos, sin, pos_id, 1, Dh, cfg->rope_base);
        moss_apply_rope_inplace(q, cos, sin, 1, H, Dh);
        moss_apply_rope_inplace(k, cos, sin, 1, H, Dh);

        float *kc = cache->k_cache + ((size_t)li * cache->max_seq + (size_t)pos) * D;
        float *vc = cache->v_cache + ((size_t)li * cache->max_seq + (size_t)pos) * D;
        memcpy(kc, k, (size_t)D * sizeof(float));
        memcpy(vc, v, (size_t)D * sizeof(float));

        memset(attn_out, 0, (size_t)D * sizeof(float));
        const float scale = 1.0f / sqrtf((float)Dh);
        for (int h = 0; h < H; h++) {
            const float *qh = q + h * Dh;
            for (int j = 0; j <= pos; j++) {
                const float *kj = cache->k_cache + ((size_t)li * cache->max_seq + (size_t)j) * D + h * Dh;
                float dot = 0.0f;
                for (int d = 0; d < Dh; d++) dot += qh[d] * kj[d];
                rowscores[j] = dot * scale;
            }
            moss_softmax_rows(rowscores, 1, pos + 1);
            float *oh = attn_out + h * Dh;
            for (int j = 0; j <= pos; j++) {
                float p = rowscores[j];
                if (p <= 0.0f) continue;
                const float *vj = cache->v_cache + ((size_t)li * cache->max_seq + (size_t)j) * D + h * Dh;
                for (int d = 0; d < Dh; d++) oh[d] += p * vj[d];
            }
        }

        moss_gemv_bf16_nt_bias(proj, w->c_proj_w, w->c_proj_b, attn_out, D, D);
        moss_vec_add(x, proj, D);
        moss_layernorm_bf16(x_ln, x, w->ln2w, w->ln2b, D, cfg->layer_norm_epsilon);
        moss_gemv_bf16_nt_bias(mlp_h, w->fc_in_w, w->fc_in_b, x_ln, I, D);
        moss_gelu_new_inplace(mlp_h, I);
        moss_gemv_bf16_nt_bias(proj, w->fc_out_w, w->fc_out_b, mlp_h, D, I);
        moss_vec_add(x, proj, D);
    }

    moss_layernorm_bf16(out_hidden, x, stk->ln_f_w, stk->ln_f_b, D, cfg->layer_norm_epsilon);
    cache->cur_seq++;
    free(x); free(x_ln); free(q); free(k); free(v); free(attn_out);
    free(rowscores); free(mlp_h); free(proj); free(cos); free(sin);
    return 0;
}
