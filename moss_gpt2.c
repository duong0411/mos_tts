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

static void log_row_stats(const char *tag, int call_idx, int layer_idx, const float *row, int D) {
    if (!tag || !row || D <= 0) return;
    double sum = 0.0;
    double sq = 0.0;
    float mn = row[0], mx = row[0];
    for (int i = 0; i < D; i++) {
        float v = row[i];
        sum += (double)v;
        sq += (double)v * (double)v;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    fprintf(stderr,
        "[moss_layer] call=%d layer=%d stage=%s sum=%.9g l2=%.9g min=%.9g max=%.9g\n",
        call_idx, layer_idx, tag, sum, sqrt(sq), (double)mn, (double)mx);
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
    size_t scratch_elems
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
    int dbg_enabled = 0;
    int dbg_max_calls = 1;
    {
        const char *e = getenv("MOSS_DEBUG_LAYER_STATS");
        if (e && e[0] && strcmp(e, "0") != 0) dbg_enabled = 1;
        const char *m = getenv("MOSS_DEBUG_LAYER_CALLS");
        if (m && m[0]) {
            int v = atoi(m);
            if (v > 0) dbg_max_calls = v;
        }
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
        fprintf(stderr, "[moss_layer] begin call=%d n_layer=%d S=%d D=%d last_row=%d\n", call_idx, nL, S, D, last_row);
        log_row_stats("input", call_idx, -1, hidden + last_row * D, D);
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
            log_row_stats("comp_ln1_out", call_idx, li, x_ln + last_row * D, D);
        }

        for (int s = 0; s < S; s++) {
            moss_gemv_bf16_nt_bias(rowqkv, w->c_attn_w, w->c_attn_b, x_ln + s * D, 3 * D, D);
            memcpy(q + s * D, rowqkv, (size_t)D * sizeof(float));
            memcpy(k + s * D, rowqkv + D, (size_t)D * sizeof(float));
            memcpy(v + s * D, rowqkv + 2 * D, (size_t)D * sizeof(float));
        }
        if (dbg_block0_components && call_idx == 1 && li == 0) {
            log_row_stats("comp_q_pre_rope", call_idx, li, q + last_row * D, D);
            log_row_stats("comp_k_pre_rope", call_idx, li, k + last_row * D, D);
            log_row_stats("comp_v", call_idx, li, v + last_row * D, D);
        }

        moss_rope_cos_sin(cos, sin, pos_ids, S, Dh, cfg->rope_base);
        moss_apply_rope_inplace(q, cos, sin, S, H, Dh);
        moss_apply_rope_inplace(k, cos, sin, S, H, Dh);
        if (dbg_block0_components && call_idx == 1 && li == 0) {
            log_row_stats("comp_q_post_rope", call_idx, li, q + last_row * D, D);
            log_row_stats("comp_k_post_rope", call_idx, li, k + last_row * D, D);
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
            log_row_stats("comp_attn_out_pre_proj", call_idx, li, attn_out + last_row * D, D);
        }

        memset(mlp_h, 0, (size_t)S * D * sizeof(float));
        for (int s = 0; s < S; s++) {
            moss_gemv_bf16_nt_bias(mlp_h + s * D, w->c_proj_w, w->c_proj_b, attn_out + s * D, D, D);
            moss_vec_add(hidden + s * D, mlp_h + s * D, D);
        }
        if (dbg_block0_components && call_idx == 1 && li == 0) {
            log_row_stats("comp_c_proj_out", call_idx, li, mlp_h + last_row * D, D);
            log_row_stats("comp_post_attn_res", call_idx, li, hidden + last_row * D, D);
        }
        apply_attn_mask_scale_rows(hidden, attn_mask, S, D);
        if (dbg_this_call) log_row_stats("post_attn_res", call_idx, li, hidden + last_row * D, D);

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
            log_row_stats("comp_ln2_out", call_idx, li, x_ln + last_row * D, D);
        }

        for (int s = 0; s < S; s++) {
            moss_gemv_bf16_nt_bias(mlp_h + s * I, w->fc_in_w, w->fc_in_b, x_ln + s * D, I, D);
            moss_gelu_new_inplace(mlp_h + s * I, I);
        }
        if (dbg_block0_components && call_idx == 1 && li == 0) {
            log_row_stats("comp_mlp_fc_in_gelu", call_idx, li, mlp_h + last_row * I, I);
        }
        memset(x_ln, 0, (size_t)S * D * sizeof(float));
        for (int s = 0; s < S; s++) {
            moss_gemv_bf16_nt_bias(x_ln + s * D, w->fc_out_w, w->fc_out_b, mlp_h + s * I, D, I);
            moss_vec_add(hidden + s * D, x_ln + s * D, D);
        }
        if (dbg_block0_components && call_idx == 1 && li == 0) {
            log_row_stats("comp_mlp_fc_out", call_idx, li, x_ln + last_row * D, D);
            log_row_stats("comp_post_mlp_res", call_idx, li, hidden + last_row * D, D);
        }
        apply_attn_mask_scale_rows(hidden, attn_mask, S, D);
        if (dbg_this_call) log_row_stats("post_mlp_res", call_idx, li, hidden + last_row * D, D);
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
    if (dbg_this_call) log_row_stats("output_ln_f", call_idx, nL, hidden + last_row * D, D);

    free(rowqkv);
    free(pos_ids);
    return 0;
}
