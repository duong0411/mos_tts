#include "moss_gpt2.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "moss_kernel.h"

static void zero_row(float *row, int D) {
    memset(row, 0, (size_t)D * sizeof(float));
}

static void apply_attn_mask_scale_rows(float *hidden, const unsigned char *mask, int S, int D) {
    for (int s = 0; s < S; s++) {
        if (!mask[s]) zero_row(hidden + s * D, D);
    }
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

        for (int s = 0; s < S; s++) {
            moss_gemv_bf16_nt_bias(rowqkv, w->c_attn_w, w->c_attn_b, x_ln + s * D, 3 * D, D);
            memcpy(q + s * D, rowqkv, (size_t)D * sizeof(float));
            memcpy(k + s * D, rowqkv + D, (size_t)D * sizeof(float));
            memcpy(v + s * D, rowqkv + 2 * D, (size_t)D * sizeof(float));
        }

        moss_rope_cos_sin(cos, sin, pos_ids, S, Dh, cfg->rope_base);
        moss_apply_rope_inplace(q, cos, sin, S, H, Dh);
        moss_apply_rope_inplace(k, cos, sin, S, H, Dh);

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

        memset(mlp_h, 0, (size_t)S * D * sizeof(float));
        for (int s = 0; s < S; s++) {
            moss_gemv_bf16_nt_bias(mlp_h + s * D, w->c_proj_w, w->c_proj_b, attn_out + s * D, D, D);
            moss_vec_add(hidden + s * D, mlp_h + s * D, D);
        }
        apply_attn_mask_scale_rows(hidden, attn_mask, S, D);

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

        for (int s = 0; s < S; s++) {
            moss_gemv_bf16_nt_bias(mlp_h + s * I, w->fc_in_w, w->fc_in_b, x_ln + s * D, I, D);
            moss_gelu_new_inplace(mlp_h + s * I, I);
        }
        memset(x_ln, 0, (size_t)S * D * sizeof(float));
        for (int s = 0; s < S; s++) {
            moss_gemv_bf16_nt_bias(x_ln + s * D, w->fc_out_w, w->fc_out_b, mlp_h + s * I, D, I);
            moss_vec_add(hidden + s * D, x_ln + s * D, D);
        }
        apply_attn_mask_scale_rows(hidden, attn_mask, S, D);
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

    free(rowqkv);
    free(pos_ids);
    return 0;
}
