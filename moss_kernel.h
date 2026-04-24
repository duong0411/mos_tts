#ifndef MOSS_KERNEL_H
#define MOSS_KERNEL_H

#include <stdint.h>

float moss_bf16_to_f32(uint16_t bf16);

/* y += W^T x with W row-major [rows, cols], BF16 weights, x/y FP32 */
void moss_gemv_bf16_nt(float *y, const uint16_t *W, const float *x, int rows, int cols);

void moss_vec_add(float *y, const float *x, int n);
void moss_vec_scale(float *y, float s, int n);

void moss_layernorm_forward(
    float *out,
    const float *x,
    const float *weight,
    const float *bias,
    int n,
    float eps
);

void moss_layernorm_bf16(
    float *out,
    const float *x,
    const uint16_t *w_bf16,
    const uint16_t *b_bf16,
    int n,
    float eps
);

/* y = W @ x + b (overwrite y) */
void moss_gemv_bf16_nt_bias(float *y, const uint16_t *W, const uint16_t *b_bf16, const float *x, int rows, int cols);

void moss_gelu_new_inplace(float *x, int n);

/* Softmax over last dimension: x [rows * cols], softmax each row of length cols */
void moss_softmax_rows(float *x, int rows, int cols);

/* RoPE cos/sin length head_dim for positions [seq] */
void moss_rope_cos_sin(
    float *cos, float *sin,
    const int *position_ids,
    int seq,
    int head_dim,
    float rope_base
);

void moss_apply_rope_inplace(
    float *q_or_k,
    const float *cos,
    const float *sin,
    int seq,
    int n_heads,
    int head_dim
);

#endif
