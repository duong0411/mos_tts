#include "moss_kernel.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef MOSS_USE_CBLAS
#include <cblas.h>
#endif

float moss_bf16_to_f32(uint16_t bf16) {
    union {
        uint32_t u;
        float f;
    } v;
    v.u = ((uint32_t)bf16) << 16;
    return v.f;
}

static void bf16_row_to_f32(float *dst, const uint16_t *src, int n) {
    for (int i = 0; i < n; i++) dst[i] = moss_bf16_to_f32(src[i]);
}

static int moss_use_pure_gemv(void) {
    static int initialized = 0;
    static int use_pure = 0;
    if (!initialized) {
        const char *v = getenv("MOSS_FORCE_PURE_GEMV");
        use_pure = (v && v[0] && strcmp(v, "0") != 0) ? 1 : 0;
        initialized = 1;
    }
    return use_pure;
}

void moss_layernorm_bf16(
    float *out,
    const float *x,
    const uint16_t *w_bf16,
    const uint16_t *b_bf16,
    int n,
    float eps
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
    float inv = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < n; i++) {
        float w = moss_bf16_to_f32(w_bf16[i]);
        float b = moss_bf16_to_f32(b_bf16[i]);
        out[i] = (x[i] - mean) * inv * w + b;
    }
}

void moss_gemv_bf16_nt(float *y, const uint16_t *W, const float *x, int rows, int cols) {
    if (moss_use_pure_gemv()) {
        for (int r = 0; r < rows; r++) {
            const uint16_t *wr = W + (size_t)r * cols;
            float s = 0.0f;
            for (int c = 0; c < cols; c++) s += moss_bf16_to_f32(wr[c]) * x[c];
            y[r] += s;
        }
        return;
    }
#ifdef MOSS_USE_CBLAS
    float *tmp = (float *)malloc((size_t)cols * sizeof(float));
    if (!tmp) {
        for (int r = 0; r < rows; r++) {
            const uint16_t *wr = W + (size_t)r * cols;
            float s = 0.0f;
            for (int c = 0; c < cols; c++) s += moss_bf16_to_f32(wr[c]) * x[c];
            y[r] += s;
        }
        return;
    }
    for (int r = 0; r < rows; r++) {
        const uint16_t *wr = W + (size_t)r * cols;
        bf16_row_to_f32(tmp, wr, cols);
        y[r] += cblas_sdot(cols, tmp, 1, x, 1);
    }
    free(tmp);
#else
    for (int r = 0; r < rows; r++) {
        const uint16_t *wr = W + (size_t)r * cols;
        float s = 0.0f;
        for (int c = 0; c < cols; c++) s += moss_bf16_to_f32(wr[c]) * x[c];
        y[r] += s;
    }
#endif
}

void moss_gemv_bf16_nt_bias(float *y, const uint16_t *W, const uint16_t *b_bf16, const float *x, int rows, int cols) {
    if (moss_use_pure_gemv()) {
        for (int r = 0; r < rows; r++) {
            const uint16_t *wr = W + (size_t)r * cols;
            float s = moss_bf16_to_f32(b_bf16[r]);
            for (int c = 0; c < cols; c++) s += moss_bf16_to_f32(wr[c]) * x[c];
            y[r] = s;
        }
        return;
    }
#ifdef MOSS_USE_CBLAS
    float *tmp = (float *)malloc((size_t)cols * sizeof(float));
    if (!tmp) {
        for (int r = 0; r < rows; r++) {
            const uint16_t *wr = W + (size_t)r * cols;
            float s = moss_bf16_to_f32(b_bf16[r]);
            for (int c = 0; c < cols; c++) s += moss_bf16_to_f32(wr[c]) * x[c];
            y[r] = s;
        }
        return;
    }
    for (int r = 0; r < rows; r++) {
        const uint16_t *wr = W + (size_t)r * cols;
        bf16_row_to_f32(tmp, wr, cols);
        y[r] = moss_bf16_to_f32(b_bf16[r]) + cblas_sdot(cols, tmp, 1, x, 1);
    }
    free(tmp);
#else
    for (int r = 0; r < rows; r++) {
        const uint16_t *wr = W + (size_t)r * cols;
        float s = moss_bf16_to_f32(b_bf16[r]);
        for (int c = 0; c < cols; c++) s += moss_bf16_to_f32(wr[c]) * x[c];
        y[r] = s;
    }
#endif
}

void moss_vec_add(float *y, const float *x, int n) {
#ifdef MOSS_USE_CBLAS
    cblas_saxpy(n, 1.0f, x, 1, y, 1);
#else
    for (int i = 0; i < n; i++) y[i] += x[i];
#endif
}

void moss_vec_scale(float *y, float s, int n) {
#ifdef MOSS_USE_CBLAS
    cblas_sscal(n, s, y, 1);
#else
    for (int i = 0; i < n; i++) y[i] *= s;
#endif
}

void moss_layernorm_forward(
    float *out,
    const float *x,
    const float *weight,
    const float *bias,
    int n,
    float eps
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
    float inv = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < n; i++) out[i] = (x[i] - mean) * inv * weight[i] + bias[i];
}

void moss_gelu_new_inplace(float *x, int n) {
    const float c = 0.7978845608028654f;
    for (int i = 0; i < n; i++) {
        float v = x[i];
        float v3 = v * v * v;
        float t = tanhf(c * (v + 0.044715f * v3));
        x[i] = 0.5f * v * (1.0f + t);
    }
}

void moss_softmax_rows(float *x, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float *row = x + r * cols;
        float m = row[0];
        for (int j = 1; j < cols; j++) if (row[j] > m) m = row[j];
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            row[j] = expf(row[j] - m);
            sum += row[j];
        }
        float inv = 1.0f / sum;
        for (int j = 0; j < cols; j++) row[j] *= inv;
    }
}

void moss_rope_cos_sin(
    float *cos, float *sin,
    const int *position_ids,
    int seq,
    int head_dim,
    float rope_base
) {
    int half = head_dim / 2;
    float *inv = (float *)malloc((size_t)half * sizeof(float));
    for (int i = 0; i < half; i++) inv[i] = 1.0f / powf(rope_base, (float)(2 * i) / (float)head_dim);
    for (int s = 0; s < seq; s++) {
        int pos = position_ids[s];
        float *cs = cos + s * head_dim;
        float *sn = sin + s * head_dim;
        for (int i = 0; i < half; i++) {
            float ang = (float)pos * inv[i];
            float c = cosf(ang);
            float si = sinf(ang);
            cs[2 * i] = c;
            cs[2 * i + 1] = c;
            sn[2 * i] = si;
            sn[2 * i + 1] = si;
        }
    }
    free(inv);
}

void moss_apply_rope_inplace(
    float *q_or_k,
    const float *cos,
    const float *sin,
    int seq,
    int n_heads,
    int head_dim
) {
    int stride = n_heads * head_dim;
    for (int s = 0; s < seq; s++) {
        const float *cs = cos + s * head_dim;
        const float *sn = sin + s * head_dim;
        float *row = q_or_k + s * stride;
        for (int h = 0; h < n_heads; h++) {
            float *hk = row + h * head_dim;
            for (int i = 0; i < head_dim; i += 2) {
                float x0 = hk[i];
                float x1 = hk[i + 1];
                float rh0 = -x1;
                float rh1 = x0;
                hk[i] = x0 * cs[i] + rh0 * sn[i];
                hk[i + 1] = x1 * cs[i + 1] + rh1 * sn[i + 1];
            }
        }
    }
}

void moss_linear_f32(float *y, const float *W, const float *x, int out_dim, int in_dim) {
    for (int o = 0; o < out_dim; o++) {
        const float *wo = W + (size_t)o * in_dim;
        float s = 0.0f;
        for (int i = 0; i < in_dim; i++) s += wo[i] * x[i];
        y[o] = s;
    }
}

void moss_matvec_bias_f32(float *y, const float *W, const float *b, const float *x, int rows, int cols) {
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

void moss_build_weightnorm_matrix_f32(float *W_eff, const float *g, const float *v, int out_dim, int in_dim) {
    for (int o = 0; o < out_dim; o++) {
        const float *vo = v + (size_t)o * in_dim;
        float n2 = 0.0f;
        for (int i = 0; i < in_dim; i++) n2 += vo[i] * vo[i];
        float scale = g[o] / sqrtf(n2 + 1e-12f);
        float *wo = W_eff + (size_t)o * in_dim;
        for (int i = 0; i < in_dim; i++) wo[i] = vo[i] * scale;
    }
}

void moss_batch_mm_nt_f32(const float *X, const float *W, float *Y, int T, int out_dim, int in_dim) {
#ifdef MOSS_USE_CBLAS
    if (T > 0 && out_dim > 0 && in_dim > 0) {
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
        return;
    }
#endif
    for (int t = 0; t < T; t++) {
        moss_linear_f32(Y + (size_t)t * out_dim, W, X + (size_t)t * in_dim, out_dim, in_dim);
    }
}
