#include "moss_audio_tok_internal.h"
#include "moss_kernel.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

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
    if (moss_audio_tok_get_f32_tensor(tok->sf, "quantizer.output_proj.parametrizations.weight.original0", &q_out_g, &n) != 0 || n != 768) return -1;
    if (moss_audio_tok_get_f32_tensor(tok->sf, "quantizer.output_proj.parametrizations.weight.original1", &q_out_v, &n) != 0 || n != 768 * 512) return -1;
    if (moss_audio_tok_get_f32_tensor(tok->sf, "quantizer.output_proj.bias", &q_out_b, &n) != 0 || n != 768) return -1;

    float *q_out_w = (float *)malloc((size_t)768 * 512 * sizeof(float));
    if (!q_out_w) return -1;
    moss_build_weightnorm_matrix_f32(q_out_w, q_out_g, q_out_v, 768, 512);

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
        if (moss_audio_tok_get_f32_tensor(tok->sf, k_cb, &cbw, &n) != 0 || n != cb * 8) goto fail;
        if (moss_audio_tok_get_f32_tensor(tok->sf, k_og, &og, &n) != 0 || n != 512) goto fail;
        if (moss_audio_tok_get_f32_tensor(tok->sf, k_ov, &ov, &n) != 0 || n != 512 * 8) goto fail;
        if (moss_audio_tok_get_f32_tensor(tok->sf, k_ob, &ob, &n) != 0 || n != 512) goto fail;
        q_q_out_w[q] = (float *)malloc((size_t)512 * 8 * sizeof(float));
        if (!q_q_out_w[q]) goto fail;
        moss_build_weightnorm_matrix_f32(q_q_out_w[q], og, ov, 512, 8);
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
            moss_matvec_bias_f32(t512, q_q_out_w[q], q_q_out_b[q], cbv, 512, 8);
            for (int i = 0; i < 512; i++) z512[i] += t512[i];
        }

        moss_matvec_bias_f32(z768, q_out_w, q_out_b, z512, 768, 512);
        memcpy(z768_tm + (size_t)t * 768, z768, 768 * sizeof(float));
    }

    float *x0 = z768_tm;
    int T0 = frames, D0 = 768;
    int T1 = 0, D1 = 0, T2 = 0, D2 = 0, T3 = 0, D3 = 0, T4 = 0, D4 = 0;
    int T5 = 0, D5 = 0, T6 = 0, D6 = 0, T7 = 0, D7 = 0, T8 = 0, D8 = 0;
    if (moss_audio_tok_patched_decode_tm(x0, T0, D0, 4, &x1, &T1, &D1) != 0) goto fail;
    if (moss_audio_tok_transformer_module_tm(tok, "decoder", 1, 4, D1, 768, 500, x1, T1, &x2) != 0) goto fail;
    T2 = T1; D2 = 768;
    if (moss_audio_tok_patched_decode_tm(x2, T2, D2, 2, &x3, &T3, &D3) != 0) goto fail;
    if (moss_audio_tok_transformer_module_tm(tok, "decoder", 3, 2, D3, 768, 400, x3, T3, &x4) != 0) goto fail;
    T4 = T3; D4 = 768;
    if (moss_audio_tok_patched_decode_tm(x4, T4, D4, 2, &x5, &T5, &D5) != 0) goto fail;
    if (moss_audio_tok_transformer_module_tm(tok, "decoder", 5, 2, D5, 768, 300, x5, T5, &x6) != 0) goto fail;
    T6 = T5; D6 = 768;
    if (moss_audio_tok_patched_decode_tm(x6, T6, D6, 2, &x7, &T7, &D7) != 0) goto fail;
    if (moss_audio_tok_transformer_module_tm(tok, "decoder", 7, 4, D7, 240, 200, x7, T7, &x8) != 0) goto fail;
    T8 = T7; D8 = 240;

    int Ti = 0, Di = 0;
    if (moss_audio_tok_patched_decode_tm(x8, T8, D8, 240, &wav_interleave, &Ti, &Di) != 0) goto fail;
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
