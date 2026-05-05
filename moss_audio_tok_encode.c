#include "moss_audio_tok_internal.h"
#include "moss_kernel.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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
    const int frames = n_samples / hop;
    if (frames <= 0) return -1;
    int *codes = (int *)malloc((size_t)frames * (size_t)nq * sizeof(int));
    if (!codes) return -1;

    int64_t _n = 0;
    const float *q_in_g = NULL, *q_in_v = NULL, *q_in_b = NULL;
    if (moss_audio_tok_get_f32_tensor(tok->sf, "quantizer.input_proj.parametrizations.weight.original0", &q_in_g, &_n) != 0 || _n != 512) { free(codes); return -1; }
    if (moss_audio_tok_get_f32_tensor(tok->sf, "quantizer.input_proj.parametrizations.weight.original1", &q_in_v, &_n) != 0 || _n != 512 * 768) { free(codes); return -1; }
    if (moss_audio_tok_get_f32_tensor(tok->sf, "quantizer.input_proj.bias", &q_in_b, &_n) != 0 || _n != 512) { free(codes); return -1; }
    float *q_in_w = (float *)malloc((size_t)512 * 768 * sizeof(float));
    if (!q_in_w) { free(codes); return -1; }
    moss_build_weightnorm_matrix_f32(q_in_w, q_in_g, q_in_v, 512, 768);

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
        if (moss_audio_tok_get_f32_tensor(tok->sf, k_g, &ig, &_n) != 0 || _n != 8) { err_step = 11; goto fail; }
        if (moss_audio_tok_get_f32_tensor(tok->sf, k_v, &iv, &_n) != 0 || _n != 8 * 512) { err_step = 12; goto fail; }
        if (moss_audio_tok_get_f32_tensor(tok->sf, k_b, &ib, &_n) != 0 || _n != 8) { err_step = 13; goto fail; }
        if (moss_audio_tok_get_f32_tensor(tok->sf, k_cb, &cbw, &_n) != 0 || _n != cb * 8) { err_step = 14; goto fail; }
        if (moss_audio_tok_get_f32_tensor(tok->sf, k_og, &og, &_n) != 0 || _n != 512) { err_step = 15; goto fail; }
        if (moss_audio_tok_get_f32_tensor(tok->sf, k_ov, &ov, &_n) != 0 || _n != 512 * 8) { err_step = 16; goto fail; }
        if (moss_audio_tok_get_f32_tensor(tok->sf, k_ob, &ob, &_n) != 0 || _n != 512) { err_step = 17; goto fail; }

        q_q_in_w[q] = (float *)malloc((size_t)8 * 512 * sizeof(float));
        q_q_out_w[q] = (float *)malloc((size_t)512 * 8 * sizeof(float));
        if (!q_q_in_w[q] || !q_q_out_w[q]) { err_step = 18; goto fail; }
        moss_build_weightnorm_matrix_f32(q_q_in_w[q], ig, iv, 8, 512);
        moss_build_weightnorm_matrix_f32(q_q_out_w[q], og, ov, 512, 8);
        q_q_in_b[q] = ib;
        q_q_codebook[q] = cbw;
        q_q_out_b[q] = ob;
    }

    float z768[768], z512[512], e8[8], zq512[512];
    int ch = tok->cfg.channels > 0 ? tok->cfg.channels : 2;
    int n_interleave = n_samples * ch;
    int full_patch = 240 * 2 * 2 * 2 * 4;
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
    if (moss_audio_tok_patched_encode_tm(x0, T0, D0, 240, &x1, &T1, &D1) != 0) { err_step = 22; goto fail; }
    if (moss_audio_tok_transformer_module_tm(tok, "encoder", 1, 4, D1, 384, 1600, x1, T1, &x2) != 0) { err_step = 23; goto fail; }
    T2 = T1; D2 = 384;
    if (moss_audio_tok_patched_encode_tm(x2, T2, D2, 2, &x3, &T3, &D3) != 0) { err_step = 24; goto fail; }
    if (moss_audio_tok_transformer_module_tm(tok, "encoder", 3, 2, D3, 384, 1200, x3, T3, &x4) != 0) { err_step = 25; goto fail; }
    T4 = T3; D4 = 384;
    if (moss_audio_tok_patched_encode_tm(x4, T4, D4, 2, &x5, &T5, &D5) != 0) { err_step = 26; goto fail; }
    if (moss_audio_tok_transformer_module_tm(tok, "encoder", 5, 2, D5, 384, 800, x5, T5, &x6) != 0) { err_step = 27; goto fail; }
    T6 = T5; D6 = 384;
    if (moss_audio_tok_patched_encode_tm(x6, T6, D6, 2, &x7, &T7, &D7) != 0) { err_step = 28; goto fail; }
    if (moss_audio_tok_transformer_module_tm(tok, "encoder", 7, 4, D7, 192, 500, x7, T7, &x8) != 0) { err_step = 29; goto fail; }
    T8 = T7; D8 = 192;
    int Tf = 0, Df = 0;
    if (moss_audio_tok_patched_encode_tm(x8, T8, D8, 4, &z768_tm, &Tf, &Df) != 0) { err_step = 30; goto fail; }
    if (Df != 768) { err_step = 31; goto fail; }
    if (Tf < frames) { err_step = 32; goto fail; }

    for (int t = 0; t < frames; t++) {
        memcpy(z768, z768_tm + (size_t)t * 768, 768 * sizeof(float));
        moss_matvec_bias_f32(z512, q_in_w, q_in_b, z768, 512, 768);
        for (int q = 0; q < nq; q++) {
            moss_matvec_bias_f32(e8, q_q_in_w[q], q_q_in_b[q], z512, 8, 512);
            float en = 0.0f;
            for (int i = 0; i < 8; i++) en += e8[i] * e8[i];
            en = sqrtf(en + 1e-12f);
            for (int i = 0; i < 8; i++) e8[i] /= en;

            int best = 0;
            float best_dot = -1e30f;
            for (int k = 0; k < cb; k++) {
                const float *cw = q_q_codebook[q] + (size_t)k * 8;
                float cn = 0.0f, dot = 0.0f;
                for (int d = 0; d < 8; d++) { cn += cw[d] * cw[d]; dot += e8[d] * cw[d]; }
                dot /= sqrtf(cn + 1e-12f);
                if (dot > best_dot) { best_dot = dot; best = k; }
            }
            codes[(size_t)t * nq + q] = best;
            const float *cbv = q_q_codebook[q] + (size_t)best * 8;
            moss_matvec_bias_f32(zq512, q_q_out_w[q], q_q_out_b[q], cbv, 512, 8);
            for (int i = 0; i < 512; i++) z512[i] -= zq512[i];
        }
    }

    for (int q = 0; q < nq; q++) { free(q_q_in_w[q]); free(q_q_out_w[q]); }
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
    for (int q = 0; q < nq; q++) { free(q_q_in_w[q]); free(q_q_out_w[q]); }
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
    if (dn > 0 && dst[dn - 1] != '/' && dst[dn - 1] != '\\') { dst[dn++] = sep; dst[dn] = '\0'; }
    snprintf(dst + dn, dst_sz - dn, "moss_aud_%lx.wav", (unsigned long)salt);
    return 0;
}

static int moss_run_ffmpeg_to_pcm_wav(const char *ffmpeg_exe, const char *in_path, const char *out_path, int dst_sr, int dst_ch) {
    char cmd[9216];
    int nw = snprintf(cmd, sizeof(cmd),
        "\"%s\" -nostdin -hide_banner -loglevel error -y -i \"%s\" -ac %d -ar %d -sample_fmt s16 -c:a pcm_s16le \"%s\"",
        ffmpeg_exe, in_path, dst_ch, dst_sr, out_path);
    if (nw <= 0 || (size_t)nw >= sizeof(cmd)) return -1;
    return system(cmd) == 0 ? 0 : -1;
}

static int moss_aud_read_riff_wave_and_encode(const moss_audio_tok_t *tok, FILE *f, int dst_sr, int dst_ch, int **out_codes, int *out_frames) {
    char riff[4], wave[4];
    unsigned int riff_size_unused = 0;
    if (fread(riff, 1, 4, f) != 4 || read_le32(f, &riff_size_unused) != 0 || fread(wave, 1, 4, f) != 4) return -1;
    if (memcmp(riff, "RIFF", 4) != 0 || memcmp(wave, "WAVE", 4) != 0) return -1;

    unsigned short audio_format = 1, channels = 1, bits = 16;
    unsigned int sample_rate = 0, data_size = 0;
    long data_pos = -1;
    while (!feof(f)) {
        char ck[4];
        unsigned int csz = 0;
        if (fread(ck, 1, 4, f) != 4) break;
        if (read_le32(f, &csz) != 0) break;
        if (memcmp(ck, "fmt ", 4) == 0) {
            read_le16(f, &audio_format); read_le16(f, &channels); read_le32(f, &sample_rate);
            unsigned int byte_rate = 0; unsigned short align = 0;
            read_le32(f, &byte_rate); read_le16(f, &align); read_le16(f, &bits);
            if (csz > 16) fseek(f, (long)(csz - 16), SEEK_CUR);
        } else if (memcmp(ck, "data", 4) == 0) {
            data_pos = ftell(f); data_size = csz; fseek(f, (long)csz, SEEK_CUR);
        } else fseek(f, (long)csz, SEEK_CUR);
    }
    if (data_pos < 0 || data_size == 0 || sample_rate == 0) return -1;
    if (!(audio_format == 1 || audio_format == 3)) return -1;
    if (!(bits == 16 || bits == 32)) return -1;

    fseek(f, data_pos, SEEK_SET);
    int src_channels = (int)channels;
    int src_samples = (int)(data_size / (unsigned int)(src_channels * (bits / 8)));
    float *src = (float *)malloc((size_t)src_samples * (size_t)src_channels * sizeof(float));
    if (!src) return -1;

    if (bits == 16) {
        for (int i = 0; i < src_samples * src_channels; i++) { short v = 0; if (fread(&v, 2, 1, f) != 1) { free(src); return -1; } src[i] = (float)v / 32768.0f; }
    } else {
        for (int i = 0; i < src_samples * src_channels; i++) { float v = 0.0f; if (fread(&v, 4, 1, f) != 1) { free(src); return -1; } src[i] = v; }
    }

    int dst_samples = (int)((int64_t)src_samples * dst_sr / (int)sample_rate);
    if (dst_samples < 1) dst_samples = 1;
    float *dst = (float *)malloc((size_t)dst_samples * (size_t)dst_ch * sizeof(float));
    if (!dst) { free(src); return -1; }
    for (int t = 0; t < dst_samples; t++) {
        float pos = ((float)t * (float)sample_rate) / (float)dst_sr;
        int i0 = (int)pos, i1 = i0 + 1;
        if (i0 < 0) i0 = 0;
        if (i0 >= src_samples) i0 = src_samples - 1;
        if (i1 >= src_samples) i1 = src_samples - 1;
        float a = pos - (float)i0;
        for (int c = 0; c < dst_ch; c++) {
            int sc = (src_channels == 1) ? 0 : (c < src_channels ? c : src_channels - 1);
            float s0 = src[(size_t)i0 * src_channels + sc], s1 = src[(size_t)i1 * src_channels + sc];
            dst[(size_t)t * dst_ch + c] = s0 + (s1 - s0) * a;
        }
    }
    free(src);
    int rc = moss_audio_tok_encode_wav(tok, dst, dst_samples, dst_ch, out_codes, out_frames);
    free(dst);
    return rc;
}

int moss_audio_tok_encode_wav_file(const moss_audio_tok_t *tok, const char *wav_path, int **out_codes, int *out_frames) {
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
        fprintf(stderr, "[moss_audio_tok] not a RIFF/WAVE PCM file (e.g. FLAC-in-.wav): %s.\n", wav_path);
        fprintf(stderr, "  Use PCM WAV, or unset MOSS_DISABLE_FFMPEG and install ffmpeg for auto-transcode.\n");
        return -1;
    }
    const char *ffmpeg_exe = getenv("MOSS_FFMPEG");
    if (!ffmpeg_exe || !ffmpeg_exe[0]) ffmpeg_exe = "ffmpeg";
    char tmp_path[768];
    if (moss_aud_tmp_pcm_wav_path(tmp_path, sizeof(tmp_path), moss_aud_tmp_salt()) != 0) return -1;
    fprintf(stderr, "[moss_audio_tok] transcoding via ffmpeg → %dkHz %dch PCM WAV (%s …)\n", dst_sr / 1000, dst_ch, ffmpeg_exe);
    if (moss_run_ffmpeg_to_pcm_wav(ffmpeg_exe, wav_path, tmp_path, dst_sr, dst_ch) != 0) {
        fprintf(stderr, "[moss_audio_tok] ffmpeg failed (need ffmpeg in PATH, or MOSS_FFMPEG to its binary).\n");
        remove(tmp_path);
        return -1;
    }
    FILE *tf = fopen(tmp_path, "rb");
    if (!tf) { remove(tmp_path); return -1; }
    nh = fread(fh, 1, 12, tf);
    if (nh < 12 || memcmp(fh, "RIFF", 4) != 0 || memcmp(fh + 8, "WAVE", 4) != 0) {
        fclose(tf); remove(tmp_path); fprintf(stderr, "[moss_audio_tok] ffmpeg produced non-WAV output\n"); return -1;
    }
    rewind(tf);
    int rc = moss_aud_read_riff_wave_and_encode(tok, tf, dst_sr, dst_ch, out_codes, out_frames);
    fclose(tf);
    remove(tmp_path);
    return rc;
}
