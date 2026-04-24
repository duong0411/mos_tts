#include "moss_tts.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *argv0) {
    fprintf(stderr,
        "Usage: %s --model-dir DIR --text TEXT --out out.wav [options]\n"
        "  (--output is an alias for --out)\n"
        "Options:\n"
        "  --prompt-audio-path WAV  Prompt WAV path (encoded to VQ at runtime)\n"
        "  --prompt-audio-codes FILE  Precomputed VQ text file (line1=T, then T lines of 16 ints)\n"
        "  --frames N        Number of generated frames (default: 96)\n"
        "  --sample-rate N   Output sample rate (default: 24000)\n"
        "  --backend NAME    auto|avx2|neon|generic (default: auto)\n",
        argv0);
}

static moss_backend_t parse_backend(const char *s) {
    if (strcmp(s, "avx2") == 0) return MOSS_BACKEND_AVX2;
    if (strcmp(s, "neon") == 0) return MOSS_BACKEND_NEON;
    if (strcmp(s, "generic") == 0) return MOSS_BACKEND_GENERIC;
    return MOSS_BACKEND_AUTO;
}

int main(int argc, char **argv) {
    setbuf(stderr, NULL);
    const char *model_dir = NULL;
    const char *text = NULL;
    const char *out = NULL;
    const char *prompt_audio_input_path = NULL;
    moss_backend_t backend = MOSS_BACKEND_AUTO;
    moss_generate_params_t params = {
        .max_new_frames = 96,
        .sample_rate = 24000,
        .temperature = 1.0f,
    };

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model-dir") == 0 && i + 1 < argc) model_dir = argv[++i];
        else if (strcmp(argv[i], "--text") == 0 && i + 1 < argc) text = argv[++i];
        else if ((strcmp(argv[i], "--out") == 0 || strcmp(argv[i], "--output") == 0) && i + 1 < argc)
            out = argv[++i];
        else if (strcmp(argv[i], "--frames") == 0 && i + 1 < argc) params.max_new_frames = atoi(argv[++i]);
        else if (strcmp(argv[i], "--sample-rate") == 0 && i + 1 < argc) params.sample_rate = atoi(argv[++i]);
        else if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) backend = parse_backend(argv[++i]);
        else if (strcmp(argv[i], "--prompt-audio-codes") == 0 && i + 1 < argc)
            prompt_audio_input_path = argv[++i];
        else if (strcmp(argv[i], "--prompt-audio-path") == 0 && i + 1 < argc)
            prompt_audio_input_path = argv[++i];
        else {
            usage(argv[0]);
            return 1;
        }
    }

    if (!model_dir || !text || !out) {
        usage(argv[0]);
        return 1;
    }
    if (params.max_new_frames > 0 && params.max_new_frames < 16) {
        fprintf(stderr,
            "[moss_tts] warning: --frames=%d is too short and often sounds empty; clamping to 16\n",
            params.max_new_frames);
        params.max_new_frames = 16;
    }

    fprintf(stderr, "[moss_tts] loading %s ...\n", model_dir);
    moss_tts_ctx_t *ctx = moss_tts_load(model_dir, backend);
    if (!ctx) {
        fprintf(stderr, "Failed to load model from %s\n", model_dir);
        return 2;
    }
    fprintf(stderr, "[moss_tts] loaded backend=%s n_embd=%d n_layer=%d\n",
        moss_backend_name(ctx->backend), ctx->cfg.n_embd, ctx->cfg.n_layer);

    int *codes = (int *)malloc((size_t)MOSS_MAX_FRAMES * 16 * sizeof(int));
    int n_frames = 0;
    if (!codes) {
        moss_tts_unload(ctx);
        return 3;
    }

    fprintf(stderr, "[moss_tts] generating (max %d frames) ...\n", params.max_new_frames);
    int rc = moss_tts_generate_codes(ctx, text, prompt_audio_input_path, &params, codes, &n_frames);
    if (rc != 0) {
        fprintf(stderr, "Generation failed (code %d)\n", rc);
        free(codes);
        moss_tts_unload(ctx);
        return 4;
    }

    float *samples = NULL;
    int n_samples = 0;
    fprintf(stderr, "[moss_tts] decoding codes -> wav via native audio tokenizer (%d frames) ...\n", n_frames);
    if (moss_audio_tok_decode_codes(&ctx->audio_tok, codes, n_frames, params.sample_rate, &samples, &n_samples) != 0) {
        fprintf(stderr, "[moss_tts] native decode failed, fallback to placeholder decoder\n");
    }
    if (!samples || n_samples <= 0) {
        if (moss_tts_decode_codes_to_wav(codes, n_frames, params.sample_rate, &samples, &n_samples) != 0) {
            fprintf(stderr, "Codec decode failed\n");
            free(codes);
            moss_tts_unload(ctx);
            return 5;
        }
    }

    fprintf(stderr, "[moss_tts] writing %s ...\n", out);
    if (moss_write_wav16(out, samples, n_samples, params.sample_rate) != 0) {
        fprintf(stderr, "Failed to write wav: %s\n", out);
        free(samples);
        free(codes);
        moss_tts_unload(ctx);
        return 6;
    }

    printf("backend=%s frames=%d samples=%d output=%s\n",
        moss_backend_name(ctx->backend), n_frames, n_samples, out);

    free(samples);
    free(codes);
    moss_tts_unload(ctx);
    return 0;
}
