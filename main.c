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
        "  --max-new-frames N  Maximum number of generated frames (default: 96)\n"
        "  --frames N        Alias for --max-new-frames\n"
        "  --min-frames N    Minimum generated frames before allowing end token (default: 32)\n"
        "  --sample-rate N   Output sample rate (default: 48000)\n"
        "  --backend NAME    auto|avx2|neon|generic (default: auto)\n"
        "  --do-sample 0|1   1=text assistant/end sampled like infer.py (default: 1); 0=greedy 2-way\n"
        "  --text-temperature F  Temperature for that 2-way text step (default: 1.5)\n"
        "  --audio-temperature F  Audio token sampling temperature (default: 0.8)\n"
        "  --audio-top-p F   Audio nucleus sampling top-p in (0,1] (default: 0.95)\n"
        "  --audio-top-k N   Audio top-k sampling (default: 25)\n"
        "  --audio-repetition-penalty F  Audio repetition penalty >= 1.0 (default: 1.2)\n"
        "  --seed U          RNG seed (unsigned); default: nondeterministic\n",
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
        .min_frames = 32,
        .sample_rate = 48000,
        .temperature = 1.0f,
        .do_sample = 1,
        .text_temperature = 1.5f,
        .audio_temperature = 0.8f,
        .audio_top_p = 0.95f,
        .audio_top_k = 25,
        .audio_repetition_penalty = 1.2f,
        .rng_seed = 0,
    };

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model-dir") == 0 && i + 1 < argc) model_dir = argv[++i];
        else if (strcmp(argv[i], "--text") == 0 && i + 1 < argc) text = argv[++i];
        else if ((strcmp(argv[i], "--out") == 0 || strcmp(argv[i], "--output") == 0) && i + 1 < argc)
            out = argv[++i];
        else if ((strcmp(argv[i], "--frames") == 0 || strcmp(argv[i], "--max-new-frames") == 0) && i + 1 < argc)
            params.max_new_frames = atoi(argv[++i]);
        else if (strcmp(argv[i], "--min-frames") == 0 && i + 1 < argc) params.min_frames = atoi(argv[++i]);
        else if (strcmp(argv[i], "--sample-rate") == 0 && i + 1 < argc) params.sample_rate = atoi(argv[++i]);
        else if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) backend = parse_backend(argv[++i]);
        else if (strcmp(argv[i], "--do-sample") == 0 && i + 1 < argc) params.do_sample = atoi(argv[++i]) != 0;
        else if (strcmp(argv[i], "--text-temperature") == 0 && i + 1 < argc)
            params.text_temperature = (float)strtod(argv[++i], NULL);
        else if (strcmp(argv[i], "--audio-temperature") == 0 && i + 1 < argc)
            params.audio_temperature = (float)strtod(argv[++i], NULL);
        else if (strcmp(argv[i], "--audio-top-p") == 0 && i + 1 < argc)
            params.audio_top_p = (float)strtod(argv[++i], NULL);
        else if (strcmp(argv[i], "--audio-top-k") == 0 && i + 1 < argc)
            params.audio_top_k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--audio-repetition-penalty") == 0 && i + 1 < argc)
            params.audio_repetition_penalty = (float)strtod(argv[++i], NULL);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
            params.rng_seed = (unsigned long long)strtoull(argv[++i], NULL, 0);
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
    if (params.min_frames < 0) params.min_frames = 0;
    if (params.max_new_frames > 0 && params.min_frames > params.max_new_frames) {
        params.min_frames = params.max_new_frames;
    }
    if (params.audio_top_k < 1) params.audio_top_k = 1;
    if (params.audio_top_p <= 0.0f || params.audio_top_p > 1.0f) params.audio_top_p = 1.0f;
    if (params.audio_repetition_penalty < 1.0f) params.audio_repetition_penalty = 1.0f;

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

    fprintf(stderr,
        "[moss_tts] generating (max_new_frames=%d, min_frames=%d) do_sample=%d text_temperature=%g audio_temp=%g top_p=%g top_k=%d rep_pen=%g seed=%llu ...\n",
        params.max_new_frames,
        params.min_frames,
        params.do_sample != 0,
        (double)params.text_temperature,
        (double)params.audio_temperature,
        (double)params.audio_top_p,
        params.audio_top_k,
        (double)params.audio_repetition_penalty,
        (unsigned long long)params.rng_seed);
    int rc = moss_tts_generate_codes(ctx, text, prompt_audio_input_path, &params, codes, &n_frames);
    if (rc != 0) {
        fprintf(stderr, "Generation failed (code %d)\n", rc);
        free(codes);
        moss_tts_unload(ctx);
        return 4;
    }

    float *samples = NULL;
    int n_samples = 0;
    int n_channels = 1;
    fprintf(stderr, "[moss_tts] decoding codes -> wav via native audio tokenizer (%d frames) ...\n", n_frames);
    if (moss_audio_tok_decode_codes(&ctx->audio_tok, codes, n_frames, params.sample_rate, &samples, &n_samples, &n_channels) != 0) {
        fprintf(stderr, "[moss_tts] native decode failed, fallback to placeholder decoder\n");
    }
    if (!samples || n_samples <= 0) {
        if (moss_tts_decode_codes_to_wav(codes, n_frames, params.sample_rate, &samples, &n_samples) != 0) {
            fprintf(stderr, "Codec decode failed\n");
            free(codes);
            moss_tts_unload(ctx);
            return 5;
        }
        n_channels = 1;
    }

    fprintf(stderr, "[moss_tts] writing %s ...\n", out);
    if (moss_write_wav16(out, samples, n_samples, params.sample_rate, n_channels) != 0) {
        fprintf(stderr, "Failed to write wav: %s\n", out);
        free(samples);
        free(codes);
        moss_tts_unload(ctx);
        return 6;
    }

    fprintf(stderr,
        "[moss_tts] saved generated audio to %s sample_rate=%d frames=%d\n",
        out,
        params.sample_rate,
        n_frames);
    printf("backend=%s samples=%d channels=%d output=%s\n",
        moss_backend_name(ctx->backend), n_samples, n_channels, out);

    free(samples);
    free(codes);
    moss_tts_unload(ctx);
    return 0;
}
