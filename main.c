#include "moss_tts.h"
#include "moss_sp_prompt.h"

#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>


static void moss_argv_strip_tail_cr(int argc, char **argv) {
    if (!argv) return;
    for (int i = 1; i < argc; i++) {
        char *s = argv[i];
        if (!s) continue;
        size_t len = strlen(s);
        while (len > 0 && (s[len - 1] == '\r' || s[len - 1] == '\n')) s[--len] = '\0';
    }
}

static void usage(const char *argv0) {
    fprintf(stderr,
        "Usage: %s --model-dir DIR --text TEXT --out out.wav [options]\n"
        "Options:\n"
        "  --prompt-audio-path PATH  Reference audio (PCM RIFF/WAVE or any format ffmpeg can read; see MOSS_FFMPEG)\n"
        "  --max-new-frames N  Maximum generated frames per chunk (default: %d, same as infer.py)\n"
        "  --min-frames N    Ignore end token until N frames (default: 0; infer.py has no minimum)\n"
        "  --sample-rate N   Output sample rate (default: 48000)\n"
        "  --backend NAME    auto|avx2|neon|generic (default: auto; AVX2/NEON reserved, use BLAS when built with it)\n"
        "  --do-sample 0|1   1=text assistant/end sampled (default: 1); 0=greedy 2-way\n"
        "  --text-temperature F  Temperature for that 2-way text step (default: 1.0)\n"
        "  --text-top-p F    Text nucleus top-p for assistant-vs-end step (default: 1.0)\n"
        "  --text-top-k N    Text top-k for assistant-vs-end step (default: 50)\n"
        "  --text STRING     Raw UTF-8 text input.\n"
        "  --audio-temperature F  Audio token sampling temperature (default: 0.8)\n"
        "  --audio-top-p F   Audio nucleus sampling top-p in (0,1] (default: 0.95)\n"
        "  --audio-top-k N   Audio top-k sampling (default: 25)\n"
        "  --audio-repetition-penalty F  Audio repetition penalty >= 1.0 (default: 1.2)\n"
        "  --repetition-penalty F  Alias for --audio-repetition-penalty\n"
        "  --end-token-logit-bias F  Additive bias on audio_end logit (default: 0.0; >0 stops earlier)\n"
        "  --voice-clone-max-text-tokens N  Pocket-tts chunking like infer.py (default: 75; <=0 disables)\n"
        "  --dump-codes FILE  Dump generated audio token ids (line1=frames, then 16 ints per line)\n"
        "  --stream           Stream audio (decode during generation)\n"
        "  --stream-every N   Stream decode cadence in frames (default: 8)\n"
        "  --threads N        Set OPENBLAS/OMP threads (default: keep runtime default)\n"
        "  --seed U          RNG seed (unsigned); 0 = auto (time-based). Omit flag for same auto behavior.\n",
        argv0,
        MOSS_DEFAULT_MAX_NEW_FRAMES);
}

static int moss_append_float_pcm(float **acc, int *acc_n, const float *blk, int blk_n) {
    if (!acc || !acc_n || (!blk && blk_n > 0) || blk_n < 0) return -1;
    float *p = (float *)realloc(*acc, (size_t)(*acc_n + blk_n) * sizeof(float));
    if (!p) return -1;
    if (blk_n > 0) memcpy(p + *acc_n, blk, (size_t)blk_n * sizeof(float));
    *acc_n += blk_n;
    *acc = p;
    return 0;
}

static int moss_append_zero_pcm(float **acc, int *acc_n, int nfloats) {
    if (!acc || !acc_n || nfloats < 0) return -1;
    float *p = (float *)realloc(*acc, (size_t)(*acc_n + nfloats) * sizeof(float));
    if (!p) return -1;
    memset(p + *acc_n, 0, (size_t)nfloats * sizeof(float));
    *acc_n += nfloats;
    *acc = p;
    return 0;
}



static moss_backend_t parse_backend(const char *s) {
    if (strcmp(s, "avx2") == 0) return MOSS_BACKEND_AVX2;
    if (strcmp(s, "neon") == 0) return MOSS_BACKEND_NEON;
    if (strcmp(s, "generic") == 0) return MOSS_BACKEND_GENERIC;
    return MOSS_BACKEND_AUTO;
}

int main(int argc, char **argv) {
    moss_argv_strip_tail_cr(argc, argv);

    /* Unbuffered stderr forces a flush per log line and can dominate wall time on long runs. */
    if (setvbuf(stderr, NULL, _IOLBF, 0) != 0) {
        setbuf(stderr, NULL);
    }
    const char *model_dir = NULL;
    const char *text = NULL;
    const char *out = NULL;
    const char *prompt_audio_input_path = NULL;
    int num_threads = 0;
    moss_backend_t backend = MOSS_BACKEND_AUTO;
    moss_generate_params_t params = {
        .max_new_frames = MOSS_DEFAULT_MAX_NEW_FRAMES,
        .min_frames = 0,
        .fill_to_max = 0,
        .sample_rate = 48000,
        .temperature = 1.0f,
        .do_sample = 1,
        .text_temperature = 1.0f,
        .text_top_p = 1.0f,
        .text_top_k = 50,
        .audio_temperature = 0.8f,
        .audio_top_p = 0.95f,
        .audio_top_k = 25,
        .audio_repetition_penalty = 1.2f,
        .end_token_logit_bias = 0.0f,
        .rng_seed = 0,
        .voice_clone_max_text_tokens = 75,
        .stream_decode = 0,
        .stream_every_frames = 8,
        .stream_output_path = NULL,
        .rng_state = 0,
    };

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model-dir") == 0 && i + 1 < argc) model_dir = argv[++i];
        else if (strcmp(argv[i], "--text") == 0 && i + 1 < argc) text = argv[++i];
        else if ((strcmp(argv[i], "--out") == 0 || strcmp(argv[i], "--output") == 0) && i + 1 < argc)
            out = argv[++i];
        else if (strcmp(argv[i], "--max-new-frames") == 0 && i + 1 < argc)
            params.max_new_frames = atoi(argv[++i]);
        else if (strcmp(argv[i], "--min-frames") == 0 && i + 1 < argc) params.min_frames = atoi(argv[++i]);
        else if (strcmp(argv[i], "--sample-rate") == 0 && i + 1 < argc) params.sample_rate = atoi(argv[++i]);
        else if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) backend = parse_backend(argv[++i]);
        else if (strcmp(argv[i], "--do-sample") == 0 && i + 1 < argc) params.do_sample = atoi(argv[++i]) != 0;
        else if (strcmp(argv[i], "--text-temperature") == 0 && i + 1 < argc)
            params.text_temperature = (float)strtod(argv[++i], NULL);
        else if (strcmp(argv[i], "--text-top-p") == 0 && i + 1 < argc)
            params.text_top_p = (float)strtod(argv[++i], NULL);
        else if (strcmp(argv[i], "--text-top-k") == 0 && i + 1 < argc)
            params.text_top_k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--audio-temperature") == 0 && i + 1 < argc)
            params.audio_temperature = (float)strtod(argv[++i], NULL);
        else if (strcmp(argv[i], "--audio-top-p") == 0 && i + 1 < argc)
            params.audio_top_p = (float)strtod(argv[++i], NULL);
        else if (strcmp(argv[i], "--audio-top-k") == 0 && i + 1 < argc)
            params.audio_top_k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--audio-repetition-penalty") == 0 && i + 1 < argc)
            params.audio_repetition_penalty = (float)strtod(argv[++i], NULL);
        else if (strcmp(argv[i], "--repetition-penalty") == 0 && i + 1 < argc)
            params.audio_repetition_penalty = (float)strtod(argv[++i], NULL);
        else if (strcmp(argv[i], "--end-token-logit-bias") == 0 && i + 1 < argc)
            params.end_token_logit_bias = (float)strtod(argv[++i], NULL);
        else if (strcmp(argv[i], "--voice-clone-max-text-tokens") == 0 && i + 1 < argc)
            params.voice_clone_max_text_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "--stream") == 0)
            params.stream_decode = 1;
        else if (strcmp(argv[i], "--stream-every") == 0 && i + 1 < argc)
            params.stream_every_frames = atoi(argv[++i]);
        else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc)
            num_threads = atoi(argv[++i]);
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
            "[moss_tts] warning: --max-new-frames=%d is too short and often sounds empty; clamping to 16\n",
            params.max_new_frames);
        params.max_new_frames = 16;
    }
    if (params.min_frames < 0) params.min_frames = 0;
    if (params.max_new_frames > 0 && params.min_frames > params.max_new_frames) {
        params.min_frames = params.max_new_frames;
    }
    if (params.audio_top_k < 1) params.audio_top_k = 1;
    if (params.audio_top_p <= 0.0f || params.audio_top_p > 1.0f) params.audio_top_p = 1.0f;
    if (params.text_top_p <= 0.0f || params.text_top_p > 1.0f) params.text_top_p = 1.0f;
    if (params.audio_repetition_penalty < 1.0f) params.audio_repetition_penalty = 1.0f;
    if (params.stream_every_frames < 1) params.stream_every_frames = 1;
    if (num_threads > 0) {
        char tbuf[32];
        snprintf(tbuf, sizeof(tbuf), "%d", num_threads);
        setenv("OPENBLAS_NUM_THREADS", tbuf, 1);
        setenv("GOTO_NUM_THREADS", tbuf, 1);
        setenv("OMP_NUM_THREADS", tbuf, 1);
        fprintf(stderr, "[moss_tts] threads=%d (OPENBLAS_NUM_THREADS/GOTO_NUM_THREADS/OMP_NUM_THREADS)\n", num_threads);
    }
    if (params.stream_decode) {
        params.stream_output_path = out;
    }

    fprintf(stderr, "[moss_tts] loading %s ...\n", model_dir);
    moss_tts_ctx_t *ctx = moss_tts_load(model_dir, backend);
    if (!ctx) {
        fprintf(stderr, "Failed to load model from %s\n", model_dir);
        return 2;
    }
    if (!moss_audio_tok_is_loaded(&ctx->audio_tok)) {
        fprintf(stderr,
            "[moss_tts] fatal: native audio tokenizer not loaded. Need %s/audio_tokenizer/config.json "
            "and model-*.safetensors (same layout as HF export). Speech decode is not available without it.\n",
            model_dir);
        moss_tts_unload(ctx);
        return 2;
    }
    fprintf(stderr,
        "[moss_tts] loaded backend=%s n_embd=%d n_head=%d n_inner=%d "
        "global_transformer_layers=%d local_transformer_layers=%d n_vq=%d rope_base=%g\n",
        moss_backend_name(ctx->backend),
        ctx->cfg.n_embd,
        ctx->cfg.n_head,
        ctx->cfg.n_inner,
        ctx->cfg.n_layer,
        ctx->cfg.local_n_layer,
        ctx->cfg.n_vq,
        (double)ctx->cfg.rope_base);

    int *codes = (int *)malloc((size_t)MOSS_MAX_FRAMES * 16 * sizeof(int));
    int n_frames = 0;
    if (!codes) {
        moss_tts_unload(ctx);
        return 3;
    }

    params.rng_state = 0ULL;

    char **vc_chunks = NULL;
    int n_vc_chunks = 0;
    int vc_chunks_owned = 0;
    if (prompt_audio_input_path && params.voice_clone_max_text_tokens > 0) {
        if (moss_voice_clone_split_text(
                model_dir, text, params.voice_clone_max_text_tokens, &vc_chunks, &n_vc_chunks)
            != 0) {
            fprintf(stderr,
                "[moss_tts] moss_voice_clone_split_text failed (need tokenizer.model under model-dir or "
                "checkpoint/)\n");
            free(codes);
            moss_tts_unload(ctx);
            return 4;
        }
        vc_chunks_owned = 1;
        fprintf(stderr,
            "[moss_tts] voice_clone chunking: %d segment(s) (infer.py-style; max_text_tokens=%d)\n",
            n_vc_chunks,
            params.voice_clone_max_text_tokens);
    } else {
        n_vc_chunks = 1;
    }

    fprintf(stderr,
        "[moss_tts] generating (max_new_frames=%d, min_frames=%d, fill_to_max=%d) do_sample=%d "
        "text_temperature=%g text_top_p=%g text_top_k=%d end_bias=%g audio_temp=%g top_p=%g top_k=%d rep_pen=%g ",
        params.max_new_frames,
        params.min_frames,
        params.fill_to_max,
        params.do_sample != 0,
        (double)params.text_temperature,
        (double)params.text_top_p,
        params.text_top_k,
        (double)params.end_token_logit_bias,
        (double)params.audio_temperature,
        (double)params.audio_top_p,
        params.audio_top_k,
        (double)params.audio_repetition_penalty);
    if (params.rng_seed != 0ull)
        fprintf(stderr, "seed=%llu ...\n", (unsigned long long)params.rng_seed);
    else
        fprintf(stderr, "seed=auto(time) ...\n");

    float *samples = NULL;
    int n_samples = 0;
    int n_channels = 1;
    int total_frames = 0;

    for (int ci = 0; ci < n_vc_chunks; ci++) {
        const char *chunk_txt = vc_chunks_owned ? vc_chunks[ci] : text;
        if (n_vc_chunks > 1)
            fprintf(stderr, "[moss_tts] chunk %d/%d\n", ci + 1, n_vc_chunks);

        if (ci > 0) {
            const char *prev_txt = vc_chunks_owned ? vc_chunks[ci - 1] : text;
            float psec = moss_voice_clone_inter_chunk_pause_seconds(prev_txt);
            int per_ch = (int)((double)params.sample_rate * (double)psec + 0.5);
            if (per_ch < 0) per_ch = 0;
            int silence = per_ch * n_channels;
            if (moss_append_zero_pcm(&samples, &n_samples, silence) != 0) {
                fprintf(stderr, "[moss_tts] OOM appending inter-chunk silence\n");
                free(samples);
                if (vc_chunks_owned) moss_voice_clone_free_split(vc_chunks, n_vc_chunks);
                free(codes);
                moss_tts_unload(ctx);
                return 3;
            }
            fprintf(stderr, "[moss_tts] inter-chunk pause %.3fs (%d interleaved samples)\n", psec, silence);
        }

        int rc = moss_tts_generate_codes(ctx, chunk_txt, prompt_audio_input_path, &params, codes, &n_frames);
        if (rc != 0) {
            fprintf(stderr, "Generation failed (code %d)\n", rc);
            free(samples);
            if (vc_chunks_owned) moss_voice_clone_free_split(vc_chunks, n_vc_chunks);
            free(codes);
            moss_tts_unload(ctx);
            return 4;
        }
        total_frames += n_frames;

        float *chunk_samples = NULL;
        int chunk_n = 0;
        int chunk_ch = 1;
        fprintf(stderr, "[moss_tts] decoding codes -> wav (%d frames) ...\n", n_frames);
        if (moss_audio_tok_decode_codes(
                &ctx->audio_tok, codes, n_frames, params.sample_rate, &chunk_samples, &chunk_n, &chunk_ch)
            != 0) {
            fprintf(stderr, "[moss_tts] native decode failed (chunk %d)\n", ci + 1);
            free(chunk_samples);
            free(samples);
            if (vc_chunks_owned) moss_voice_clone_free_split(vc_chunks, n_vc_chunks);
            free(codes);
            moss_tts_unload(ctx);
            return 5;
        }
        if (!chunk_samples || chunk_n <= 0) {
            free(chunk_samples);
            free(samples);
            if (vc_chunks_owned) moss_voice_clone_free_split(vc_chunks, n_vc_chunks);
            free(codes);
            moss_tts_unload(ctx);
            const char *ph = getenv("MOSS_PLACEHOLDER_DECODE");
            int allow_placeholder = ph && ph[0] && strcmp(ph, "0") != 0;
            fprintf(stderr,
                allow_placeholder
                    ? "[moss_tts] MOSS_PLACEHOLDER_DECODE not applied per chunk; decode produced no samples.\n"
                    : "[moss_tts] fatal: no decoded samples for chunk.\n");
            return 5;
        }
        if (ci == 0)
            n_channels = chunk_ch;
        else if (chunk_ch != n_channels) {
            fprintf(stderr,
                "[moss_tts] fatal: channel count changed between chunks (%d vs %d)\n",
                n_channels,
                chunk_ch);
            free(chunk_samples);
            free(samples);
            if (vc_chunks_owned) moss_voice_clone_free_split(vc_chunks, n_vc_chunks);
            free(codes);
            moss_tts_unload(ctx);
            return 5;
        }
        if (moss_append_float_pcm(&samples, &n_samples, chunk_samples, chunk_n) != 0) {
            fprintf(stderr, "[moss_tts] OOM merging decoded audio\n");
            free(chunk_samples);
            free(samples);
            if (vc_chunks_owned) moss_voice_clone_free_split(vc_chunks, n_vc_chunks);
            free(codes);
            moss_tts_unload(ctx);
            return 3;
        }
        free(chunk_samples);
    }

    if (vc_chunks_owned) moss_voice_clone_free_split(vc_chunks, n_vc_chunks);


    if (!samples || n_samples <= 0) {
        const char *ph = getenv("MOSS_PLACEHOLDER_DECODE");
        int allow_placeholder = ph && ph[0] && strcmp(ph, "0") != 0;
        if (!allow_placeholder) {
            fprintf(stderr,
                "[moss_tts] fatal: no decoded samples. If you are debugging, set MOSS_PLACEHOLDER_DECODE=1 for a "
                "test-tone fallback (not speech).\n");
            free(codes);
            moss_tts_unload(ctx);
            return 5;
        }
        fprintf(stderr, "[moss_tts] MOSS_PLACEHOLDER_DECODE enabled: writing test-tone placeholder (not speech)\n");
        if (moss_tts_decode_codes_to_wav(codes, n_frames, params.sample_rate, &samples, &n_samples) != 0) {
            fprintf(stderr, "Placeholder decode failed\n");
            free(codes);
            moss_tts_unload(ctx);
            return 5;
        }
        n_channels = 1;
    }

    fprintf(stderr, "[moss_tts] writing %s ...\n", out);
    if (moss_write_wav16(out, samples, n_samples, params.sample_rate, n_channels) != 0) {
        fprintf(stderr, "Failed to write wav: %s (%s)\n", out, strerror(errno));
        free(samples);
        free(codes);
        moss_tts_unload(ctx);
        return 6;
    }

    fprintf(stderr,
        "[moss_tts] saved generated audio to %s sample_rate=%d frames=%d (chunks=%d)\n",
        out,
        params.sample_rate,
        total_frames,
        n_vc_chunks);
    printf("backend=%s samples=%d channels=%d output=%s\n",
        moss_backend_name(ctx->backend), n_samples, n_channels, out);

    free(samples);
    free(codes);
    moss_tts_unload(ctx);
    return 0;
}
