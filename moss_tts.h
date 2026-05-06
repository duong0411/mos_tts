#ifndef MOSS_TTS_H
#define MOSS_TTS_H

#include <stdint.h>

#include "moss_audio_tok.h"
#include "moss_config.h"
#include "moss_weights.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MOSS_MAX_TEXT_LEN 4096
#define MOSS_MAX_FRAMES 512
/* Upper cap for one generate call; must stay <= MOSS_MAX_FRAMES (buffer sizing in main.c). */
#define MOSS_DEFAULT_MAX_NEW_FRAMES 375 /* infer.py --max-new-frames default */
#define MOSS_MAX_JOINT_ROWS 1024
#define MOSS_MAX_PROMPT_AUDIO_FRAMES 1024
#define MOSS_N_COLS (1 + MOSS_MAX_NVQ)

typedef enum {
    MOSS_BACKEND_AUTO = 0,
    MOSS_BACKEND_AVX2 = 1,
    MOSS_BACKEND_NEON = 2,
    MOSS_BACKEND_GENERIC = 3,
} moss_backend_t;

typedef struct {
    moss_run_config_t cfg;
    moss_weight_bundle_t wb;
    moss_audio_tok_t audio_tok;
    moss_backend_t backend;
    char model_dir[1024];
} moss_tts_ctx_t;

typedef struct {
    /* Max audio frames to *attempt*; 0 or negative = use MOSS_DEFAULT_MAX_NEW_FRAMES (same as infer.py). */
    int max_new_frames;
    int min_frames;
    int fill_to_max; /* 1: ignore audio_end until max_new_frames (forces ~--frames new audio frames) */
    int sample_rate;
    float temperature; /* legacy; also used as default for text_temperature if unset elsewhere */
    int do_sample;       /* 1 = match infer.py default: sample assistant vs end (2-way only) */
    float text_temperature;
    float text_top_p; /* infer.py resolve_sampling_kwargs: default 1.0); HF _sample_next_assistant_text_token */
    int text_top_k; /* default 50; only values 1 or >=2 affect the 2-way head */
    float audio_temperature;
    float audio_top_p;
    int audio_top_k;
    float audio_repetition_penalty;
    /* Additive bias on audio_end logit in assistant-vs-end text step.
     * 0.0 keeps baseline behavior; >0 prefers earlier stop; <0 delays stop. */
    float end_token_logit_bias;
    unsigned long long rng_seed; /* 0 = seed from time (non-deterministic) */
    /* infer.py voice_clone: chunk target token budget; <= 0 disables (single joint, full text). Default 75. */
    int voice_clone_max_text_tokens;
    /* Stream decode during generation: rewrite stream_output_path every stream_every_frames. */
    int stream_decode;
    int stream_every_frames;
    const char *stream_output_path;
    /* After each moss_tts_generate_codes: updated PRNG state (xoroshiro); 0 on entry = init from rng_seed/time. */
    unsigned long long rng_state;
} moss_generate_params_t;

moss_tts_ctx_t *moss_tts_load(const char *model_dir, moss_backend_t backend);
void moss_tts_unload(moss_tts_ctx_t *ctx);
const char *moss_backend_name(moss_backend_t backend);

/*
 * prompt_audio_codes_path: optional text file with precomputed VQ rows (line1 = T, then T lines of
 * n_vq ints), or a WAV path encoded by the native audio tokenizer. WAV->VQ uses C/C++ only.
 */
int moss_tts_generate_codes(
    moss_tts_ctx_t *ctx,
    const char *text,
    const char *prompt_audio_codes_path,
    moss_generate_params_t *params,
    int *out_codes,
    int *out_frames
);

int moss_tts_decode_codes_to_wav(
    const int *codes,
    int frames,
    int sample_rate,
    float **out_samples,
    int *out_n_samples
);

int moss_write_wav16(const char *path, const float *samples, int n_samples, int sample_rate, int n_channels);

#ifdef __cplusplus
}
#endif

#endif
