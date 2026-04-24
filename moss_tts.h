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
#define MOSS_MAX_JOINT_ROWS 1024
#define MOSS_MAX_PROMPT_AUDIO_FRAMES 1024
#define MOSS_N_COLS 17

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
    int max_new_frames;
    int sample_rate;
    float temperature;
} moss_generate_params_t;

moss_tts_ctx_t *moss_tts_load(const char *model_dir, moss_backend_t backend);
void moss_tts_unload(moss_tts_ctx_t *ctx);
const char *moss_backend_name(moss_backend_t backend);

/*
 * prompt_audio_codes_path: optional text file with precomputed VQ rows (same stdout format as old
 * moss_encode_prompt_wav.py: line1 = T, then T lines of n_vq ints). WAV→codes is a separate audio
 * tokenizer (not in TTS safetensors); without ONNX/Python, prepare this file offline.
 */
int moss_tts_generate_codes(
    moss_tts_ctx_t *ctx,
    const char *text,
    const char *prompt_audio_codes_path,
    const moss_generate_params_t *params,
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

int moss_write_wav16(const char *path, const float *samples, int n_samples, int sample_rate);

#ifdef __cplusplus
}
#endif

#endif
