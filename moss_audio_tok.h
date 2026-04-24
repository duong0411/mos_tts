#ifndef MOSS_AUDIO_TOK_H
#define MOSS_AUDIO_TOK_H

#include "safetensors.h"

typedef struct {
    int sample_rate;
    int downsample_rate;
    int channels;
    int num_quantizers;
    int codebook_size;
} moss_audio_tok_config_t;

typedef struct {
    moss_audio_tok_config_t cfg;
    safetensors_file_t *sf;
    char model_path[1024];
} moss_audio_tok_t;

/* Load audio tokenizer config + safetensors shard from model_dir/audio_tokenizer. */
int moss_audio_tok_load(const char *model_dir, moss_audio_tok_t *out);
void moss_audio_tok_unload(moss_audio_tok_t *tok);

/* Phase-2 target (native WAV->codes). Placeholder for now. */
int moss_audio_tok_encode_wav(
    const moss_audio_tok_t *tok,
    const float *pcm,
    int n_samples,
    int n_channels,
    int **out_codes,
    int *out_frames
);

/* Convenience API: read WAV file then encode to [frames * num_quantizers] int codes. */
int moss_audio_tok_encode_wav_file(
    const moss_audio_tok_t *tok,
    const char *wav_path,
    int **out_codes,
    int *out_frames
);

/* Decode [frames * num_quantizers] codes to mono PCM float samples. */
int moss_audio_tok_decode_codes(
    const moss_audio_tok_t *tok,
    const int *codes,
    int frames,
    int sample_rate,
    float **out_samples,
    int *out_n_samples
);

#endif
