#ifndef MOSS_AUDIO_TOK_INTERNAL_H
#define MOSS_AUDIO_TOK_INTERNAL_H

#include "moss_audio_tok.h"

#include <stdint.h>

const safetensor_t *moss_audio_tok_find_tensor(const safetensors_file_t *sf, const char *name);
int moss_audio_tok_get_f32_tensor(const safetensors_file_t *sf, const char *name, const float **ptr, int64_t *numel);

int moss_audio_tok_patched_decode_tm(const float *in, int T, int Dh, int patch, float **out, int *outT, int *outD);
int moss_audio_tok_patched_encode_tm(const float *in, int T, int D, int patch, float **out, int *outT, int *outD);
int moss_audio_tok_transformer_module_tm(
    const moss_audio_tok_t *tok,
    const char *prefix,
    int module_id,
    int num_layers,
    int in_dim,
    int out_dim,
    int context_tokens,
    const float *in,
    int T,
    float **out
);

#endif
