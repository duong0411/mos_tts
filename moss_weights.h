#ifndef MOSS_WEIGHTS_H
#define MOSS_WEIGHTS_H

#include "moss_config.h"
#include "moss_gpt2.h"
#include "safetensors.h"

typedef struct {
    safetensors_file_t *sf;
    moss_gpt2_stack_t global;
    moss_gpt2_stack_t local;
    const uint16_t *wte;
    const uint16_t *text_lm_head;
    const uint16_t *audio_emb[16];
    const uint16_t *audio_head[16];
} moss_weight_bundle_t;

int moss_weights_load(const char *model_dir, moss_weight_bundle_t *out, const moss_run_config_t *cfg);

void moss_weights_unload(moss_weight_bundle_t *w);

#endif
