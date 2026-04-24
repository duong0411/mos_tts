#ifndef MOSS_GPT2_H
#define MOSS_GPT2_H

#include <stddef.h>
#include <stdint.h>

#include "moss_config.h"

typedef struct {
    const uint16_t *ln1w, *ln1b, *ln2w, *ln2b;
    const uint16_t *c_attn_w, *c_attn_b, *c_proj_w, *c_proj_b;
    const uint16_t *fc_in_w, *fc_in_b, *fc_out_w, *fc_out_b;
} moss_gpt2_layer_w_t;

typedef struct {
    moss_gpt2_layer_w_t layers[28];
    int n_layer;
    const uint16_t *ln_f_w, *ln_f_b;
} moss_gpt2_stack_t;

/*
 * Forward GPT-2 stack (no dropout). hidden [S * D] in/out.
 * attn_mask: length S, 1 = keep position, 0 = masked (zero output for that row).
 * scratch: must hold at least 10 * S * D + 3 * S * S floats (caller passes size).
 */
int moss_gpt2_forward(
    const moss_gpt2_stack_t *stk,
    const moss_run_config_t *cfg,
    float *hidden,
    int S,
    const unsigned char *attn_mask,
    float *scratch,
    size_t scratch_elems
);

#endif
