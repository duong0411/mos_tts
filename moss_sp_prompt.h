#ifndef MOSS_SP_PROMPT_H
#define MOSS_SP_PROMPT_H

#include "moss_config.h"

#ifdef __cplusplus
extern "C" {
#endif

/* SentencePiece + MOSS chat template (same as moss_prompt_tokens.py). Returns count, or <0 on error. */
int moss_build_prompt_token_ids(
    const char *model_dir,
    const moss_run_config_t *cfg,
    const char *text_utf8,
    int *out_ids,
    int max_ids
);

#ifdef __cplusplus
}
#endif

#endif
