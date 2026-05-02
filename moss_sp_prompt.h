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

/* Build voice_clone text sections:
 *   prefix = build_user_prompt_prefix + [audio_start]
 *   suffix = [audio_end] + build_user_prompt_after_reference + text + build_assistant_prompt_prefix + [audio_start]
 */
int moss_build_voice_clone_sections(
    const char *model_dir,
    const moss_run_config_t *cfg,
    const char *text_utf8,
    int *out_prefix_ids,
    int max_prefix_ids,
    int *out_suffix_ids,
    int max_suffix_ids,
    int *out_prefix_len,
    int *out_suffix_len
);

/*
 * Match MossTTSNanoForCausalLM._split_text_into_best_sentences + pause logic for infer.py voice_clone.
 * max_tokens <= 0: one chunk (strdup of text).
 * On success: *out_chunks is calloc'd array of strdup'd UTF-8 strings; free with moss_voice_clone_free_split.
 */
int moss_voice_clone_split_text(
    const char *model_dir,
    const char *text_utf8,
    int max_tokens,
    char ***out_chunks,
    int *out_n_chunks
);
void moss_voice_clone_free_split(char **chunks, int n_chunks);

/* Same as Python _estimate_voice_clone_inter_chunk_pause_seconds (short vs long silence). */
float moss_voice_clone_inter_chunk_pause_seconds(const char *chunk_utf8);

#ifdef __cplusplus
}
#endif

#endif
