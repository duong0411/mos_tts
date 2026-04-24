#ifndef MOSS_CONFIG_H
#define MOSS_CONFIG_H

typedef struct {
    int n_vq;
    int n_embd;
    int n_head;
    int n_layer;
    int n_inner;
    float layer_norm_epsilon;
    float rope_base;
    int vocab_size;
    int audio_vocab_size;
    int audio_pad_token_id;
    int pad_token_id;
    int im_start_token_id;
    int im_end_token_id;
    int audio_start_token_id;
    int audio_end_token_id;
    int audio_user_slot_token_id;
    int audio_assistant_slot_token_id;
    int local_n_layer;
} moss_run_config_t;

/* Parse ../weight/config.json into cfg. Returns 0 on success. */
int moss_config_load(const char *model_dir, moss_run_config_t *cfg);

#endif
