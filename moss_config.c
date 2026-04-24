#include "moss_config.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int parse_json_int_in(const char *start, const char *end, const char *key, int *out) {
    char pat[128];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = start;
    while (p && p < end) {
        p = strstr(p, pat);
        if (!p || p >= end) return -1;
        p = strchr(p + strlen(pat), ':');
        if (!p || p >= end) return -1;
        p++;
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
        *out = atoi(p);
        return 0;
    }
    return -1;
}

int moss_config_load(const char *model_dir, moss_run_config_t *cfg) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/config.json", model_dir);
    FILE *f = fopen(path, "rb");
    if (!f) {
        /* New layout: model_dir/checkpoint/config.json */
        snprintf(path, sizeof(path), "%s/checkpoint/config.json", model_dir);
        f = fopen(path, "rb");
        if (!f) return -1;
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) { fclose(f); return -1; }
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) {
        free(buf);
        fclose(f);
        return -1;
    }
    buf[n] = '\0';
    fclose(f);

    memset(cfg, 0, sizeof(*cfg));
    cfg->n_vq = 16;
    cfg->n_embd = 768;
    cfg->n_head = 12;
    cfg->n_layer = 12;
    cfg->n_inner = 3072;
    cfg->layer_norm_epsilon = 1e-5f;
    cfg->rope_base = 10000.0f;
    cfg->vocab_size = 16384;
    cfg->audio_vocab_size = 1024;
    cfg->audio_pad_token_id = 1024;
    cfg->pad_token_id = 3;
    cfg->im_start_token_id = 4;
    cfg->im_end_token_id = 5;
    cfg->audio_start_token_id = 6;
    cfg->audio_end_token_id = 7;
    cfg->audio_user_slot_token_id = 8;
    cfg->audio_assistant_slot_token_id = 9;
    cfg->local_n_layer = 1;

    const char *end = buf + n;
    parse_json_int_in(buf, end, "n_vq", &cfg->n_vq);
    parse_json_int_in(buf, end, "audio_vocab_size", &cfg->audio_vocab_size);
    parse_json_int_in(buf, end, "audio_pad_token_id", &cfg->audio_pad_token_id);
    parse_json_int_in(buf, end, "pad_token_id", &cfg->pad_token_id);
    parse_json_int_in(buf, end, "im_start_token_id", &cfg->im_start_token_id);
    parse_json_int_in(buf, end, "im_end_token_id", &cfg->im_end_token_id);
    parse_json_int_in(buf, end, "audio_start_token_id", &cfg->audio_start_token_id);
    parse_json_int_in(buf, end, "audio_end_token_id", &cfg->audio_end_token_id);
    parse_json_int_in(buf, end, "audio_user_slot_token_id", &cfg->audio_user_slot_token_id);
    parse_json_int_in(buf, end, "audio_assistant_slot_token_id", &cfg->audio_assistant_slot_token_id);
    parse_json_int_in(buf, end, "local_transformer_layers", &cfg->local_n_layer);

    const char *g2 = strstr(buf, "\"gpt2_config\"");
    if (g2) {
        const char *g2end = strchr(g2 + 1, '}');
        if (!g2end) g2end = end;
        else g2end += 1;
        parse_json_int_in(g2, g2end, "n_embd", &cfg->n_embd);
        parse_json_int_in(g2, g2end, "n_head", &cfg->n_head);
        parse_json_int_in(g2, g2end, "n_layer", &cfg->n_layer);
        parse_json_int_in(g2, g2end, "n_inner", &cfg->n_inner);
        parse_json_int_in(g2, g2end, "vocab_size", &cfg->vocab_size);
        float rb = cfg->rope_base;
        const char *rp = strstr(g2, "\"rope_base\"");
        if (rp && rp < g2end) {
            rp = strchr(rp, ':');
            if (rp) { rp++; while (*rp == ' ') rp++; sscanf(rp, "%f", &rb); }
        }
        cfg->rope_base = rb;
        float eps = cfg->layer_norm_epsilon;
        const char *ep = strstr(g2, "\"layer_norm_epsilon\"");
        if (ep && ep < g2end) {
            ep = strchr(ep, ':');
            if (ep) { ep++; while (*ep == ' ') ep++; sscanf(ep, "%f", &eps); }
        }
        cfg->layer_norm_epsilon = eps;
    }

    free(buf);
    return 0;
}
