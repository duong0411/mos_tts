#include "moss_audio_tok.h"

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

static int has_required_tensor(safetensors_file_t *sf, const char *name) {
    for (int i = 0; i < sf->num_tensors; i++) {
        if (strcmp(sf->tensors[i].name, name) == 0) return 1;
    }
    return 0;
}

int moss_audio_tok_load(const char *model_dir, moss_audio_tok_t *out) {
    if (!model_dir || !out) return -1;
    memset(out, 0, sizeof(*out));

    char cfg_path[1024];
    snprintf(cfg_path, sizeof(cfg_path), "%s/audio_tokenizer/config.json", model_dir);
    FILE *f = fopen(cfg_path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) {
        fclose(f);
        return -1;
    }
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) {
        free(buf);
        fclose(f);
        return -1;
    }
    fclose(f);
    buf[n] = '\0';

    const char *end = buf + n;
    out->cfg.sample_rate = 48000;
    out->cfg.downsample_rate = 3840;
    out->cfg.channels = 2;
    out->cfg.num_quantizers = 16;
    out->cfg.codebook_size = 1024;
    parse_json_int_in(buf, end, "sample_rate", &out->cfg.sample_rate);
    parse_json_int_in(buf, end, "downsample_rate", &out->cfg.downsample_rate);
    parse_json_int_in(buf, end, "number_channels", &out->cfg.channels);
    parse_json_int_in(buf, end, "codebook_size", &out->cfg.codebook_size);
    parse_json_int_in(buf, end, "num_quantizers", &out->cfg.num_quantizers);
    free(buf);

    snprintf(out->model_path, sizeof(out->model_path), "%s/audio_tokenizer/model-00001-of-00001.safetensors", model_dir);
    out->sf = safetensors_open(out->model_path);
    if (!out->sf) return -1;

    if (!has_required_tensor(out->sf, "encoder.1.input_proj.weight") ||
        !has_required_tensor(out->sf, "quantizer.quantizers.0.codebook.weight") ||
        !has_required_tensor(out->sf, "decoder.7.output_proj.weight")) {
        moss_audio_tok_unload(out);
        return -1;
    }
    return 0;
}

void moss_audio_tok_unload(moss_audio_tok_t *tok) {
    if (!tok) return;
    if (tok->sf) {
        safetensors_close(tok->sf);
        tok->sf = NULL;
    }
    memset(tok, 0, sizeof(*tok));
}

int moss_audio_tok_is_loaded(const moss_audio_tok_t *tok) {
    return tok && tok->sf != NULL;
}
