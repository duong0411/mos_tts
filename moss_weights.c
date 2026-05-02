#include "moss_weights.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const uint16_t *get_bf16_ptr(safetensors_file_t *sf, const char *name) {
    for (int i = 0; i < sf->num_tensors; i++) {
        if (strcmp(sf->tensors[i].name, name) == 0) {
            if (sf->tensors[i].dtype != DTYPE_BF16) return NULL;
            return (const uint16_t *)safetensors_data(sf, &sf->tensors[i]);
        }
    }
    return NULL;
}

static int fill_layer(moss_gpt2_layer_w_t *L, safetensors_file_t *sf, const char *prefix) {
    char buf[256];
#define LD(field, suffix) \
    do { \
        snprintf(buf, sizeof(buf), "%s%s", prefix, suffix); \
        L->field = get_bf16_ptr(sf, buf); \
        if (!L->field) { fprintf(stderr, "missing tensor %s\n", buf); return -1; } \
    } while (0)
    LD(ln1w, "ln_1.weight");
    LD(ln1b, "ln_1.bias");
    LD(ln2w, "ln_2.weight");
    LD(ln2b, "ln_2.bias");
    LD(c_attn_w, "attn.c_attn.weight");
    LD(c_attn_b, "attn.c_attn.bias");
    LD(c_proj_w, "attn.c_proj.weight");
    LD(c_proj_b, "attn.c_proj.bias");
    LD(fc_in_w, "mlp.fc_in.weight");
    LD(fc_in_b, "mlp.fc_in.bias");
    LD(fc_out_w, "mlp.fc_out.weight");
    LD(fc_out_b, "mlp.fc_out.bias");
#undef LD
    return 0;
}

int moss_weights_load(const char *model_dir, moss_weight_bundle_t *out, const moss_run_config_t *cfg) {
    memset(out, 0, sizeof(*out));
    if (!cfg || cfg->n_vq < 1 || cfg->n_vq > MOSS_MAX_NVQ) {
        fprintf(stderr, "moss_weights_load: invalid n_vq=%d (expected 1..%d)\n", cfg ? cfg->n_vq : -1, MOSS_MAX_NVQ);
        return -1;
    }
    char path[1024];
    snprintf(path, sizeof(path), "%s/pytorch_model.safetensors", model_dir);
    out->sf = safetensors_open(path);
    if (!out->sf) {
        /* New layout: model_dir/checkpoint/pytorch_model.safetensors */
        snprintf(path, sizeof(path), "%s/checkpoint/pytorch_model.safetensors", model_dir);
        out->sf = safetensors_open(path);
        if (!out->sf) {
            fprintf(stderr, "cannot open %s (or checkpoint fallback)\n", path);
            return -1;
        }
    }
    safetensors_file_t *sf = out->sf;

    out->global.n_layer = cfg->n_layer;
    for (int i = 0; i < cfg->n_layer; i++) {
        char px[128];
        snprintf(px, sizeof(px), "transformer.h.%d.", i);
        if (fill_layer(&out->global.layers[i], sf, px) != 0) return -1;
    }
    out->global.ln_f_w = get_bf16_ptr(sf, "transformer.ln_f.weight");
    out->global.ln_f_b = get_bf16_ptr(sf, "transformer.ln_f.bias");
    if (!out->global.ln_f_w || !out->global.ln_f_b) return -1;

    out->local.n_layer = cfg->local_n_layer;
    for (int i = 0; i < cfg->local_n_layer; i++) {
        char px[128];
        snprintf(px, sizeof(px), "local_transformer.h.%d.", i);
        if (fill_layer(&out->local.layers[i], sf, px) != 0) return -1;
    }
    out->local.ln_f_w = get_bf16_ptr(sf, "local_transformer.ln_f.weight");
    out->local.ln_f_b = get_bf16_ptr(sf, "local_transformer.ln_f.bias");
    if (!out->local.ln_f_w || !out->local.ln_f_b) return -1;

    out->wte = get_bf16_ptr(sf, "transformer.wte.weight");
    out->text_lm_head = get_bf16_ptr(sf, "text_lm_head.weight");
    if (!out->wte || !out->text_lm_head) return -1;

    for (int k = 0; k < cfg->n_vq; k++) {
        char n1[128], n2[128];
        snprintf(n1, sizeof(n1), "audio_embeddings.%d.weight", k);
        snprintf(n2, sizeof(n2), "audio_lm_heads.%d.weight", k);
        out->audio_emb[k] = get_bf16_ptr(sf, n1);
        out->audio_head[k] = get_bf16_ptr(sf, n2);
        if (!out->audio_emb[k] || !out->audio_head[k]) return -1;
    }
    return 0;
}

void moss_weights_unload(moss_weight_bundle_t *w) {
    if (w->sf) {
        safetensors_close(w->sf);
        w->sf = NULL;
    }
    memset(w, 0, sizeof(*w));
}
