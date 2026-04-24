#include "moss_sp_prompt.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <sentencepiece_processor.h>

static void append_vec(std::vector<int> *dst, const std::vector<int> &src) {
    dst->insert(dst->end(), src.begin(), src.end());
}

static void append_one(std::vector<int> *dst, int x) { dst->push_back(x); }

static std::vector<int> encode_sp(const sentencepiece::SentencePieceProcessor &sp, const char *utf8) {
    std::vector<int> ids;
    const std::string s(utf8 ? utf8 : "");
    if (!sp.Encode(s, &ids).ok()) return {};
    return ids;
}

extern "C" int moss_build_prompt_token_ids(
    const char *model_dir,
    const moss_run_config_t *cfg,
    const char *text_utf8,
    int *out_ids,
    int max_ids
) {
    if (!model_dir || !cfg || !text_utf8 || !out_ids || max_ids <= 0) return -1;

    char path[2048];
    if (snprintf(path, sizeof(path), "%s/tokenizer.model", model_dir) >= (int)sizeof(path)) return -2;

    sentencepiece::SentencePieceProcessor sp;
    sentencepiece::util::Status ld = sp.Load(std::string(path));
    if (!ld.ok()) {
        /* New layout: model_dir/checkpoint/tokenizer.model */
        if (snprintf(path, sizeof(path), "%s/checkpoint/tokenizer.model", model_dir) >= (int)sizeof(path)) return -2;
        ld = sp.Load(std::string(path));
        if (!ld.ok()) return -3;
    }

    static const char USER_ROLE_PREFIX[] = "user\n";
    static const char USER_TEMPLATE_REFERENCE_PREFIX[] =
        "<user_inst>\n"
        "- Reference(s):\n";
    static const char USER_TEMPLATE_AFTER_REFERENCE[] =
        "\n- Instruction:\nNone\n"
        "- Tokens:\nNone\n"
        "- Quality:\nNone\n"
        "- Sound Event:\nNone\n"
        "- Ambient Sound:\nNone\n"
        "- Language:\nNone\n"
        "- Text:\n";
    static const char USER_TEMPLATE_SUFFIX[] = "\n</user_inst>";
    static const char ASSISTANT_TURN_PREFIX[] = "\n";
    static const char ASSISTANT_ROLE_PREFIX[] = "assistant\n";

    std::vector<int> prefix;
    append_one(&prefix, cfg->im_start_token_id);
    append_vec(&prefix, encode_sp(sp, USER_ROLE_PREFIX));
    append_vec(&prefix, encode_sp(sp, USER_TEMPLATE_REFERENCE_PREFIX));
    append_vec(&prefix, encode_sp(sp, "None"));
    append_vec(&prefix, encode_sp(sp, USER_TEMPLATE_AFTER_REFERENCE));

    std::vector<int> body = encode_sp(sp, text_utf8);

    std::vector<int> suffix;
    append_vec(&suffix, encode_sp(sp, USER_TEMPLATE_SUFFIX));
    append_one(&suffix, cfg->im_end_token_id);
    append_vec(&suffix, encode_sp(sp, ASSISTANT_TURN_PREFIX));
    append_one(&suffix, cfg->im_start_token_id);
    append_vec(&suffix, encode_sp(sp, ASSISTANT_ROLE_PREFIX));

    const size_t total = prefix.size() + body.size() + suffix.size();
    if (total > (size_t)max_ids) return -4;

    size_t o = 0;
    for (int x : prefix) out_ids[o++] = x;
    for (int x : body) out_ids[o++] = x;
    for (int x : suffix) out_ids[o++] = x;
    return (int)o;
}
