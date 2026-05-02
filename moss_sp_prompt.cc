#include "moss_sp_prompt.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_set>
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

extern "C" int moss_build_voice_clone_sections(
    const char *model_dir,
    const moss_run_config_t *cfg,
    const char *text_utf8,
    int *out_prefix_ids,
    int max_prefix_ids,
    int *out_suffix_ids,
    int max_suffix_ids,
    int *out_prefix_len,
    int *out_suffix_len
) {
    if (!model_dir || !cfg || !text_utf8 || !out_prefix_ids || !out_suffix_ids || !out_prefix_len || !out_suffix_len) return -1;
    if (max_prefix_ids <= 0 || max_suffix_ids <= 0) return -1;

    char path[2048];
    if (snprintf(path, sizeof(path), "%s/tokenizer.model", model_dir) >= (int)sizeof(path)) return -2;
    sentencepiece::SentencePieceProcessor sp;
    sentencepiece::util::Status ld = sp.Load(std::string(path));
    if (!ld.ok()) {
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
    append_one(&prefix, cfg->audio_start_token_id);

    std::vector<int> suffix;
    append_one(&suffix, cfg->audio_end_token_id);
    append_vec(&suffix, encode_sp(sp, USER_TEMPLATE_AFTER_REFERENCE));
    append_vec(&suffix, encode_sp(sp, text_utf8));
    append_vec(&suffix, encode_sp(sp, USER_TEMPLATE_SUFFIX));
    append_one(&suffix, cfg->im_end_token_id);
    append_vec(&suffix, encode_sp(sp, ASSISTANT_TURN_PREFIX));
    append_one(&suffix, cfg->im_start_token_id);
    append_vec(&suffix, encode_sp(sp, ASSISTANT_ROLE_PREFIX));
    append_one(&suffix, cfg->audio_start_token_id);

    if ((int)prefix.size() > max_prefix_ids || (int)suffix.size() > max_suffix_ids) return -4;
    for (size_t i = 0; i < prefix.size(); i++) out_prefix_ids[i] = prefix[i];
    for (size_t i = 0; i < suffix.size(); i++) out_suffix_ids[i] = suffix[i];
    *out_prefix_len = (int)prefix.size();
    *out_suffix_len = (int)suffix.size();
    return 0;
}

static char *dup_cstr(const char *s) {
    if (!s) return nullptr;
    size_t n = std::strlen(s);
    char *p = (char *)std::malloc(n + 1);
    if (!p) return nullptr;
    std::memcpy(p, s, n + 1);
    return p;
}

static bool utf8_decode_one(const std::string &s, size_t &i, uint32_t *out) {
    if (i >= s.size()) return false;
    unsigned char c0 = (unsigned char)s[i];
    if (c0 < 0x80u) {
        *out = c0;
        i += 1;
        return true;
    }
    if ((c0 >> 5) == 6u && i + 1 < s.size()) {
        unsigned char c1 = (unsigned char)s[i + 1];
        if ((c1 >> 6) != 2u) return false;
        *out = ((uint32_t)(c0 & 0x1fu) << 6) | (uint32_t)(c1 & 0x3fu);
        i += 2;
        return true;
    }
    if ((c0 >> 4) == 0xeu && i + 2 < s.size()) {
        unsigned char c1 = (unsigned char)s[i + 1];
        unsigned char c2 = (unsigned char)s[i + 2];
        if ((c1 >> 6) != 2u || (c2 >> 6) != 2u) return false;
        *out = ((uint32_t)(c0 & 0x0fu) << 12) | ((uint32_t)(c1 & 0x3fu) << 6) | (uint32_t)(c2 & 0x3fu);
        i += 3;
        return true;
    }
    if ((c0 >> 3) == 0x1eu && i + 3 < s.size()) {
        unsigned char c1 = (unsigned char)s[i + 1];
        unsigned char c2 = (unsigned char)s[i + 2];
        unsigned char c3 = (unsigned char)s[i + 3];
        if ((c1 >> 6) != 2u || (c2 >> 6) != 2u || (c3 >> 6) != 2u) return false;
        *out = ((uint32_t)(c0 & 0x07u) << 18) | ((uint32_t)(c1 & 0x3fu) << 12) | ((uint32_t)(c2 & 0x3fu) << 6)
            | (uint32_t)(c3 & 0x3fu);
        i += 4;
        return true;
    }
    *out = (uint32_t)c0;
    i += 1;
    return true;
}

static std::vector<uint32_t> utf8_to_u32(const std::string &s) {
    std::vector<uint32_t> out;
    size_t i = 0;
    while (i < s.size()) {
        uint32_t cp;
        if (!utf8_decode_one(s, i, &cp)) break;
        out.push_back(cp);
    }
    return out;
}

static void append_utf8(std::string *dst, uint32_t cp) {
    if (cp <= 0x7fu) {
        *dst += (char)cp;
    } else if (cp <= 0x7ffu) {
        *dst += (char)(0xc0u | ((cp >> 6) & 0x1fu));
        *dst += (char)(0x80u | (cp & 0x3fu));
    } else if (cp <= 0xffffu) {
        *dst += (char)(0xe0u | ((cp >> 12) & 0x0fu));
        *dst += (char)(0x80u | ((cp >> 6) & 0x3fu));
        *dst += (char)(0x80u | (cp & 0x3fu));
    } else {
        *dst += (char)(0xf0u | ((cp >> 18) & 0x07u));
        *dst += (char)(0x80u | ((cp >> 12) & 0x3fu));
        *dst += (char)(0x80u | ((cp >> 6) & 0x3fu));
        *dst += (char)(0x80u | (cp & 0x3fu));
    }
}

static std::string u32_to_utf8(const std::vector<uint32_t> &v) {
    std::string s;
    for (uint32_t cp : v) append_utf8(&s, cp);
    return s;
}

static bool is_space_u32(uint32_t cp) {
    if (cp == 0x20u || cp == 0x09u || cp == 0x0au || cp == 0x0du) return true;
    if (cp == 0x3000u) return true;
    return false;
}

static void trim_u32(std::vector<uint32_t> *v) {
    while (!v->empty() && is_space_u32(v->front())) v->erase(v->begin());
    while (!v->empty() && is_space_u32(v->back())) v->pop_back();
}

static std::string trim_ascii_ws_copy(const std::string &s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) a++;
    while (b > a && std::isspace((unsigned char)s[b - 1])) b--;
    return s.substr(a, b - a);
}

static bool contains_cjk_u32(const std::vector<uint32_t> &v) {
    for (uint32_t cp : v) {
        if (cp >= 0x4e00u && cp <= 0x9fffu) return true;
        if (cp >= 0x3400u && cp <= 0x4dbfu) return true;
        if (cp >= 0x3040u && cp <= 0x30ffu) return true;
        if (cp >= 0xac00u && cp <= 0xd7afu) return true;
    }
    return false;
}

static bool contains_cjk_str(const std::string &s) { return contains_cjk_u32(utf8_to_u32(s)); }

static std::string prepare_text_for_sentence_chunking(const std::string &text_in) {
    std::string normalized_text = trim_ascii_ws_copy(text_in);
    for (char &c : normalized_text) {
        if (c == '\n' || c == '\r') c = ' ';
    }
    while (normalized_text.find("  ") != std::string::npos)
        normalized_text.replace(normalized_text.find("  "), 2, " ");

    if (normalized_text.empty()) return normalized_text;

    if (contains_cjk_str(normalized_text)) {
        static const std::unordered_set<uint32_t> sent_end = {'.', '!', '?', 0x3002u, 0xff01u, 0xff1fu, 0xff1bu, ';'};
        std::vector<uint32_t> u = utf8_to_u32(normalized_text);
        uint32_t last_cp = u.empty() ? 0u : u.back();
        if (sent_end.count(last_cp) == 0) normalized_text.append("\xe3\x80\x82");
        return normalized_text;
    }

    if (!normalized_text.empty() && std::isalpha((unsigned char)normalized_text[0])) {
        if (!std::isupper((unsigned char)normalized_text[0]))
            normalized_text[0] = (char)std::toupper((unsigned char)normalized_text[0]);
    }
    if (!normalized_text.empty()) {
        unsigned char last = (unsigned char)normalized_text.back();
        if (std::isalnum(last)) normalized_text.push_back('.');
    }
    {
        int words = 0;
        bool iw = false;
        for (unsigned char c : normalized_text) {
            if (std::isspace(c))
                iw = false;
            else {
                if (!iw) {
                    words++;
                    iw = true;
                }
            }
        }
        if (words < 5) normalized_text = std::string(8, ' ') + normalized_text;
    }
    return normalized_text;
}

static int count_tokens_sp(const sentencepiece::SentencePieceProcessor &sp, const std::string &t) {
    std::vector<int> ids;
    if (!sp.Encode(t, &ids).ok()) return 1000000000;
    return (int)ids.size();
}

static std::vector<std::string> split_text_by_punctuation_u32(
    const std::vector<uint32_t> &text,
    const std::unordered_set<uint32_t> &punctuation,
    const std::unordered_set<uint32_t> &closing
) {
    std::vector<std::string> sentences;
    std::vector<uint32_t> current_chars;
    size_t index = 0;
    while (index < text.size()) {
        uint32_t ch = text[index];
        current_chars.push_back(ch);
        if (punctuation.count(ch)) {
            size_t lookahead = index + 1;
            while (lookahead < text.size() && closing.count(text[lookahead])) {
                current_chars.push_back(text[lookahead]);
                lookahead++;
            }
            trim_u32(&current_chars);
            if (!current_chars.empty()) sentences.push_back(u32_to_utf8(current_chars));
            current_chars.clear();
            while (lookahead < text.size() && is_space_u32(text[lookahead])) lookahead++;
            index = lookahead;
            continue;
        }
        index++;
    }
    trim_u32(&current_chars);
    if (!current_chars.empty()) sentences.push_back(u32_to_utf8(current_chars));
    return sentences;
}

static std::vector<std::string> split_text_by_token_budget(
    const sentencepiece::SentencePieceProcessor &sp,
    const std::string &text,
    int max_tokens
) {
    std::vector<std::string> pieces;
    std::string remaining_text = trim_ascii_ws_copy(text);
    while (!remaining_text.empty()) {
        if (count_tokens_sp(sp, remaining_text) <= max_tokens) {
            pieces.push_back(remaining_text);
            break;
        }
        int low = 1;
        int high = (int)utf8_to_u32(remaining_text).size();
        if (high < 1) high = 1;
        int best_prefix_length = 1;
        while (low <= high) {
            int middle = (low + high) / 2;
            std::vector<uint32_t> u = utf8_to_u32(remaining_text);
            if (middle > (int)u.size()) middle = (int)u.size();
            std::vector<uint32_t> pref(u.begin(), u.begin() + middle);
            std::string candidate = trim_ascii_ws_copy(u32_to_utf8(pref));
            if (candidate.empty()) {
                low = middle + 1;
                continue;
            }
            if (count_tokens_sp(sp, candidate) <= max_tokens) {
                best_prefix_length = middle;
                low = middle + 1;
            } else {
                high = middle - 1;
            }
        }
        int cut_index = best_prefix_length;
        std::vector<uint32_t> u_full = utf8_to_u32(remaining_text);
        if (cut_index > (int)u_full.size()) cut_index = (int)u_full.size();
        std::vector<uint32_t> pref_prefix(u_full.begin(), u_full.begin() + cut_index);
        static const std::unordered_set<uint32_t> preferred_boundary = {
            ',', 0xff0cu, 0x3001u, 0xff1bu, ';', ':', 0xff1au, '.', '!', '?', 0x3002u, 0xff01u, 0xff1fu, 0x20u};
        int preferred_index = -1;
        int pref_u32_len = (int)pref_prefix.size();
        int scan_lo = std::max(-1, pref_u32_len - 25);
        for (int scan_index = pref_u32_len - 1; scan_index > scan_lo; scan_index--) {
            if (preferred_boundary.count(pref_prefix[(size_t)scan_index])) {
                preferred_index = scan_index + 1;
                break;
            }
        }
        if (preferred_index > 0) cut_index = preferred_index;

        std::vector<uint32_t> cut_u(u_full.begin(), u_full.begin() + cut_index);
        std::string piece = trim_ascii_ws_copy(u32_to_utf8(cut_u));
        if (piece.empty()) {
            piece = trim_ascii_ws_copy(u32_to_utf8(std::vector<uint32_t>(u_full.begin(),
                u_full.begin() + best_prefix_length)));
            cut_index = best_prefix_length;
        }
        pieces.push_back(piece);
        std::vector<uint32_t> rest_u(u_full.begin() + cut_index, u_full.end());
        remaining_text = trim_ascii_ws_copy(u32_to_utf8(rest_u));
    }
    return pieces;
}

static std::string join_sentence_parts(const std::string &left, const std::string &right) {
    if (left.empty()) return right;
    if (right.empty()) return left;
    if (contains_cjk_str(left) || contains_cjk_str(right)) return left + right;
    return left + " " + right;
}

static int load_sp_ro(const char *model_dir, sentencepiece::SentencePieceProcessor *sp) {
    char path[2048];
    if (snprintf(path, sizeof(path), "%s/tokenizer.model", model_dir) >= (int)sizeof(path)) return -1;
    if (sp->Load(std::string(path)).ok()) return 0;
    if (snprintf(path, sizeof(path), "%s/checkpoint/tokenizer.model", model_dir) >= (int)sizeof(path)) return -1;
    return sp->Load(std::string(path)).ok() ? 0 : -2;
}

extern "C" float moss_voice_clone_inter_chunk_pause_seconds(const char *chunk_utf8) {
    if (!chunk_utf8) return 0.24f;
    std::string t = trim_ascii_ws_copy(std::string(chunk_utf8));
    int cnt = 0;
    bool in_word = false;
    for (unsigned char c : t) {
        if (std::isspace(c))
            in_word = false;
        else {
            if (!in_word) {
                cnt++;
                in_word = true;
            }
        }
    }
    return (cnt <= 4) ? 0.40f : 0.24f;
}

extern "C" int moss_voice_clone_split_text(
    const char *model_dir,
    const char *text_utf8,
    int max_tokens,
    char ***out_chunks,
    int *out_n_chunks
) {
    if (!model_dir || !text_utf8 || !out_chunks || !out_n_chunks) return -1;
    *out_chunks = nullptr;
    *out_n_chunks = 0;
    if (max_tokens <= 0) {
        char **arr = (char **)std::malloc(sizeof(char *));
        if (!arr) return -2;
        arr[0] = dup_cstr(text_utf8);
        if (!arr[0]) {
            std::free(arr);
            return -2;
        }
        *out_chunks = arr;
        *out_n_chunks = 1;
        return 0;
    }

    sentencepiece::SentencePieceProcessor sp;
    if (load_sp_ro(model_dir, &sp) != 0) return -3;

    std::string prepared = prepare_text_for_sentence_chunking(std::string(text_utf8));
    if (prepared.empty()) return -4;

    static const std::unordered_set<uint32_t> sentence_end = {'.', '!', '?', 0x3002u, 0xff01u, 0xff1fu, 0xff1bu, ';'};
    static const std::unordered_set<uint32_t> clause_split = {
        ',', 0xff0cu, 0x3001u, 0xff1bu, ';', ':', 0xff1au};
    static const std::unordered_set<uint32_t> closing = {'"', '\'', 0x201cu, 0x201du, 0x2018u, 0x2019u,
        ')', ']', '}', 0xff09u, 0xff5du, 0x300du, 0x300fu, 0x300bu};

    std::vector<uint32_t> prep_u = utf8_to_u32(prepared);
    std::vector<std::string> sentence_candidates = split_text_by_punctuation_u32(prep_u, sentence_end, closing);
    if (sentence_candidates.empty()) sentence_candidates.push_back(trim_ascii_ws_copy(prepared));

    std::vector<std::pair<int, std::string>> sentence_slices;
    for (const std::string &sentence_text : sentence_candidates) {
        std::string normalized_sentence = trim_ascii_ws_copy(sentence_text);
        if (normalized_sentence.empty()) continue;
        int sentence_token_count = count_tokens_sp(sp, normalized_sentence);
        if (sentence_token_count <= max_tokens) {
            sentence_slices.push_back({sentence_token_count, normalized_sentence});
            continue;
        }
        std::vector<std::string> clause_candidates =
            split_text_by_punctuation_u32(utf8_to_u32(normalized_sentence), clause_split, closing);
        if (clause_candidates.size() <= 1) clause_candidates = {normalized_sentence};

        for (const std::string &clause_text : clause_candidates) {
            std::string normalized_clause = trim_ascii_ws_copy(clause_text);
            if (normalized_clause.empty()) continue;
            int clause_token_count = count_tokens_sp(sp, normalized_clause);
            if (clause_token_count <= max_tokens) {
                sentence_slices.push_back({clause_token_count, normalized_clause});
                continue;
            }
            std::vector<std::string> budget_pieces =
                split_text_by_token_budget(sp, normalized_clause, max_tokens);
            for (const std::string &piece : budget_pieces) {
                std::string normalized_piece = trim_ascii_ws_copy(piece);
                if (!normalized_piece.empty())
                    sentence_slices.push_back({count_tokens_sp(sp, normalized_piece), normalized_piece});
            }
        }
    }

    std::vector<std::string> chunks;
    std::string current_chunk;
    int current_chunk_token_count = 0;
    for (const auto &pr : sentence_slices) {
        int sentence_token_count = pr.first;
        const std::string &sentence_text = pr.second;
        if (current_chunk.empty()) {
            current_chunk = sentence_text;
            current_chunk_token_count = sentence_token_count;
            continue;
        }
        if (current_chunk_token_count + sentence_token_count > max_tokens) {
            chunks.push_back(trim_ascii_ws_copy(current_chunk));
            current_chunk = sentence_text;
            current_chunk_token_count = sentence_token_count;
        } else {
            current_chunk = join_sentence_parts(current_chunk, sentence_text);
            current_chunk_token_count = count_tokens_sp(sp, current_chunk);
        }
    }
    if (!current_chunk.empty()) chunks.push_back(trim_ascii_ws_copy(current_chunk));

    if (chunks.empty()) chunks.push_back(trim_ascii_ws_copy(prepared));

    *out_n_chunks = (int)chunks.size();
    char **arr = (char **)std::calloc((size_t)chunks.size(), sizeof(char *));
    if (!arr) return -2;
    for (size_t i = 0; i < chunks.size(); i++) {
        arr[i] = dup_cstr(chunks[i].c_str());
        if (!arr[i]) {
            for (size_t j = 0; j < i; j++) std::free(arr[j]);
            std::free(arr);
            return -2;
        }
    }
    *out_chunks = arr;
    return 0;
}

extern "C" void moss_voice_clone_free_split(char **chunks, int n_chunks) {
    if (!chunks) return;
    for (int i = 0; i < n_chunks; i++) std::free(chunks[i]);
    std::free(chunks);
}
