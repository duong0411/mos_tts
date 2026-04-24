#!/usr/bin/env python3
"""Emit prompt token ids (same as prompting.build_prompt_token_ids). Args: model_dir path, path to UTF-8 text file."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import sentencepiece as spm

USER_ROLE_PREFIX = "user\n"
USER_TEMPLATE_REFERENCE_PREFIX = (
    "<user_inst>\n"
    "- Reference(s):\n"
)
USER_TEMPLATE_AFTER_REFERENCE = (
    "\n- Instruction:\nNone\n"
    "- Tokens:\nNone\n"
    "- Quality:\nNone\n"
    "- Sound Event:\nNone\n"
    "- Ambient Sound:\nNone\n"
    "- Language:\nNone\n"
    "- Text:\n"
)
USER_TEMPLATE_SUFFIX = "\n</user_inst>"
ASSISTANT_TURN_PREFIX = "\n"
ASSISTANT_ROLE_PREFIX = "assistant\n"


def encode_sp(sp: spm.SentencePieceProcessor, text: str) -> list[int]:
    return list(sp.encode(text, out_type=int))


def build_user_prompt_prefix(sp: spm.SentencePieceProcessor, cfg: dict) -> list[int]:
    return (
        [int(cfg["im_start_token_id"])]
        + encode_sp(sp, USER_ROLE_PREFIX)
        + encode_sp(sp, USER_TEMPLATE_REFERENCE_PREFIX)
    )


def build_user_prompt_after_reference(sp: spm.SentencePieceProcessor) -> list[int]:
    return encode_sp(sp, USER_TEMPLATE_AFTER_REFERENCE)


def build_assistant_prompt_prefix(sp: spm.SentencePieceProcessor, cfg: dict) -> list[int]:
    return (
        encode_sp(sp, USER_TEMPLATE_SUFFIX)
        + [int(cfg["im_end_token_id"])]
        + encode_sp(sp, ASSISTANT_TURN_PREFIX)
        + [int(cfg["im_start_token_id"])]
        + encode_sp(sp, ASSISTANT_ROLE_PREFIX)
    )


def build_prompt_prefix(sp: spm.SentencePieceProcessor, cfg: dict) -> list[int]:
    return (
        build_user_prompt_prefix(sp, cfg)
        + encode_sp(sp, "None")
        + build_user_prompt_after_reference(sp)
    )


def build_prompt_suffix(sp: spm.SentencePieceProcessor, cfg: dict) -> list[int]:
    return build_assistant_prompt_prefix(sp, cfg)


def build_prompt_token_ids(sp: spm.SentencePieceProcessor, cfg: dict, text_token_ids: list[int]) -> list[int]:
    return build_prompt_prefix(sp, cfg) + [int(t) for t in text_token_ids] + build_prompt_suffix(sp, cfg)


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: moss_prompt_tokens.py MODEL_DIR TEXT_FILE", file=sys.stderr)
        sys.exit(2)
    wd = Path(sys.argv[1]).resolve()
    text = Path(sys.argv[2]).read_text(encoding="utf-8")
    cfg = json.loads((wd / "config.json").read_text(encoding="utf-8"))
    sp = spm.SentencePieceProcessor(model_file=str(wd / "tokenizer.model"))
    body_ids = encode_sp(sp, text)
    full = build_prompt_token_ids(sp, cfg, body_ids)
    print(len(full), flush=True)
    for x in full:
        print(int(x), flush=True)


if __name__ == "__main__":
    main()
