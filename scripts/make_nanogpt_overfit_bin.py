# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Create a small contiguous nanoGPT .bin shard for overfit experiments."""

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from torchtitan.hf_datasets.nanogpt_datasets import (  # noqa: E402
    _HEADER_INT32S,
    _NANOGPT_MAGIC,
    _NANOGPT_VERSION,
    load_nanogpt_bin_tokens,
)


def _load_tokenizer():
    try:
        import tiktoken
    except ImportError:
        return None
    return tiktoken.get_encoding("gpt2")


def _decode(tokenizer: Any | None, ids: list[int]) -> str:
    if tokenizer is None:
        return ""
    return tokenizer.decode(ids)


def _write_nanogpt_bin(path: Path, tokens: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tokens = tokens.to(torch.uint16).cpu()
    header = [_NANOGPT_MAGIC, _NANOGPT_VERSION, int(tokens.numel())]
    header.extend([0] * (_HEADER_INT32S - len(header)))
    with path.open("wb") as f:
        f.write(struct.pack(f"{_HEADER_INT32S}i", *header))
        f.write(tokens.numpy().tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a contiguous nanoGPT overfit shard from a source shard."
    )
    parser.add_argument(
        "--source",
        default="../modded-nanogpt/data/fineweb10B/fineweb_train_000001.bin",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--seq_len", type=int, default=20000)
    parser.add_argument("--num_sequences", type=int, required=True)
    parser.add_argument("--start_token", type=int, default=0)
    parser.add_argument("--prompt_tokens", type=int, default=32)
    parser.add_argument("--num_prompts", type=int, default=3)
    parser.add_argument("--prompt_json", default=None)
    args = parser.parse_args()

    if args.seq_len <= 0:
        raise ValueError("--seq_len must be positive")
    if args.num_sequences <= 0:
        raise ValueError("--num_sequences must be positive")

    source = Path(args.source)
    tokens = load_nanogpt_bin_tokens(source)
    total_tokens = args.num_sequences * args.seq_len + 1
    end = args.start_token + total_tokens
    if end > tokens.numel():
        raise ValueError(
            f"Source shard has {tokens.numel()} payload tokens, need slice end {end}"
        )

    overfit_tokens = tokens[args.start_token:end].clone()
    output = Path(args.output)
    _write_nanogpt_bin(output, overfit_tokens)

    tokenizer = _load_tokenizer()
    prompt_records = []
    for idx in range(min(args.num_prompts, args.num_sequences)):
        start = idx * args.seq_len
        prompt_ids = overfit_tokens[start : start + args.prompt_tokens].to(
            torch.long
        )
        expected_id = int(overfit_tokens[start + args.prompt_tokens].item())
        prompt_list = prompt_ids.tolist()
        prompt_records.append(
            {
                "sequence_index": idx,
                "prompt_token_ids": prompt_list,
                "prompt_text": _decode(tokenizer, prompt_list),
                "expected_token_id": expected_id,
                "expected_token_text": _decode(tokenizer, [expected_id]),
            }
        )

    if args.prompt_json:
        prompt_path = Path(args.prompt_json)
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(
            json.dumps(prompt_records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "output": str(output),
                "source": str(source),
                "seq_len": args.seq_len,
                "num_sequences": args.num_sequences,
                "num_tokens": int(overfit_tokens.numel()),
                "prompt_json": args.prompt_json,
                "prompts": prompt_records,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
