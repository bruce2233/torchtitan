# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Evaluate contrastive NTP checkpoints on a nanoGPT token shard."""

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from torchtitan.components.loss import _contrastive_ntp_loss_from_hidden  # noqa: E402
from torchtitan.config import ConfigManager  # noqa: E402
from torchtitan.hf_datasets.nanogpt_datasets import (  # noqa: E402
    load_nanogpt_bin_tokens,
)


GPT2_EOT_TOKEN = 50256


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def _autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype in (torch.bfloat16, torch.float16):
        return torch.autocast(device_type=device.type, dtype=dtype)
    return nullcontext()


def _load_gpt2_tokenizer():
    try:
        import tiktoken
    except ImportError:
        return None
    return tiktoken.get_encoding("gpt2")


def _decode(tokenizer: Any | None, token_ids: list[int]) -> str:
    if tokenizer is None:
        return ""
    valid_ids = [token_id for token_id in token_ids if 0 <= token_id < tokenizer.n_vocab]
    return tokenizer.decode(valid_ids)


def _safe_decode_token(tokenizer: Any | None, token_id: int) -> str:
    return _decode(tokenizer, [token_id]) if tokenizer is not None else ""


def _build_model(
    module: str,
    config_name: str,
    device: torch.device,
    *,
    seq_len: int,
):
    config = ConfigManager().parse_args(
        [
            "--module",
            module,
            "--config",
            config_name,
            "--training.seq_len",
            str(seq_len),
        ]
    )
    model_config = config.model_spec.model  # pyrefly: ignore[missing-attribute]
    model_config.update_from_config(trainer_config=config)

    with torch.device("meta"):
        model = model_config.build()

    model.to_empty(device=device)
    with torch.no_grad():
        model.init_weights(buffer_device=device)

    model._skip_lm_head = True  # pyrefly: ignore[missing-attribute]
    model.eval()
    return model


def _load_checkpoint(model: torch.nn.Module, checkpoint: str) -> None:
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint}")

    begin = time.monotonic()
    print(f"Loading checkpoint: {checkpoint}", file=sys.stderr)
    dcp.load(model.state_dict(), checkpoint_id=str(checkpoint_path))
    print(
        f"Loaded checkpoint in {time.monotonic() - begin:.2f}s",
        file=sys.stderr,
    )


@torch.no_grad()
def _sequence_metrics(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    tau: float,
    normalize: bool,
    lambda_t2c: float,
) -> dict[str, float]:
    hidden = model(input_ids)
    loss, metrics = _contrastive_ntp_loss_from_hidden(
        hidden,
        labels,
        model.tok_embeddings,  # pyrefly: ignore[missing-attribute]
        tau=tau,
        normalize=normalize,
        lambda_t2c=lambda_t2c,
        ignore_index=None,
        reduction="mean",
    )
    out = {
        "loss": float(loss.detach().float().cpu()),
    }
    for key, value in metrics.items():
        out[key] = float(value.detach().float().cpu())
    return out


@torch.no_grad()
def _next_token_logits(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    tau: float,
    normalize: bool,
    valid_vocab_size: int,
) -> torch.Tensor:
    hidden = model(input_ids)
    query = hidden[:, -1, :]
    candidate_emb = model.tok_embeddings.weight  # pyrefly: ignore[missing-attribute]

    if normalize:
        query = F.normalize(query, dim=-1)
        candidate_emb = F.normalize(candidate_emb, dim=-1)

    logits = query @ candidate_emb.T
    logits = logits / tau
    if valid_vocab_size < logits.shape[-1]:
        logits[:, valid_vocab_size:] = -torch.inf
    return logits


@torch.no_grad()
def _prompt_rank_report(
    model: torch.nn.Module,
    tokenizer: Any | None,
    tokens: torch.Tensor,
    *,
    sequence_index: int,
    start: int,
    prompt_tokens: int,
    device: torch.device,
    tau: float,
    normalize: bool,
    valid_vocab_size: int,
    top_k: int,
) -> dict[str, Any]:
    prompt = tokens[start : start + prompt_tokens].to(torch.long)
    expected_id = int(tokens[start + prompt_tokens].item())
    input_ids = prompt.to(device=device).unsqueeze(0)
    logits = _next_token_logits(
        model,
        input_ids,
        tau=tau,
        normalize=normalize,
        valid_vocab_size=valid_vocab_size,
    )[0]
    expected_score = float(logits[expected_id].detach().float().cpu())
    rank = int((logits > logits[expected_id]).sum().item()) + 1
    values, indices = torch.topk(logits, k=min(top_k, valid_vocab_size), dim=-1)
    top_candidates = []
    for idx, (token_id, score) in enumerate(
        zip(indices.tolist(), values.detach().float().cpu().tolist()),
        start=1,
    ):
        top_candidates.append(
            {
                "rank": idx,
                "token_id": int(token_id),
                "token_text": _safe_decode_token(tokenizer, int(token_id)),
                "score": float(score),
            }
        )

    prompt_list = prompt.tolist()
    return {
        "sequence_index": sequence_index,
        "prompt_token_ids": prompt_list,
        "prompt_text": _decode(tokenizer, prompt_list),
        "expected_token_id": expected_id,
        "expected_token_text": _safe_decode_token(tokenizer, expected_id),
        "expected_score": expected_score,
        "expected_rank": rank,
        "expected_in_top1": rank == 1,
        "expected_in_top5": rank <= 5,
        "top_candidates": top_candidates,
    }


def _mean_dict(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate contrastive NTP train-shard overfitting."
    )
    parser.add_argument("--module", default="llama3")
    parser.add_argument("--config", default="llama3_nanogpt_contrastive_ntp")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--seq_len", type=int, default=20000)
    parser.add_argument("--num_sequences", type=int, default=None)
    parser.add_argument("--max_eval_sequences", type=int, default=None)
    parser.add_argument("--prompt_tokens", type=int, default=32)
    parser.add_argument("--num_prompts", type=int, default=3)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--tau", type=float, default=0.07)
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--lambda_t2c", type=float, default=1.0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16", "float16"),
        default="bfloat16" if torch.cuda.is_available() else "float32",
    )
    parser.add_argument("--valid_vocab_size", type=int, default=50257)
    args = parser.parse_args()

    if args.seq_len <= 0:
        raise ValueError("--seq_len must be positive")
    if args.prompt_tokens <= 0:
        raise ValueError("--prompt_tokens must be positive")
    if args.prompt_tokens >= args.seq_len:
        raise ValueError("--prompt_tokens must be smaller than --seq_len")

    device = torch.device(args.device)
    dtype = _dtype_from_name(args.dtype)
    normalize = not args.no_normalize

    tokens = load_nanogpt_bin_tokens(Path(args.data))
    available_sequences = max(0, (tokens.numel() - 1) // args.seq_len)
    num_sequences = args.num_sequences or available_sequences
    num_sequences = min(num_sequences, available_sequences)
    if args.max_eval_sequences is not None:
        num_sequences = min(num_sequences, args.max_eval_sequences)
    if num_sequences <= 0:
        raise ValueError("No complete sequences available for evaluation")

    model = _build_model(args.module, args.config, device, seq_len=args.seq_len)
    _load_checkpoint(model, args.checkpoint)
    tokenizer = _load_gpt2_tokenizer()

    rows: list[dict[str, float]] = []
    prompt_reports = []
    with _autocast_context(device, dtype):
        for sequence_index in range(num_sequences):
            start = sequence_index * args.seq_len
            x = tokens[start : start + args.seq_len].to(
                device=device, dtype=torch.long
            )
            y = tokens[start + 1 : start + args.seq_len + 1].to(
                device=device, dtype=torch.long
            )
            rows.append(
                _sequence_metrics(
                    model,
                    x.unsqueeze(0),
                    y.unsqueeze(0),
                    tau=args.tau,
                    normalize=normalize,
                    lambda_t2c=args.lambda_t2c,
                )
            )

        for sequence_index in range(min(args.num_prompts, num_sequences)):
            start = sequence_index * args.seq_len
            prompt_reports.append(
                _prompt_rank_report(
                    model,
                    tokenizer,
                    tokens,
                    sequence_index=sequence_index,
                    start=start,
                    prompt_tokens=args.prompt_tokens,
                    device=device,
                    tau=args.tau,
                    normalize=normalize,
                    valid_vocab_size=args.valid_vocab_size,
                    top_k=args.top_k,
                )
            )

    mean_metrics = _mean_dict(rows)
    if "num_candidates" in mean_metrics and mean_metrics["num_candidates"] > 0:
        mean_metrics["random_top1_baseline"] = 1.0 / mean_metrics["num_candidates"]

    output = {
        "checkpoint": args.checkpoint,
        "data": args.data,
        "settings": {
            "seq_len": args.seq_len,
            "num_sequences": num_sequences,
            "prompt_tokens": args.prompt_tokens,
            "tau": args.tau,
            "normalize": normalize,
            "lambda_t2c": args.lambda_t2c,
            "device": str(device),
            "dtype": args.dtype,
            "valid_vocab_size": args.valid_vocab_size,
        },
        "mean_metrics": mean_metrics,
        "sequence_metrics": rows,
        "prompt_rank_reports": prompt_reports,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
