# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inference helper for batch-local contrastive NTP checkpoints.

The contrastive NTP experiment trains hidden states against token embeddings
instead of the full-vocabulary lm_head. This script therefore skips lm_head at
inference time and ranks next tokens by retrieval against tok_embeddings.weight.
"""

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

# Support running from an uninstalled checkout.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from torchtitan.config import ConfigManager  # noqa: E402


GPT2_EOT_TOKEN = 50256


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def _load_gpt2_tokenizer():
    try:
        import tiktoken
    except ImportError as exc:
        raise RuntimeError(
            "contrastive_ntp_infer.py requires tiktoken for GPT-2 tokenization. "
            "Run it in the modded-nanogpt virtualenv."
        ) from exc

    return tiktoken.get_encoding("gpt2")


def _encode_prompt(tokenizer: Any, prompt: str, add_bos: bool) -> list[int]:
    token_ids = tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
    if add_bos:
        token_ids = [GPT2_EOT_TOKEN] + token_ids
    if not token_ids:
        token_ids = [GPT2_EOT_TOKEN]
    return token_ids


def _safe_decode_token(tokenizer: Any, token_id: int) -> str:
    if token_id < 0 or token_id >= tokenizer.n_vocab:
        return f"<invalid:{token_id}>"
    return tokenizer.decode([token_id])


def _build_model(module: str, config_name: str, device: torch.device):
    config = ConfigManager().parse_args(["--module", module, "--config", config_name])
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

    state_dict = model.state_dict()
    begin = time.monotonic()
    print(f"Loading checkpoint: {checkpoint}", file=sys.stderr)
    dcp.load(state_dict, checkpoint_id=str(checkpoint_path))
    print(
        f"Loaded checkpoint in {time.monotonic() - begin:.2f}s",
        file=sys.stderr,
    )


def _autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype in (torch.bfloat16, torch.float16):
        return torch.autocast(device_type=device.type, dtype=dtype)
    return nullcontext()


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

    # nanogpt_smoke pads GPT-2's 50,257-token vocab to 50,304. Those padded ids
    # never appear in FineWeb targets, so keep them out of retrieval by default.
    if valid_vocab_size < logits.shape[-1]:
        logits[:, valid_vocab_size:] = -torch.inf

    return logits


def _topk_report(
    tokenizer: Any,
    logits: torch.Tensor,
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    probs = torch.softmax(logits.float(), dim=-1)
    values, indices = torch.topk(logits, k=min(top_k, logits.shape[-1]), dim=-1)
    top_probs = probs.gather(dim=-1, index=indices)

    rows = []
    for rank, (token_id, score, prob) in enumerate(
        zip(indices[0].tolist(), values[0].tolist(), top_probs[0].tolist()),
        start=1,
    ):
        rows.append(
            {
                "rank": rank,
                "token_id": token_id,
                "token_text": _safe_decode_token(tokenizer, token_id),
                "score": score,
                "prob": prob,
            }
        )
    return rows


def _sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int | None,
    sample: bool,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if top_k is not None:
        values, indices = torch.topk(logits, k=min(top_k, logits.shape[-1]), dim=-1)
        logits_for_sample = values
    else:
        indices = None
        logits_for_sample = logits

    if not sample:
        next_idx = torch.argmax(logits_for_sample, dim=-1, keepdim=True)
    else:
        probs = torch.softmax(logits_for_sample.float() / max(temperature, 1e-6), dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1, generator=generator)

    if indices is not None:
        return indices.gather(dim=-1, index=next_idx)
    return next_idx


@torch.no_grad()
def _generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    tokenizer: Any,
    max_new_tokens: int,
    max_context_tokens: int,
    tau: float,
    normalize: bool,
    temperature: float,
    top_k: int | None,
    sample: bool,
    seed: int | None,
    valid_vocab_size: int,
) -> torch.Tensor:
    generated = input_ids.clone()
    generator = None
    if seed is not None:
        generator = torch.Generator(device=input_ids.device).manual_seed(seed)

    for _ in range(max_new_tokens):
        context = generated[:, -max_context_tokens:]
        logits = _next_token_logits(
            model,
            context,
            tau=tau,
            normalize=normalize,
            valid_vocab_size=valid_vocab_size,
        )
        next_token = _sample_next_token(
            logits,
            temperature=temperature,
            top_k=top_k,
            sample=sample,
            generator=generator,
        )
        generated = torch.cat([generated, next_token], dim=1)
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank or generate next tokens from a contrastive NTP checkpoint."
    )
    parser.add_argument("--module", default="llama3")
    parser.add_argument("--config", default="llama3_nanogpt_contrastive_ntp")
    parser.add_argument(
        "--checkpoint",
        default="outputs/nanogpt_contrastive_ntp_100m_20k/checkpoint/step-5000",
        help="DCP checkpoint directory to load.",
    )
    parser.add_argument(
        "--allow_random_init",
        action="store_true",
        help="Run without loading a checkpoint. Intended only for smoke tests.",
    )
    parser.add_argument("--prompt", default="The meaning of life is")
    parser.add_argument("--no_add_bos", action="store_true")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=0)
    parser.add_argument("--max_context_tokens", type=int, default=2048)
    parser.add_argument("--tau", type=float, default=0.07)
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16", "float16"),
        default="bfloat16" if torch.cuda.is_available() else "float32",
    )
    parser.add_argument(
        "--include_padded_vocab",
        action="store_true",
        help="Allow GPT-2 padded vocab ids 50257..50303 to be retrieved.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output only.")
    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer = _load_gpt2_tokenizer()
    token_ids = _encode_prompt(tokenizer, args.prompt, add_bos=not args.no_add_bos)
    input_ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)

    model = _build_model(args.module, args.config, device)
    if not args.allow_random_init:
        _load_checkpoint(model, args.checkpoint)
    else:
        print("WARNING: running with randomly initialized weights", file=sys.stderr)

    dtype = _dtype_from_name(args.dtype)

    valid_vocab_size = (
        model.tok_embeddings.weight.shape[0]  # pyrefly: ignore[missing-attribute]
        if args.include_padded_vocab
        else tokenizer.n_vocab
    )
    normalize = not args.no_normalize

    rank_context = input_ids[:, -args.max_context_tokens :]
    with _autocast_context(device, dtype):
        logits = _next_token_logits(
            model,
            rank_context,
            tau=args.tau,
            normalize=normalize,
            valid_vocab_size=valid_vocab_size,
        )
        top_candidates = _topk_report(tokenizer, logits, top_k=args.top_k)

    generated_text = None
    generated_token_ids = None
    if args.max_new_tokens > 0:
        with _autocast_context(device, dtype):
            generated = _generate(
                model,
                input_ids,
                tokenizer=tokenizer,
                max_new_tokens=args.max_new_tokens,
                max_context_tokens=args.max_context_tokens,
                tau=args.tau,
                normalize=normalize,
                temperature=args.temperature,
                top_k=args.top_k,
                sample=args.sample,
                seed=args.seed,
                valid_vocab_size=valid_vocab_size,
            )
        generated_token_ids = generated[0].tolist()
        generated_text = tokenizer.decode(
            [t for t in generated_token_ids if 0 <= t < tokenizer.n_vocab]
        )

    output = {
        "checkpoint": None if args.allow_random_init else args.checkpoint,
        "prompt": args.prompt,
        "prompt_token_ids": token_ids,
        "top_candidates": top_candidates,
        "generated_token_ids": generated_token_ids,
        "generated_text": generated_text,
        "settings": {
            "module": args.module,
            "config": args.config,
            "device": str(device),
            "dtype": args.dtype,
            "tau": args.tau,
            "normalize": normalize,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "max_context_tokens": args.max_context_tokens,
            "sample": args.sample,
            "temperature": args.temperature,
            "valid_vocab_size": valid_vocab_size,
        },
    }

    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return

    print(f"Prompt: {args.prompt!r}")
    print("Top next-token candidates:")
    for row in top_candidates:
        print(
            f"{row['rank']:>2}. id={row['token_id']:<5} "
            f"text={row['token_text']!r:<16} "
            f"score={row['score']:.4f} prob={row['prob']:.6f}"
        )
    if generated_text is not None:
        print("\nGenerated text:")
        print(generated_text)


if __name__ == "__main__":
    main()
