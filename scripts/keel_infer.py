#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Single-process text generation for KEEL/Llama3 TorchTitan checkpoints."""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config import ConfigManager, TORCH_DTYPE_MAP


DEFAULT_CONFIG = "llama3_keel_gpt2_looped_768x32x16_fineweb_edu_text"
TOKENIZER_FILES = ("tokenizer.json", "vocab.json", "vocab.txt")


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run single-process autoregressive inference for KEEL/Llama3 "
            "TorchTitan checkpoints. Unknown args are forwarded as config overrides."
        )
    )
    parser.add_argument("--module", default="llama3")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "DCP checkpoint directory, e.g. outputs/.../checkpoint/step-1000. "
            "Default: latest step under config.dump_folder/config.checkpoint.folder."
        ),
    )
    parser.add_argument(
        "--allow-random",
        "--allow_random",
        action="store_true",
        help="Allow generation with randomly initialized weights if no checkpoint exists.",
    )
    parser.add_argument("--prompt", default="", help="Prompt text.")
    parser.add_argument(
        "--prompt-file",
        "--prompt_file",
        default=None,
        help="Read prompt text from a file instead of --prompt.",
    )
    parser.add_argument("--max-new-tokens", "--max_new_tokens", type=int, default=64)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Use 0 for greedy decoding.",
    )
    parser.add_argument("--top-k", "--top_k", type=int, default=None)
    parser.add_argument("--top-p", "--top_p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu.")
    parser.add_argument(
        "--dtype",
        choices=sorted(TORCH_DTYPE_MAP),
        default=None,
        help="Default: config.training.dtype.",
    )
    parser.add_argument(
        "--hf-assets-path",
        "--hf_assets_path",
        default=None,
        help="Tokenizer directory override. For GPT-2 this should contain tokenizer.json.",
    )
    parser.add_argument(
        "--gpt2-tokenizer-repo",
        "--gpt2_tokenizer_repo",
        default="gpt2",
        help="HF repo used only if a GPT-2 tokenizer auto-download is needed.",
    )
    parser.add_argument(
        "--no-auto-tokenizer-download",
        "--no_auto_tokenizer_download",
        action="store_true",
        help="Disable GPT-2 tokenizer auto-download when hf_assets_path is missing.",
    )
    parser.add_argument(
        "--no-add-bos",
        "--no_add_bos",
        action="store_true",
        help="Do not prepend BOS before encoding the prompt.",
    )
    parser.add_argument(
        "--stop-at-eos",
        "--stop_at_eos",
        action="store_true",
        help="Stop generation when the tokenizer EOS token is generated.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Keep the model loaded and read prompts from stdin.",
    )
    return parser.parse_known_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def tokenizer_ready(tokenizer_path: Path) -> bool:
    if (tokenizer_path / "tokenizer.json").is_file():
        return True
    return (
        (tokenizer_path / "merges.txt").is_file()
        and any((tokenizer_path / name).is_file() for name in TOKENIZER_FILES[1:])
    )


def ensure_gpt2_tokenizer(
    tokenizer_path: Path,
    *,
    repo_id: str,
    enabled: bool,
) -> None:
    if tokenizer_ready(tokenizer_path):
        return
    if not enabled:
        raise FileNotFoundError(
            f"Tokenizer files not found in {tokenizer_path}. "
            "Pass --hf-assets-path or remove --no-auto-tokenizer-download."
        )

    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "huggingface_hub is required to auto-download the GPT-2 tokenizer. "
            "Install it or pass --hf-assets-path to an existing tokenizer."
        ) from exc

    tokenizer_path.mkdir(parents=True, exist_ok=True)
    filenames = (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
    )
    missing = []
    for filename in filenames:
        try:
            hf_hub_download(repo_id=repo_id, filename=filename, local_dir=tokenizer_path)
        except Exception as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code == 404 or exc.__class__.__name__ in {
                "EntryNotFoundError",
                "LocalEntryNotFoundError",
                "RemoteEntryNotFoundError",
            }:
                missing.append(filename)
                continue
            raise

    if not tokenizer_ready(tokenizer_path):
        raise FileNotFoundError(
            f"Downloaded from {repo_id}, but {tokenizer_path} is not a usable "
            "tokenizer directory. Need tokenizer.json or vocab/vocab.json + merges.txt."
        )
    if missing:
        print(f"Missing optional tokenizer files: {missing}", file=sys.stderr)


def find_latest_checkpoint(dump_folder: str, checkpoint_folder: str) -> Path | None:
    checkpoint_root = Path(dump_folder) / checkpoint_folder
    if not checkpoint_root.is_dir():
        return None

    candidates: list[tuple[int, Path]] = []
    for path in checkpoint_root.iterdir():
        if not path.is_dir():
            continue
        match = re.fullmatch(r"step-(\d+)", path.name)
        if match is None:
            continue
        if (path / ".metadata").is_file() or (
            path / "model.safetensors.index.json"
        ).is_file():
            candidates.append((int(match.group(1)), path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def build_config(args: argparse.Namespace, config_overrides: list[str]):
    config_args = ["--module", args.module, "--config", args.config]
    if args.hf_assets_path is not None:
        config_args += ["--hf_assets_path", args.hf_assets_path]
    config_args += config_overrides
    return ConfigManager().parse_args(config_args)


def cast_model_dtype(model: torch.nn.Module, dtype: torch.dtype) -> None:
    for param in model.parameters():
        param.data = param.data.to(dtype=dtype)
    for module in model.modules():
        for name, buffer in module._buffers.items():
            if buffer is not None and torch.is_floating_point(buffer):
                module._buffers[name] = buffer.to(dtype=dtype)


def build_model(config, *, device: torch.device, dtype: torch.dtype):
    model_config = config.model_spec.model
    model_config.update_from_config(trainer_config=config)

    with torch.device("meta"):
        model = model_config.build()

    model.to_empty(device=device)
    with torch.no_grad():
        model.init_weights(buffer_device=device)
    cast_model_dtype(model, dtype)
    model.eval()
    return model


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> None:
    begin = time.monotonic()
    state_dict = model.state_dict()
    dcp.load(state_dict, checkpoint_id=str(checkpoint_path))
    elapsed = time.monotonic() - begin
    print(f"Loaded checkpoint {checkpoint_path} in {elapsed:.2f}s", file=sys.stderr)


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}.")
    if top_p == 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    remove_mask = cumulative_probs > top_p
    remove_mask[..., 1:] = remove_mask[..., :-1].clone()
    remove_mask[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove_mask, -torch.inf)
    return torch.empty_like(logits).scatter(-1, sorted_indices, sorted_logits)


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if temperature < 0:
        raise ValueError(f"temperature must be >= 0, got {temperature}.")

    logits = logits / max(temperature, 1e-5)
    if top_k is not None:
        if top_k < 1:
            raise ValueError(f"top_k must be positive, got {top_k}.")
        cutoff, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < cutoff[..., [-1]], -torch.inf)
    if top_p is not None:
        logits = apply_top_p(logits, top_p)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


@torch.inference_mode()
def generate_tokens(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    eos_id: int | None,
    stop_at_eos: bool,
    generator: torch.Generator | None,
) -> torch.Tensor:
    tokens = input_ids
    for _ in range(max_new_tokens):
        logits = model(tokens)[:, -1, :]
        next_token = sample_next_token(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=generator,
        )
        tokens = torch.cat((tokens, next_token), dim=1)
        if stop_at_eos and eos_id is not None and bool(torch.all(next_token == eos_id)):
            break
    return tokens


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file is not None:
        return Path(args.prompt_file).read_text()
    return args.prompt


def run_prompt(
    prompt: str,
    *,
    model: torch.nn.Module,
    tokenizer: HuggingFaceTokenizer,
    args: argparse.Namespace,
    config,
    device: torch.device,
    generator: torch.Generator | None,
) -> str:
    token_ids = tokenizer.encode(
        prompt,
        add_bos=not args.no_add_bos,
        add_eos=False,
    )
    if not token_ids:
        raise ValueError("Prompt encoded to an empty token sequence.")

    total_tokens = len(token_ids) + args.max_new_tokens
    if total_tokens > config.training.seq_len:
        raise ValueError(
            f"prompt tokens ({len(token_ids)}) + max_new_tokens "
            f"({args.max_new_tokens}) exceeds config.training.seq_len "
            f"({config.training.seq_len}). Pass --training.seq_len {total_tokens} "
            "or reduce --max-new-tokens."
        )

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    output_ids = generate_tokens(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_id=tokenizer.eos_id,
        stop_at_eos=args.stop_at_eos,
        generator=generator,
    )[0]
    new_ids = output_ids[len(token_ids) :].tolist()
    return tokenizer.decode(new_ids)


def main() -> None:
    args, config_overrides = parse_args()
    config = build_config(args, config_overrides)

    tokenizer_path = Path(config.hf_assets_path)
    should_auto_download = (
        not args.no_auto_tokenizer_download and "gpt2" in args.config.lower()
    )
    ensure_gpt2_tokenizer(
        tokenizer_path,
        repo_id=args.gpt2_tokenizer_repo,
        enabled=should_auto_download,
    )
    tokenizer = HuggingFaceTokenizer.Config().build(tokenizer_path=str(tokenizer_path))

    prompt = read_prompt(args)
    token_count = len(tokenizer.encode(prompt, add_bos=not args.no_add_bos))
    requested_seq_len = token_count + args.max_new_tokens
    if requested_seq_len > config.training.seq_len:
        raise ValueError(
            f"Requested sequence length {requested_seq_len} exceeds "
            f"config.training.seq_len={config.training.seq_len}. "
            f"Pass --training.seq_len {requested_seq_len}."
        )

    device = resolve_device(args.device)
    dtype_name = args.dtype or config.training.dtype
    dtype = TORCH_DTYPE_MAP[dtype_name]
    print(
        f"Building {args.config} on {device} with dtype={dtype_name}",
        file=sys.stderr,
    )
    model = build_model(config, device=device, dtype=dtype)

    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint is not None
        else find_latest_checkpoint(config.dump_folder, config.checkpoint.folder)
    )
    if checkpoint_path is None:
        if not args.allow_random:
            raise FileNotFoundError(
                "No checkpoint found. Pass --checkpoint outputs/.../checkpoint/step-N "
                "or use --allow-random for an untrained smoke test."
            )
        print("No checkpoint loaded; using random initialized weights.", file=sys.stderr)
    else:
        load_checkpoint(model, checkpoint_path)

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    if args.interactive:
        print("Enter prompts. Ctrl-D exits.", file=sys.stderr)
        while True:
            try:
                prompt = input("> ")
            except EOFError:
                break
            print(
                run_prompt(
                    prompt,
                    model=model,
                    tokenizer=tokenizer,
                    args=args,
                    config=config,
                    device=device,
                    generator=generator,
                ),
                flush=True,
            )
        return

    print(
        run_prompt(
            prompt,
            model=model,
            tokenizer=tokenizer,
            args=args,
            config=config,
            device=device,
            generator=generator,
        )
    )


if __name__ == "__main__":
    main()
