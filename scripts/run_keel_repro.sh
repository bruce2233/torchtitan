#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  ./scripts/run_keel_repro.sh [TorchTitan overrides...]

Modes:
  keel_looped_768x32x16_fineweb_edu
                     Default. KEEL Post-LN looped GPT-2 shape:
                     hidden=768, physical blocks=32, loop count=16,
                     logical blocks=512, seq_len=4096, steps=1000,
                     full activation checkpointing.
  smoke              Local KEEL debug run on c4_test, forced to 3 steps.
  keel_512x1024_1t   Paper-scale 512-sub-layer KEEL config.
  preln_512x1024_1t  Paper-scale 512-sub-layer Pre-LN baseline.

Environment:
  MODE               One of the modes above. Default: keel_looped_768x32x16_fineweb_edu.
  NGPU               GPUs on this node. Default: 1 for default/smoke, 64 for paper-scale modes.
  FINEWEB_EDU_DATASET_PATH
                     FineWeb-Edu path for the default looped mode. Default prefers
                     local HF cache parquet shards at:
                     /data/hf_home/hub/datasets--HuggingFaceFW--fineweb-edu/snapshots/*/sample/10BT/*.parquet
                     If that glob is missing, it falls back to HuggingFaceFW/fineweb-edu,
                     which uses HF streaming and can hit the network.
  ENABLE_WANDB       Enable W&B metrics for the default looped mode. Default: 1.
  WANDB_PROJECT      W&B project. Default: torchtitan-keel.
  WANDB_RUN_GROUP    W&B run group. Default: keel-fineweb-edu.
  WANDB_RUN_NAME     W&B run name. Default: MODE.
  GPT2_TOKENIZER_PATH
                     GPT-2 tokenizer directory for KEEL FineWeb text configs.
                     Default: ./outputs/gpt2_tokenizer. If missing, the default
                     looped mode downloads tokenizer files there before launch.
  GPT2_TOKENIZER_REPO
                     HF repo used when downloading GPT-2 tokenizer files.
                     Default: gpt2.
  HF_DATASETS_CACHE  Hugging Face dataset cache. Default: ./outputs/hf_datasets_cache.
  HF_HOME            Hugging Face hub cache root. Default: /data/hf_home if present.
  LOG_RANK           Ranks printed by run_train.sh. Default comes from run_train.sh.

Changing loop/layers/hidden size:
  Model shape is selected by config.model_spec and is not a CLI override.
  Edit torchtitan/models/llama3/__init__.py:
    - _keel_gpt2_looped_768x32x16(): dim controls hidden size.
    - n_layers controls physical block count.
    - block_loop_count controls fixed loops per physical block.
    - n_heads should divide dim, e.g. dim=768 uses n_heads=12.
  Then register or reuse the flavor in llama3_configs and point the training config
  in torchtitan/models/llama3/config_registry.py at that flavor.
  Logical blocks = n_layers * block_loop_count, and KEEL residual alpha defaults
  to 2 * logical blocks unless residual_scale is set explicitly.

Examples:
  ./scripts/run_keel_repro.sh
  MODE=smoke NGPU=1 ./scripts/run_keel_repro.sh
  COMM_MODE=fake_backend MODE=keel_512x1024_1t NGPU=64 ./scripts/run_keel_repro.sh
  ENABLE_WANDB=0 ./scripts/run_keel_repro.sh
  FINEWEB_EDU_DATASET_PATH='/data/fineweb_edu/*.parquet' ./scripts/run_keel_repro.sh
  ./scripts/run_keel_repro.sh --training.steps 2000 --training.seq_len 8192
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

has_cli_override() {
    local flag="$1"
    shift
    local arg
    for arg in "$@"; do
        if [[ "${arg}" == "${flag}" || "${arg}" == "${flag}="* ]]; then
            return 0
        fi
    done
    return 1
}

gpt2_tokenizer_ready() {
    local tokenizer_path="$1"
    [[ -f "${tokenizer_path}/tokenizer.json" ]] || \
        [[ -f "${tokenizer_path}/vocab.json" && -f "${tokenizer_path}/merges.txt" ]]
}

ensure_gpt2_tokenizer() {
    GPT2_TOKENIZER_PATH="${GPT2_TOKENIZER_PATH:-${PWD}/outputs/gpt2_tokenizer}"
    GPT2_TOKENIZER_REPO="${GPT2_TOKENIZER_REPO:-gpt2}"
    export GPT2_TOKENIZER_PATH GPT2_TOKENIZER_REPO

    if gpt2_tokenizer_ready "${GPT2_TOKENIZER_PATH}"; then
        return
    fi

    echo "GPT-2 tokenizer is missing at ${GPT2_TOKENIZER_PATH}; downloading from ${GPT2_TOKENIZER_REPO}."
    mkdir -p "${GPT2_TOKENIZER_PATH}"
    python - <<'PY'
import os
import sys

try:
    from huggingface_hub import hf_hub_download
except ModuleNotFoundError:
    sys.exit(
        "huggingface_hub is required to download the GPT-2 tokenizer. "
        "Install it or set GPT2_TOKENIZER_PATH to an existing tokenizer directory."
    )

repo_id = os.environ["GPT2_TOKENIZER_REPO"]
local_dir = os.environ["GPT2_TOKENIZER_PATH"]
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
        hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
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

has_tokenizer_json = os.path.isfile(os.path.join(local_dir, "tokenizer.json"))
has_bpe_pair = all(
    os.path.isfile(os.path.join(local_dir, filename))
    for filename in ("vocab.json", "merges.txt")
)
if not (has_tokenizer_json or has_bpe_pair):
    sys.exit(
        f"Downloaded from {repo_id}, but {local_dir} is not a usable GPT-2 "
        "tokenizer directory. Need tokenizer.json or vocab.json + merges.txt."
    )

print(f"GPT-2 tokenizer ready at {local_dir}. Missing optional files: {missing}")
PY
}

if [[ -d /data/hf_home ]]; then
    export HF_HOME="${HF_HOME:-/data/hf_home}"
fi

MODE="${MODE:-keel_looped_768x32x16_fineweb_edu}"
extra_args=()
case "${MODE}" in
    keel_looped_768x32x16_fineweb_edu)
        CONFIG="llama3_keel_gpt2_looped_768x32x16_fineweb_edu_text"
        default_ngpu=1
        default_fineweb_edu_dataset_path="HuggingFaceFW/fineweb-edu"
        fineweb_edu_cache_root="${HF_HOME:-/data/hf_home}/hub/datasets--HuggingFaceFW--fineweb-edu/snapshots"
        if [[ -d "${fineweb_edu_cache_root}" ]]; then
            fineweb_edu_cached_glob="${fineweb_edu_cache_root}"/*/sample/10BT/*.parquet
            if compgen -G "${fineweb_edu_cached_glob}" >/dev/null; then
                default_fineweb_edu_dataset_path="${fineweb_edu_cached_glob}"
            fi
        fi
        FINEWEB_EDU_DATASET_PATH="${FINEWEB_EDU_DATASET_PATH:-${default_fineweb_edu_dataset_path}}"
        extra_args+=(
            --dataloader.dataset fineweb_edu
            --dataloader.dataset_path "${FINEWEB_EDU_DATASET_PATH}"
            --metrics.log_freq 1
            --metrics.disable-color-printing
        )
        if ! has_cli_override "--hf_assets_path" "$@"; then
            ensure_gpt2_tokenizer
            extra_args+=(--hf_assets_path "${GPT2_TOKENIZER_PATH}")
        fi
        if [[ "${ENABLE_WANDB:-1}" == "1" ]]; then
            export WANDB_PROJECT="${WANDB_PROJECT:-torchtitan-keel}"
            export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-keel-fineweb-edu}"
            export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${MODE}}"
            export WANDB_SILENT="${WANDB_SILENT:-true}"
            extra_args+=(--metrics.enable_wandb)
        fi
        ;;
    smoke)
        CONFIG="llama3_keel_debugmodel"
        extra_args+=(--training.steps 3)
        default_ngpu=1
        ;;
    keel_512x1024_1t)
        CONFIG="llama3_keel_512x1024_1t"
        default_ngpu=64
        ;;
    preln_512x1024_1t)
        CONFIG="llama3_preln_512x1024_1t"
        default_ngpu=64
        ;;
    *)
        echo "Unknown MODE=${MODE}" >&2
        echo "Expected one of: keel_looped_768x32x16_fineweb_edu, smoke, keel_512x1024_1t, preln_512x1024_1t" >&2
        exit 2
        ;;
esac

NGPU="${NGPU:-${default_ngpu}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${PWD}/outputs/hf_datasets_cache}"
mkdir -p "${HF_DATASETS_CACHE}"

echo "Launching KEEL reproduction: MODE=${MODE} CONFIG=${CONFIG} NGPU=${NGPU}"
if [[ -n "${FINEWEB_EDU_DATASET_PATH:-}" ]]; then
    echo "FINEWEB_EDU_DATASET_PATH=${FINEWEB_EDU_DATASET_PATH}"
fi
if [[ -n "${HF_HOME:-}" ]]; then
    echo "HF_HOME=${HF_HOME}"
fi
echo "HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"

exec env MODULE=llama3 CONFIG="${CONFIG}" NGPU="${NGPU}" \
    ./run_train.sh "${extra_args[@]}" "$@"
