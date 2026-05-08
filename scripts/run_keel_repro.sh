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

MODE="${MODE:-keel_looped_768x32x16_fineweb_edu}"
extra_args=()
case "${MODE}" in
    keel_looped_768x32x16_fineweb_edu)
        CONFIG="llama3_keel_gpt2_looped_768x32x16_fineweb_edu_text"
        default_ngpu=1
        default_fineweb_edu_dataset_path="HuggingFaceFW/fineweb-edu"
        fineweb_edu_cache_root="/data/hf_home/hub/datasets--HuggingFaceFW--fineweb-edu/snapshots"
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
if [[ -d /data/hf_home ]]; then
    export HF_HOME="${HF_HOME:-/data/hf_home}"
fi
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
