#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  MODE=smoke NGPU=1 ./scripts/run_keel_repro.sh [TorchTitan overrides...]

Modes:
  smoke              Local KEEL debug run on c4_test, forced to 3 steps.
  keel_512x1024_1t   Paper-scale 512-sub-layer KEEL config.
  preln_512x1024_1t  Paper-scale 512-sub-layer Pre-LN baseline.

Environment:
  MODE               One of the modes above. Default: smoke.
  NGPU               GPUs on this node. Default: 1 for smoke, 64 otherwise.
  HF_DATASETS_CACHE  Hugging Face dataset cache. Default: ./outputs/hf_datasets_cache.
  LOG_RANK           Ranks printed by run_train.sh. Default comes from run_train.sh.

Examples:
  MODE=smoke NGPU=1 ./scripts/run_keel_repro.sh
  COMM_MODE=fake_backend MODE=keel_512x1024_1t NGPU=64 ./scripts/run_keel_repro.sh
  MODE=keel_512x1024_1t NGPU=8 ./scripts/run_keel_repro.sh \
      --dataloader.dataset fineweb_edu \
      --dataloader.dataset_path '/data/fineweb_edu/*.parquet'
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

MODE="${MODE:-smoke}"
extra_args=()
case "${MODE}" in
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
        echo "Expected one of: smoke, keel_512x1024_1t, preln_512x1024_1t" >&2
        exit 2
        ;;
esac

NGPU="${NGPU:-${default_ngpu}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${PWD}/outputs/hf_datasets_cache}"
mkdir -p "${HF_DATASETS_CACHE}"

echo "Launching KEEL reproduction: MODE=${MODE} CONFIG=${CONFIG} NGPU=${NGPU}"
echo "HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"

exec env MODULE=llama3 CONFIG="${CONFIG}" NGPU="${NGPU}" \
    ./run_train.sh "${extra_args[@]}" "$@"
