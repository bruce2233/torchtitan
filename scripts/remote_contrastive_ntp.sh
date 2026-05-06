#!/usr/bin/env bash
set -euo pipefail

HOME_DIR="${HOME:?HOME is not set}"
REPO_DIR="${REPO_DIR:-${HOME_DIR}/torchtitan}"
TMP_ROOT="${TMP_ROOT:-${HOME_DIR}/torchtitan_tmp}"
VENV_DIR="${VENV_DIR:-${REPO_DIR}/.venv}"
DATA_DIR="${DATA_DIR:-${TMP_ROOT}/modded-nanogpt/data/fineweb10B}"
DUMP_ROOT="${DUMP_ROOT:-${TMP_ROOT}/torchtitan_outputs}"

PYPI_INDEX="${PYPI_INDEX:-https://mirrors.aliyun.com/pypi/simple/}"
TORCH_VERSION="${TORCH_VERSION:-2.10.0}"
UV_INSTALL_USER="${UV_INSTALL_USER:-1}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${TMP_ROOT}/uv-cache}"
TMPDIR="${TMPDIR:-${TMP_ROOT}/tmp}"

NUM_TRAIN_SHARDS="${NUM_TRAIN_SHARDS:-103}"
INCLUDE_VAL="${INCLUDE_VAL:-1}"
HF_HOME="${HF_HOME:-${TMP_ROOT}/hf_home}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${TMP_ROOT}/hf_hub}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${TMP_ROOT}/hf_datasets}"

USER_RUN_NAME="${RUN_NAME-}"
USER_STEPS="${STEPS-}"
RUN_NAME="${RUN_NAME:-nanogpt_contrastive_ntp_muon_bf16_seq20k}"
SEQ_LEN="${SEQ_LEN:-20000}"
STEPS="${STEPS:-50000}"
LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1}"
DTYPE="${DTYPE:-bfloat16}"
LOG_FREQ="${LOG_FREQ:-50}"

OPTIMIZER_NAME="${OPTIMIZER_NAME:-MuonAdamW}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
OPTIMIZER_IMPL="${OPTIMIZER_IMPL:-fused}"
MUON_LR="${MUON_LR:-0.02}"

CHECKPOINT_ENABLE="${CHECKPOINT_ENABLE:-1}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-3000}"
KEEP_LATEST_K="${KEEP_LATEST_K:-2}"
BACKGROUND="${BACKGROUND:-1}"
PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/remote_contrastive_ntp.sh setup
  bash scripts/remote_contrastive_ntp.sh data
  bash scripts/remote_contrastive_ntp.sh smoke
  bash scripts/remote_contrastive_ntp.sh train
  bash scripts/remote_contrastive_ntp.sh all

Defaults are non-root:
  REPO_DIR=$HOME/torchtitan
  TMP_ROOT=$HOME/torchtitan_tmp
  DATA_DIR=$HOME/torchtitan_tmp/modded-nanogpt/data/fineweb10B
  DUMP_ROOT=$HOME/torchtitan_tmp/torchtitan_outputs
  PYPI_INDEX=https://mirrors.aliyun.com/pypi/simple/

Common overrides:
  REPO_URL=...                         clone repo if REPO_DIR is missing
  PYTHON_BIN=$HOME/miniconda3/bin/python
  NUM_TRAIN_SHARDS=9                   smaller data download for smoke
  HF_ENDPOINT=https://hf-mirror.com    optional Hugging Face mirror
  SEQ_LEN=20000 STEPS=50000
  OPTIMIZER_NAME=AdamW                 use AdamW instead of MuonAdamW
  BACKGROUND=0                         foreground training
EOF
}

find_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    command -v "${PYTHON_BIN}" >/dev/null 2>&1 || [[ -x "${PYTHON_BIN}" ]]
    printf '%s\n' "${PYTHON_BIN}"
    return
  fi

  for candidate in \
    "${HOME_DIR}/miniconda3/bin/python" \
    "${HOME_DIR}/miniforge3/bin/python" \
    "${HOME_DIR}/mambaforge/bin/python" \
    python3 \
    python; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      command -v "${candidate}"
      return
    fi
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return
    fi
  done

  echo "Could not find Python. Set PYTHON_BIN=/path/to/python." >&2
  exit 1
}

ensure_repo() {
  if [[ ! -d "${REPO_DIR}" ]]; then
    if [[ -z "${REPO_URL:-}" ]]; then
      echo "Missing ${REPO_DIR}. Clone/copy the repo first, or set REPO_URL." >&2
      exit 1
    fi
    git clone "${REPO_URL}" "${REPO_DIR}"
  fi

  if [[ ! -f "${REPO_DIR}/pyproject.toml" ]]; then
    echo "${REPO_DIR} does not look like the TorchTitan repo." >&2
    exit 1
  fi
}

setup_env() {
  ensure_repo
  mkdir -p "${UV_CACHE_DIR}" "${TMPDIR}"

  local python_bin
  python_bin="$(find_python)"

  local pip_args=(install -i "${PYPI_INDEX}" --upgrade uv)
  if [[ "${UV_INSTALL_USER}" == "1" ]]; then
    pip_args+=(--user)
  fi
  "${python_bin}" -m pip "${pip_args[@]}"

  export PATH="${HOME_DIR}/.local/bin:$(dirname "${python_bin}"):${PATH}"
  local uv_bin="${HOME_DIR}/.local/bin/uv"
  if [[ ! -x "${uv_bin}" ]]; then
    uv_bin="$(command -v uv || true)"
  fi
  if [[ -z "${uv_bin}" || ! -x "${uv_bin}" ]]; then
    echo "Could not find uv after installation. Check ${HOME_DIR}/.local/bin or set PATH." >&2
    exit 1
  fi

  UV_CACHE_DIR="${UV_CACHE_DIR}" TMPDIR="${TMPDIR}" \
    "${uv_bin}" venv "${VENV_DIR}" --python "${python_bin}" --clear

  UV_CACHE_DIR="${UV_CACHE_DIR}" TMPDIR="${TMPDIR}" \
    "${uv_bin}" pip install \
    --python "${VENV_DIR}/bin/python" \
    --index-url "${PYPI_INDEX}" \
    --upgrade "torch==${TORCH_VERSION}"

  UV_CACHE_DIR="${UV_CACHE_DIR}" TMPDIR="${TMPDIR}" \
    "${uv_bin}" pip install \
    --python "${VENV_DIR}/bin/python" \
    --index-url "${PYPI_INDEX}" \
    -e "${REPO_DIR}" \
    tiktoken \
    pytest \
    huggingface_hub

  "${VENV_DIR}/bin/python" - <<'PY'
import torch
from torchtitan.config import ConfigManager

cfg = ConfigManager().parse_args(
    ["--module", "llama3", "--config", "llama3_nanogpt_contrastive_ntp"]
)
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
print("config:", type(cfg).__name__)
PY

  echo "Environment ready: ${VENV_DIR}"
}

ensure_venv() {
  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "Missing ${VENV_DIR}/bin/python. Run: bash scripts/remote_contrastive_ntp.sh setup" >&2
    exit 1
  fi
}

download_data() {
  ensure_venv
  mkdir -p "${DATA_DIR}" "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}"

  DATA_DIR="${DATA_DIR}" \
  NUM_TRAIN_SHARDS="${NUM_TRAIN_SHARDS}" \
  INCLUDE_VAL="${INCLUDE_VAL}" \
  HF_HOME="${HF_HOME}" \
  HF_HUB_CACHE="${HF_HUB_CACHE}" \
  HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
  "${VENV_DIR}/bin/python" - <<'PY'
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

data_dir = Path(os.environ["DATA_DIR"])
num_train_shards = int(os.environ["NUM_TRAIN_SHARDS"])
include_val = os.environ["INCLUDE_VAL"] not in {"0", "false", "False", "no", "No"}

if num_train_shards < 0 or num_train_shards > 103:
    raise ValueError("NUM_TRAIN_SHARDS must be in [0, 103]")

filenames = []
if include_val:
    filenames.append("fineweb_val_000000.bin")
filenames.extend(f"fineweb_train_{i:06d}.bin" for i in range(1, num_train_shards + 1))

for filename in filenames:
    path = hf_hub_download(
        repo_id="kjj0/fineweb10B-gpt2",
        filename=filename,
        repo_type="dataset",
        local_dir=str(data_dir),
    )
    size_gb = Path(path).stat().st_size / 1e9
    print(f"{filename}: {size_gb:.2f} GB")

print(f"ready: {data_dir}")
PY

  find "${DATA_DIR}" -maxdepth 1 -name 'fineweb_*.bin' -printf '%f %s\n' | sort
  du -sh "${DATA_DIR}"
}

start_train() {
  ensure_repo
  ensure_venv
  if ! compgen -G "${DATA_DIR}/fineweb_train_*.bin" >/dev/null; then
    echo "Missing train shards under ${DATA_DIR}. Run: bash scripts/remote_contrastive_ntp.sh data" >&2
    exit 1
  fi

  local out_dir="${DUMP_ROOT}/${RUN_NAME}"
  local log_file="${out_dir}/train.log"
  local pid_file="${out_dir}/train.pid"
  mkdir -p "${out_dir}" "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}"

  local optimizer_args=(
    --optimizer.name "${OPTIMIZER_NAME}"
    --optimizer.lr "${LR}"
    --optimizer.weight_decay "${WEIGHT_DECAY}"
    --optimizer.implementation "${OPTIMIZER_IMPL}"
  )
  if [[ "${OPTIMIZER_NAME}" == "MuonAdamW" ]]; then
    optimizer_args+=(--optimizer.muon_lr "${MUON_LR}")
  fi

  local checkpoint_args=(--checkpoint.no-enable)
  if [[ "${CHECKPOINT_ENABLE}" == "1" ]]; then
    checkpoint_args=(
      --checkpoint.enable
      --checkpoint.interval "${CHECKPOINT_INTERVAL}"
      --checkpoint.keep_latest_k "${KEEP_LATEST_K}"
      --checkpoint.no-last-save-model-only
    )
  fi

  local run_env=(
    HF_HOME="${HF_HOME}"
    HF_HUB_CACHE="${HF_HUB_CACHE}"
    HF_DATASETS_CACHE="${HF_DATASETS_CACHE}"
    PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF}"
    LOG_RANK="${LOG_RANK:-0}"
    WANDB_MODE="${WANDB_MODE:-offline}"
  )

  local train_cmd=(
    "${VENV_DIR}/bin/python" -m torch.distributed.run
    --nproc_per_node=1
    --rdzv_backend c10d
    --rdzv_endpoint=localhost:0
    -m torchtitan.train
    --module llama3
    --config llama3_nanogpt_contrastive_ntp
    --dump_folder "${out_dir}"
    --dataloader.dataset_path "${DATA_DIR}/fineweb_train_*.bin"
    --dataloader.no-align-to-bos
    "${optimizer_args[@]}"
    --training.local_batch_size "${LOCAL_BATCH_SIZE}"
    --training.global_batch_size "${GLOBAL_BATCH_SIZE}"
    --training.seq_len "${SEQ_LEN}"
    --training.steps "${STEPS}"
    --training.dtype "${DTYPE}"
    --metrics.log_freq "${LOG_FREQ}"
    "${checkpoint_args[@]}"
  )

  printf 'Repo: %s\n' "${REPO_DIR}"
  printf 'Data: %s\n' "${DATA_DIR}"
  printf 'Output: %s\n' "${out_dir}"
  printf 'Command:\n'
  printf ' %q' env "${run_env[@]}" "${train_cmd[@]}"
  printf '\n'

  if [[ "${BACKGROUND}" == "1" ]]; then
    (
      cd "${REPO_DIR}"
      nohup env "${run_env[@]}" "${train_cmd[@]}" > "${log_file}" 2>&1 < /dev/null &
      echo "$!" > "${pid_file}"
    )
    echo "Started background training. PID: $(cat "${pid_file}")"
    echo "Log: ${log_file}"
  else
    cd "${REPO_DIR}"
    exec env "${run_env[@]}" "${train_cmd[@]}" 2>&1 | tee "${log_file}"
  fi
}

smoke_train() {
  local saved_run_name="${RUN_NAME}"
  local saved_steps="${STEPS}"
  local saved_checkpoint_enable="${CHECKPOINT_ENABLE}"
  local saved_background="${BACKGROUND}"

  RUN_NAME="${SMOKE_RUN_NAME:-${USER_RUN_NAME:-contrastive_ntp_smoke}}"
  STEPS="${SMOKE_STEPS:-${USER_STEPS:-1}}"
  CHECKPOINT_ENABLE=0
  BACKGROUND=0
  start_train

  RUN_NAME="${saved_run_name}"
  STEPS="${saved_steps}"
  CHECKPOINT_ENABLE="${saved_checkpoint_enable}"
  BACKGROUND="${saved_background}"
}

cmd="${1:-}"
case "${cmd}" in
  setup)
    setup_env
    ;;
  data)
    download_data
    ;;
  smoke)
    smoke_train
    ;;
  train)
    start_train
    ;;
  all)
    setup_env
    download_data
    smoke_train
    start_train
    ;;
  -h|--help|help|"")
    usage
    ;;
  *)
    echo "Unknown command: ${cmd}" >&2
    usage >&2
    exit 1
    ;;
esac
