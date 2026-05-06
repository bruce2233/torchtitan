# Contrastive NTP Remote Training Runbook

This runbook reproduces the remote single-GPU training setup for the symmetric
batch-local contrastive NTP experiment on a normal non-root account. All default
paths stay under `~`; no `/root`, no system install, and no data sync with
`rsync`. Everything is driven by one script:

```bash
bash scripts/remote_contrastive_ntp.sh <setup|data|smoke|train|all>
```

## Paths

```bash
export REPO_DIR=$HOME/torchtitan
export TMP_ROOT=$HOME/torchtitan_tmp
export DATA_DIR=$HOME/torchtitan_tmp/modded-nanogpt/data/fineweb10B
export DUMP_ROOT=$HOME/torchtitan_tmp/torchtitan_outputs
```

## 1. Put The Repo On The Remote

Preferred path is Git, because the data is downloaded separately and should not
be copied from another machine:

```bash
git clone git@github.com:bruce2233/torchtitan-contrastive-ntp.git ~/torchtitan
cd ~/torchtitan
git checkout main
```

If GitHub SSH auth is not configured, use HTTPS or upload a source tarball:

```bash
# local machine
tar \
  --exclude .venv \
  --exclude outputs \
  --exclude __pycache__ \
  -czf /tmp/torchtitan-src.tgz \
  -C /data/app torchtitan
scp /tmp/torchtitan-src.tgz <USER>@<HOST>:~/

# remote machine
tar -xzf ~/torchtitan-src.tgz -C ~
cd ~/torchtitan
```

Do not sync FineWeb shards with `rsync`; use the download script in section 3.

## 2. Reproduce The Environment

This creates `~/torchtitan/.venv`, installs `uv` into the user site with the
Aliyun PyPI mirror, then installs `torch==2.10.0` and the TorchTitan
dependencies. The `uv` cache and temporary files go under `~/torchtitan_tmp`.

```bash
cd ~/torchtitan
bash scripts/remote_contrastive_ntp.sh setup
```

Useful overrides:

```bash
PYTHON_BIN=$HOME/miniconda3/bin/python \
TORCH_VERSION=2.10.0 \
PYPI_INDEX=https://mirrors.aliyun.com/pypi/simple/ \
bash scripts/remote_contrastive_ntp.sh setup
```

If the Python environment does not allow `pip --user`, disable user install for
`uv`:

```bash
UV_INSTALL_USER=0 bash scripts/remote_contrastive_ntp.sh setup
```

The script verifies that `llama3_nanogpt_contrastive_ntp` can be parsed and that
CUDA is visible.

## 3. Reproduce The FineWeb Data

The experiment reads modded-nanogpt GPT-2-tokenized FineWeb `.bin` shards from
Hugging Face dataset repo `kjj0/fineweb10B-gpt2`. Full FineWeb10B is 103 train
shards, roughly 20.6 GB on disk. For a quick smoke test, set
`NUM_TRAIN_SHARDS=1` or `9`.

```bash
cd ~/torchtitan
NUM_TRAIN_SHARDS=103 bash scripts/remote_contrastive_ntp.sh data
```

If Hugging Face access is slow, set a mirror before running:

```bash
export HF_ENDPOINT=https://hf-mirror.com
NUM_TRAIN_SHARDS=103 bash scripts/remote_contrastive_ntp.sh data
```

Expected files:

```text
~/torchtitan_tmp/modded-nanogpt/data/fineweb10B/fineweb_train_000001.bin
...
~/torchtitan_tmp/modded-nanogpt/data/fineweb10B/fineweb_train_000103.bin
~/torchtitan_tmp/modded-nanogpt/data/fineweb10B/fineweb_val_000000.bin
```

## 4. One-Step Smoke Test

Run one step without checkpointing:

```bash
cd ~/torchtitan
bash scripts/remote_contrastive_ntp.sh smoke
```

The smoke should print contrastive metrics such as `contrastive/loss_c2t`,
`contrastive/loss_t2c`, `contrastive/local_acc`, `contrastive/local_acc5`,
`contrastive/num_candidates`, and `contrastive/num_queries`.

## 5. Start The Main Training

Default training is:

- `llama3_nanogpt_contrastive_ntp`
- causal SDPA, no row/document isolation mask
- symmetric contrastive loss: context-to-token CE plus token-to-context
  multi-positive InfoNCE
- batch-local unique target token candidates
- `seq_len=20000`
- `local_batch_size=1`
- non-FP8 `bfloat16` training
- `MuonAdamW`
- checkpoint every 3000 steps, keep latest 2

Start in the background:

```bash
cd ~/torchtitan
RUN_NAME=nanogpt_contrastive_ntp_muon_bf16_seq20k_50k \
SEQ_LEN=20000 \
STEPS=50000 \
CHECKPOINT_INTERVAL=3000 \
KEEP_LATEST_K=2 \
bash scripts/remote_contrastive_ntp.sh train
```

For AdamW instead of Muon:

```bash
OPTIMIZER_NAME=AdamW bash scripts/remote_contrastive_ntp.sh train
```

TorchTitan's `TrainingConfig` currently exposes `bfloat16` and `float32`; this
run uses `bfloat16` as the normal non-FP8 training mode.

## 6. Monitor Or Stop

```bash
OUT=$HOME/torchtitan_tmp/torchtitan_outputs/nanogpt_contrastive_ntp_muon_bf16_seq20k_50k
tail -f "$OUT/train.log"
cat "$OUT/train.pid"
nvidia-smi
```

Stop the background run:

```bash
kill "$(cat "$OUT/train.pid")"
```

Checkpoints are under:

```text
$OUT/checkpoint/step-3000
$OUT/checkpoint/step-6000
...
```

## 7. Notes

`--dataloader.no-align-to-bos` means samples are contiguous `seq_len` windows
over the token stream. The candidate set for both contrastive directions is the
unique target token ids inside the current 20K-token window.

This is not a full-vocabulary language model objective. Do not use perplexity as
the primary metric; use local contrastive accuracy and candidate count.

If a run OOMs, first reduce `SEQ_LEN`; the contrastive logits scale as
`num_queries * num_unique_targets`, so very long windows increase both attention
and loss-layer memory.
