# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Monitor long contrastive overfit runs from TorchTitan training logs."""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(r"step:\s*(\d+)")
LOCAL_ACC_RE = re.compile(r"local_acc:\s*([0-9.]+)")


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def _parse_log(log_path: Path) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    if not log_path.is_file():
        return rows

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = _strip_ansi(line)
            step_match = STEP_RE.search(line)
            acc_match = LOCAL_ACC_RE.search(line)
            if step_match is None or acc_match is None:
                continue
            rows.append((int(step_match.group(1)), float(acc_match.group(1))))
    return rows


def _latest_rolling(
    rows: list[tuple[int, float]],
    *,
    window: int,
) -> tuple[int, float, float] | None:
    if len(rows) < window:
        return None

    seen: dict[int, float] = {}
    for step, acc in rows:
        seen[step] = acc

    ordered = sorted(seen.items())
    recent: deque[tuple[int, float]] = deque(maxlen=window)
    latest: tuple[int, float, float] | None = None
    for step, acc in ordered:
        recent.append((step, acc))
        if len(recent) < window:
            continue
        steps = [item[0] for item in recent]
        if steps[-1] - steps[0] != window - 1:
            continue
        rolling_acc = sum(item[1] for item in recent) / window
        latest = (step, acc, rolling_acc)
    return latest


def _threshold_crossing(
    rows: list[tuple[int, float]],
    *,
    window: int,
    target_acc: float,
) -> tuple[int, float] | None:
    seen: dict[int, float] = {}
    for step, acc in rows:
        seen[step] = acc

    recent: deque[tuple[int, float]] = deque(maxlen=window)
    for step, acc in sorted(seen.items()):
        recent.append((step, acc))
        if len(recent) < window:
            continue
        steps = [item[0] for item in recent]
        if steps[-1] - steps[0] != window - 1:
            continue
        rolling_acc = sum(item[1] for item in recent) / window
        if rolling_acc >= target_acc:
            return step, rolling_acc
    return None


def _ready_checkpoints(checkpoint_dir: Path) -> list[int]:
    if not checkpoint_dir.is_dir():
        return []
    steps = []
    for path in checkpoint_dir.glob("step-*"):
        if not path.is_dir() or not (path / ".metadata").is_file():
            continue
        try:
            steps.append(int(path.name.split("-", 1)[1]))
        except (IndexError, ValueError):
            continue
    return sorted(steps)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _stop_training(pid_file: Path) -> None:
    pid = int(pid_file.read_text(encoding="utf-8").strip())
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return
    os.killpg(pgid, signal.SIGTERM)


def _run_final_eval(args: argparse.Namespace, checkpoint_step: int) -> None:
    checkpoint = (
        Path(args.run_dir) / args.checkpoint_folder / f"step-{checkpoint_step}"
    )
    output = Path(args.run_dir) / f"eval_train_step{checkpoint_step}.json"
    cmd = [
        args.python,
        args.eval_script,
        "--checkpoint",
        str(checkpoint),
        "--data",
        args.data,
        "--seq_len",
        str(args.seq_len),
        "--num_sequences",
        str(args.num_sequences),
        "--prompt_tokens",
        str(args.prompt_tokens),
        "--num_prompts",
        str(args.num_prompts),
        "--top_k",
        str(args.top_k),
        "--tau",
        str(args.tau),
        "--lambda_t2c",
        str(args.lambda_t2c),
        "--dtype",
        args.dtype,
    ]
    with output.open("w", encoding="utf-8") as f:
        subprocess.run(cmd, check=True, stdout=f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stop a long overfit run after train local_acc reaches target."
    )
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--pid_file", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint_folder", default="checkpoint")
    parser.add_argument("--checkpoint_interval", type=int, default=5000)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--target_acc", type=float, default=0.9)
    parser.add_argument("--poll_seconds", type=float, default=60.0)
    parser.add_argument("--report_jsonl", required=True)
    parser.add_argument("--warm_start_steps", type=int, default=0)
    parser.add_argument("--eval_after_stop", action="store_true")
    parser.add_argument("--eval_script", default="scripts/eval_contrastive_ntp_dataset.py")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seq_len", type=int, default=20000)
    parser.add_argument("--num_sequences", type=int, default=64)
    parser.add_argument("--prompt_tokens", type=int, default=32)
    parser.add_argument("--num_prompts", type=int, default=3)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--tau", type=float, default=0.07)
    parser.add_argument("--lambda_t2c", type=float, default=1.0)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    log_path = Path(args.log)
    run_dir = Path(args.run_dir)
    checkpoint_dir = run_dir / args.checkpoint_folder
    pid_file = Path(args.pid_file)
    report_jsonl = Path(args.report_jsonl)
    stopped = False

    while True:
        rows = _parse_log(log_path)
        latest = _latest_rolling(rows, window=args.window)
        crossing = _threshold_crossing(
            rows,
            window=args.window,
            target_acc=args.target_acc,
        )
        checkpoints = _ready_checkpoints(checkpoint_dir)
        latest_checkpoint = checkpoints[-1] if checkpoints else None

        record: dict[str, Any] = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_logged_steps": len(rows),
            "latest_checkpoint": latest_checkpoint,
        }
        if latest is not None:
            step, local_acc, rolling_acc = latest
            record.update(
                {
                    "latest_step": step,
                    "latest_total_step_including_warm_start": (
                        step + args.warm_start_steps
                    ),
                    "latest_local_acc": local_acc,
                    f"rolling{args.window}_local_acc": rolling_acc,
                }
            )
        if crossing is not None:
            crossing_step, crossing_acc = crossing
            stop_checkpoint = (
                (crossing_step + args.checkpoint_interval - 1)
                // args.checkpoint_interval
                * args.checkpoint_interval
            )
            record.update(
                {
                    "threshold_crossing_step": crossing_step,
                    "threshold_crossing_total_step_including_warm_start": (
                        crossing_step + args.warm_start_steps
                    ),
                    "threshold_crossing_acc": crossing_acc,
                    "target_stop_checkpoint": stop_checkpoint,
                }
            )
            if latest_checkpoint is not None and latest_checkpoint >= stop_checkpoint:
                record["action"] = "stop_training"
                _append_jsonl(report_jsonl, record)
                _stop_training(pid_file)
                stopped = True
                if args.eval_after_stop:
                    time.sleep(20)
                    _run_final_eval(args, latest_checkpoint)
                break

        _append_jsonl(report_jsonl, record)
        if not pid_file.exists():
            break
        try:
            os.kill(int(pid_file.read_text(encoding="utf-8").strip()), 0)
        except (ProcessLookupError, ValueError):
            break
        time.sleep(args.poll_seconds)

    if not stopped:
        _append_jsonl(
            report_jsonl,
            {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "action": "monitor_exit_without_stop",
            },
        )


if __name__ == "__main__":
    main()
