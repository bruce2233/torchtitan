# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.tools.logging import logger


_NANOGPT_MAGIC = 20240520
_NANOGPT_VERSION = 1
_HEADER_INT32S = 256
_HEADER_BYTES = _HEADER_INT32S * 4
_HEADER_UINT16S = _HEADER_BYTES // 2


def _resolve_token_files(dataset_path: str) -> list[Path]:
    path = Path(dataset_path)
    if any(ch in dataset_path for ch in "*?[]"):
        files = [Path(p) for p in glob.glob(dataset_path)]
    elif path.is_dir():
        files = list(path.glob("*.bin"))
    else:
        files = [path]

    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No nanoGPT token files found for {dataset_path!r}")

    return files


def load_nanogpt_bin_tokens(file: Path) -> torch.Tensor:
    """Return the token payload from a modded-nanogpt FineWeb .bin shard."""
    header = torch.from_file(
        str(file), shared=False, size=_HEADER_INT32S, dtype=torch.int32
    )
    if header.numel() != _HEADER_INT32S:
        raise ValueError(f"{file} is too small to contain a nanoGPT header")

    magic = int(header[0].item())
    version = int(header[1].item())
    num_tokens = int(header[2].item())
    if magic != _NANOGPT_MAGIC:
        raise ValueError(f"{file} has bad magic {magic}; expected {_NANOGPT_MAGIC}")
    if version != _NANOGPT_VERSION:
        raise ValueError(
            f"{file} has unsupported version {version}; expected {_NANOGPT_VERSION}"
        )
    if num_tokens <= 0:
        raise ValueError(f"{file} declares non-positive token count {num_tokens}")

    expected_bytes = _HEADER_BYTES + num_tokens * 2
    actual_bytes = file.stat().st_size
    if actual_bytes < expected_bytes:
        raise ValueError(
            f"{file} is truncated: expected at least {expected_bytes} bytes, "
            f"found {actual_bytes}"
        )

    raw = torch.from_file(
        str(file),
        shared=False,
        size=_HEADER_UINT16S + num_tokens,
        dtype=torch.uint16,
    )
    return raw[_HEADER_UINT16S:]


class NanoGPTTokenDataset(IterableDataset, Stateful):
    """Iterable dataset for GPT-2-tokenized FineWeb shards from modded-nanogpt."""

    def __init__(
        self,
        *,
        token_files: list[Path],
        seq_len: int,
        dp_rank: int,
        dp_world_size: int,
        infinite: bool,
        align_to_bos: bool,
        bos_id: int,
    ) -> None:
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        if dp_rank < 0 or dp_rank >= dp_world_size:
            raise ValueError(
                f"dp_rank must be in [0, {dp_world_size}), got {dp_rank}"
            )

        self.token_files = token_files
        self.seq_len = seq_len
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.infinite = infinite
        self.align_to_bos = align_to_bos
        self.bos_id = bos_id

        self._file_idx = 0
        self._cursor = self._initial_cursor()
        self._epoch = 0

    def _initial_cursor(self) -> int:
        if self.align_to_bos:
            return self.dp_rank
        return self.dp_rank * self.seq_len

    def _advance_file(self) -> bool:
        self._file_idx += 1
        self._cursor = self._initial_cursor()
        if self._file_idx < len(self.token_files):
            return True

        if not self.infinite:
            logger.warning("NanoGPT token dataset has run out of data")
            return False

        self._file_idx = 0
        self._epoch += 1
        logger.warning(
            "NanoGPT token dataset is being re-looped (epoch %s)", self._epoch
        )
        return True

    def _iter_contiguous(self, tokens: torch.Tensor):
        max_start = tokens.numel() - self.seq_len - 1
        stride = self.seq_len * self.dp_world_size

        while self._cursor <= max_start:
            start = self._cursor
            self._cursor += stride
            x = tokens[start : start + self.seq_len].to(torch.long)
            y = tokens[start + 1 : start + self.seq_len + 1].to(torch.long)
            positions = torch.arange(self.seq_len, dtype=torch.long)
            yield {"input": x, "positions": positions}, y

    def _iter_bos_aligned(self, tokens: torch.Tensor):
        bos_positions = (tokens == self.bos_id).nonzero(as_tuple=True)[0]
        while self._cursor < bos_positions.numel():
            bos_idx = int(bos_positions[self._cursor].item())
            self._cursor += self.dp_world_size
            if bos_idx + self.seq_len + 1 > tokens.numel():
                continue

            x = tokens[bos_idx : bos_idx + self.seq_len].to(torch.long)
            y = tokens[bos_idx + 1 : bos_idx + self.seq_len + 1].to(torch.long)
            positions = torch.arange(self.seq_len, dtype=torch.long)
            yield {"input": x, "positions": positions}, y

    def __iter__(self):
        while True:
            token_file = self.token_files[self._file_idx]
            logger.info("Loading nanoGPT token shard %s", token_file)
            tokens = load_nanogpt_bin_tokens(token_file)

            if self.align_to_bos:
                yield from self._iter_bos_aligned(tokens)
            else:
                yield from self._iter_contiguous(tokens)

            if not self._advance_file():
                break

    def state_dict(self) -> dict[str, Any]:
        return {
            "file_idx": self._file_idx,
            "cursor": self._cursor,
            "epoch": self._epoch,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._file_idx = state_dict["file_idx"]
        self._cursor = state_dict["cursor"]
        self._epoch = state_dict.get("epoch", 0)


class NanoGPTTokenDataLoader(ParallelAwareDataloader):
    """Dataloader for modded-nanogpt GPT-2-tokenized FineWeb .bin shards."""

    @dataclass(kw_only=True, slots=True)
    class Config(ParallelAwareDataloader.Config):
        dataset: str = "nanogpt_fineweb10b"
        """Dataset identifier for logging/config dumps."""

        dataset_path: str | None = (
            "../modded-nanogpt/data/fineweb10B/fineweb_train_*.bin"
        )
        """Glob, directory, or single .bin file containing nanoGPT token shards."""

        infinite: bool = True
        """Whether to loop over token shards indefinitely."""

        align_to_bos: bool = True
        """Whether every sample should start at GPT-2's BOS/EOS token."""

        bos_id: int = 50256
        """GPT-2 BOS/EOS token id used by the FineWeb binary shards."""

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        seq_len: int,
        local_batch_size: int,
        **kwargs,
    ):
        del tokenizer
        if not config.dataset_path:
            raise ValueError("NanoGPTTokenDataLoader requires dataset_path")

        dataset = NanoGPTTokenDataset(
            token_files=_resolve_token_files(config.dataset_path),
            seq_len=seq_len,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=config.infinite,
            align_to_bos=config.align_to_bos,
            bos_id=config.bos_id,
        )

        dataloader_kwargs = {
            "num_workers": config.num_workers,
            "persistent_workers": config.persistent_workers,
            "pin_memory": config.pin_memory,
            "prefetch_factor": config.prefetch_factor,
            "batch_size": local_batch_size,
        }

        super().__init__(
            dataset,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            **dataloader_kwargs,
        )
