# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.components.loss import (
    ContrastiveNTPLoss,
    _build_batch_local_candidates,
    _flatten_valid_contrastive_targets,
    _token_to_context_loss,
    contrastive_ntp_loss_with_metrics,
)


class _TinyTransformer(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.proj(self.embedding(input_ids))


class TestContrastiveNTPLoss(unittest.TestCase):
    def test_shape(self):
        torch.manual_seed(42)
        vocab_size = 16
        dim = 8
        input_ids = torch.tensor(
            [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
            ],
            dtype=torch.long,
        )
        hidden = torch.randn(2, 5, dim)
        targets = input_ids[:, 1:]
        queries, targets_flat = _flatten_valid_contrastive_targets(
            hidden[:, :-1, :],
            targets,
            ignore_index=None,
        )
        unique_ids, labels = _build_batch_local_candidates(targets_flat)
        token_embedding = nn.Embedding(vocab_size, dim)
        logits = queries @ token_embedding(unique_ids).T

        self.assertEqual(queries.shape, torch.Size([8, dim]))
        self.assertEqual(targets_flat.shape, torch.Size([8]))
        self.assertEqual(labels.shape, torch.Size([8]))
        self.assertEqual(logits.shape, torch.Size([8, unique_ids.numel()]))

    def test_duplicate_tokens_share_candidate_label(self):
        input_ids = torch.tensor([[1, 2, 3, 2, 4]], dtype=torch.long)
        hidden = torch.randn(1, 5, 8)
        queries, targets = _flatten_valid_contrastive_targets(
            hidden[:, :-1, :],
            input_ids[:, 1:],
            ignore_index=None,
        )
        del queries

        unique_ids, labels = _build_batch_local_candidates(targets)

        self.assertEqual(set(unique_ids.tolist()), {2, 3, 4})
        self.assertEqual(unique_ids.numel(), 3)
        self.assertEqual(labels[0].item(), labels[2].item())

    def test_padding_targets_are_removed(self):
        input_ids = torch.tensor([[1, 2, 3, 0, 0]], dtype=torch.long)
        hidden = torch.randn(1, 5, 8)
        _, targets = _flatten_valid_contrastive_targets(
            hidden[:, :-1, :],
            input_ids[:, 1:],
            pad_id=0,
            ignore_index=None,
        )
        unique_ids, _ = _build_batch_local_candidates(targets)

        self.assertEqual(targets.tolist(), [2, 3])
        self.assertNotIn(0, unique_ids.tolist())

    def test_backward_updates_transformer_and_token_embedding(self):
        torch.manual_seed(123)
        vocab_size = 16
        dim = 8
        transformer = _TinyTransformer(vocab_size, dim)
        token_embedding = nn.Embedding(vocab_size, dim)
        input_ids = torch.tensor([[1, 2, 3, 2, 4]], dtype=torch.long)

        loss, metrics = contrastive_ntp_loss_with_metrics(
            input_ids=input_ids,
            transformer=transformer,
            token_embedding=token_embedding,
            tau=0.07,
            normalize=True,
        )
        loss.backward()

        self.assertEqual(metrics["num_queries"].item(), 4)
        self.assertEqual(metrics["num_candidates"].item(), 3)
        self.assertTrue(
            any(p.grad is not None for p in transformer.parameters()),
            "transformer parameters should receive gradients",
        )
        self.assertIsNotNone(token_embedding.weight.grad)

    def test_token_to_context_loss_is_multi_positive(self):
        logits = torch.tensor(
            [
                [3.0, 0.0],
                [0.0, 3.0],
                [2.0, 0.0],
            ]
        )
        labels = torch.tensor([0, 1, 0], dtype=torch.long)

        actual = _token_to_context_loss(logits, labels)

        log_probs = F.log_softmax(logits.T, dim=-1)
        expected_for_token_0 = torch.logsumexp(
            torch.stack([log_probs[0, 0], log_probs[0, 2]]),
            dim=0,
        )
        expected_for_token_1 = log_probs[1, 1]
        expected = -torch.stack(
            [expected_for_token_0, expected_for_token_1]
        ).mean()

        self.assertTrue(torch.allclose(actual, expected))

    def test_symmetric_loss_combines_c2t_and_t2c(self):
        torch.manual_seed(456)
        vocab_size = 16
        dim = 8
        transformer = _TinyTransformer(vocab_size, dim)
        token_embedding = nn.Embedding(vocab_size, dim)
        input_ids = torch.tensor([[1, 2, 3, 2, 4]], dtype=torch.long)

        loss, metrics = contrastive_ntp_loss_with_metrics(
            input_ids=input_ids,
            transformer=transformer,
            token_embedding=token_embedding,
            tau=0.07,
            normalize=True,
            lambda_t2c=1.0,
        )
        expected = metrics["loss_c2t"] + metrics["loss_t2c"]

        self.assertGreater(metrics["loss_t2c"].item(), 0.0)
        self.assertTrue(torch.allclose(loss.detach(), expected, atol=1e-6))

        c2t_only_loss, c2t_only_metrics = contrastive_ntp_loss_with_metrics(
            input_ids=input_ids,
            transformer=transformer,
            token_embedding=token_embedding,
            tau=0.07,
            normalize=True,
            lambda_t2c=0.0,
        )
        self.assertTrue(
            torch.allclose(
                c2t_only_loss.detach(),
                c2t_only_metrics["loss_c2t"],
                atol=1e-6,
            )
        )

    def test_loss_accepts_float_global_valid_tokens(self):
        torch.manual_seed(789)
        vocab_size = 16
        dim = 8
        token_embedding = nn.Embedding(vocab_size, dim)
        hidden = torch.randn(1, 4, dim, requires_grad=True)
        labels = torch.tensor([[2, 3, 2, 4]], dtype=torch.long)
        loss_fn = ContrastiveNTPLoss(
            ContrastiveNTPLoss.Config(tau=0.07, normalize=True)
        )
        loss_fn.set_token_embedding(token_embedding)

        loss = loss_fn(hidden, labels, global_valid_tokens=4.0)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(hidden.grad)
        self.assertIsNotNone(token_embedding.weight.grad)


if __name__ == "__main__":
    unittest.main()
