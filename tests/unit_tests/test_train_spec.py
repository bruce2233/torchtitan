# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from functools import partial

import torch
import torch.nn as nn
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.models.common.linear import Linear
from torchtitan.models.llama3 import model_registry, parallelize_llama
from torchtitan.models.llama3.model import (
    Llama3KeelTransformerBlock,
    Llama3Model,
)
from torchtitan.protocols import BaseModel
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.protocols.module import ModuleDict


class FakeModel(BaseModel):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        hidden: int = 8

        def update_from_config(self, *, trainer_config, **kwargs):
            pass

        def get_nparams_and_flops(self, model, seq_len):
            return 0, 0

    def __init__(self, config: Config):
        super().__init__()
        linear_cfg = Linear.Config(
            in_features=config.hidden, out_features=config.hidden
        )
        self.linear = linear_cfg.build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def _init_self_parameters(self) -> None:
        nn.init.trunc_normal_(self.linear.weight, std=0.02)


def fake_post_optimizer_build_fn(
    optimizers: OptimizersContainer,
    model_parts: list[nn.Module],
    parallel_dims: ParallelDims,
    optimizer_hook=None,
) -> None:
    if optimizer_hook is not None:
        optimizers.register_step_post_hook(
            partial(optimizer_hook, model_parts=model_parts)
        )


class TestModelSpec:
    def test_model_registry(self):
        spec = model_registry("debugmodel")
        assert isinstance(spec, ModelSpec)
        assert spec.name == "llama3"
        assert spec.flavor == "debugmodel"
        assert spec.model is not None
        assert spec.parallelize_fn == parallelize_llama

    def test_keel_model_registry(self):
        spec = model_registry("keel_debugmodel")
        assert isinstance(spec, ModelSpec)
        assert spec.flavor == "keel_debugmodel"
        assert len(spec.model.layers) == 6

        first_layer = spec.model.layers[0]
        second_layer = spec.model.layers[1]
        assert isinstance(first_layer, Llama3KeelTransformerBlock.Config)
        assert first_layer.attention_post_norm is None
        assert first_layer.attention_residual_scale == 1.0
        assert first_layer.ffn_residual_scale == 1.0
        assert second_layer.attention_post_norm is not None
        assert second_layer.attention_residual_scale == 12.0
        assert second_layer.ffn_residual_scale == 12.0

    def test_keel_paper_depth_config(self):
        spec = model_registry("keel_512x1024")
        assert len(spec.model.layers) == 256
        assert spec.model.dim == 1024
        assert spec.model.rope.theta == 10000

        first_layer = spec.model.layers[0]
        last_layer = spec.model.layers[-1]
        assert first_layer.feed_forward.w1.out_features == 3072
        assert first_layer.attention_post_norm is None
        assert last_layer.attention_residual_scale == 512.0
        assert last_layer.ffn_residual_scale == 512.0

    def test_looped_keel_model_registry(self):
        spec = model_registry("keel_gpt2_looped_512x128x2")
        assert spec.model.block_loop_count == 2
        assert len(spec.model.layers) == 128
        assert spec.model.dim == 512

        first_layer = spec.model.layers[0]
        last_layer = spec.model.layers[-1]
        assert isinstance(first_layer, Llama3KeelTransformerBlock.Config)
        assert first_layer.attention_post_norm is not None
        assert first_layer.attention_residual_scale == 512.0
        assert first_layer.ffn_residual_scale == 512.0
        assert last_layer.attention_residual_scale == 512.0
        assert last_layer.ffn_residual_scale == 512.0

    def test_looped_keel_16x16_model_registry(self):
        spec = model_registry("keel_gpt2_looped_512x16x16")
        assert spec.model.block_loop_count == 16
        assert len(spec.model.layers) == 16
        assert spec.model.dim == 512

        first_layer = spec.model.layers[0]
        last_layer = spec.model.layers[-1]
        assert isinstance(first_layer, Llama3KeelTransformerBlock.Config)
        assert first_layer.attention_post_norm is not None
        assert first_layer.attention_residual_scale == 512.0
        assert first_layer.ffn_residual_scale == 512.0
        assert last_layer.attention_residual_scale == 512.0
        assert last_layer.ffn_residual_scale == 512.0

    def test_looped_keel_768x32x16_model_registry(self):
        spec = model_registry("keel_gpt2_looped_768x32x16")
        assert spec.model.block_loop_count == 16
        assert len(spec.model.layers) == 32
        assert spec.model.dim == 768

        first_layer = spec.model.layers[0]
        last_layer = spec.model.layers[-1]
        assert isinstance(first_layer, Llama3KeelTransformerBlock.Config)
        assert first_layer.attention_post_norm is not None
        assert first_layer.attention_residual_scale == 1024.0
        assert first_layer.ffn_residual_scale == 1024.0
        assert last_layer.attention_residual_scale == 1024.0
        assert last_layer.ffn_residual_scale == 1024.0

    def test_looped_keel_first_logical_block_override(self):
        class RecordingKeelBlock(Llama3KeelTransformerBlock):
            def __init__(self):
                nn.Module.__init__(self)
                self.calls = []

            def forward(
                self,
                x,
                freqs_cis,
                attention_masks,
                positions=None,
                *,
                is_first_logical_block=None,
            ):
                self.calls.append(is_first_logical_block)
                return x

        model = model_registry("keel_debugmodel").model.build()
        assert isinstance(model, Llama3Model)
        first_block = RecordingKeelBlock()
        second_block = RecordingKeelBlock()
        model.layers = ModuleDict({"0": first_block, "1": second_block})
        model.block_loop_count = 2
        model._skip_lm_head = True

        model(torch.zeros((1, 4), dtype=torch.long))

        assert first_block.calls == [True, False]
        assert second_block.calls == [False, False]

    def test_model_spec_creation(self):
        fake_config = FakeModel.Config()
        spec = ModelSpec(
            name="fake",
            flavor="test",
            model=fake_config,
            parallelize_fn=parallelize_llama,
            pipelining_fn=None,
            post_optimizer_build_fn=None,
            state_dict_adapter=None,
        )
        assert spec.name == "fake"
        assert spec.flavor == "test"
        assert spec.model == fake_config

    def test_optim_hook(self):
        fake_config = FakeModel.Config()

        spec = ModelSpec(
            name="fake",
            flavor="test",
            model=fake_config,
            parallelize_fn=parallelize_llama,
            pipelining_fn=None,
            post_optimizer_build_fn=fake_post_optimizer_build_fn,
            state_dict_adapter=None,
        )

        model = FakeModel.Config().build()
        model_parts = [model]

        # Demonstrate how to register a optimizer hook for all model specs
        hook_called = False

        def my_hook(
            optimizer: torch.optim.Optimizer,
            args,
            kwargs,
            model_parts: list[nn.Module],
        ) -> None:
            nonlocal hook_called
            hook_called = True

        # Build optimizers directly and apply post-build hook
        optimizers = OptimizersContainer.Config(
            name="Adam",
            lr=0.1,
            beta1=0.9,
            beta2=0.95,
            weight_decay=0.1,
            implementation="fused",
        ).build(model_parts=model_parts)
        spec.post_optimizer_build_fn(optimizers, model_parts, None, my_hook)

        assert optimizers.optimizers[0].__class__.__name__ == "Adam"
        batch = torch.randn(8, 8)
        model(batch).sum().backward()
        assert not hook_called
        optimizers.step()
        assert hook_called
