# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from functools import partial

import torch.nn as nn

from torchtitan.components.quantization import QuantizationConverter
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import (
    compute_ffn_hidden_dim,
    Embedding,
    Linear,
    RMSNorm,
    RoPE,
    TransformerBlock,
)
from torchtitan.models.common.config_utils import (
    get_attention_config,
    make_ffn_config,
    make_gqa_config,
)
from torchtitan.models.common.param_init import depth_scaled_std, skip_param_init
from torchtitan.protocols.model_spec import ModelSpec

from .model import Llama3KeelTransformerBlock, Llama3Model, Llama3TransformerBlock
from .parallelize import parallelize_llama
from .state_dict_adapter import Llama3StateDictAdapter

__all__ = [
    "parallelize_llama",
    "Llama3Model",
    "llama3_configs",
]


_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_PAPER_LINEAR_INIT = {
    "weight": partial(nn.init.normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}
_PAPER_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=0.02)}
_EMBEDDING_SKIP_INIT = {"weight": skip_param_init}


def _output_linear_init(dim: int) -> dict[str, Callable]:
    s = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s),
        "bias": nn.init.zeros_,
    }


def _depth_init(layer_id: int) -> dict[str, Callable]:
    return {
        "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "bias": nn.init.zeros_,
    }


def _norm_config(dim: int) -> RMSNorm.Config:
    return RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT)


def _build_llama3_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    hidden_dim: int,
    n_kv_heads: int | None = None,
    fuse_qkv: bool = False,
    attn_backend: str,
    depth_scaled_init: bool = True,
) -> list[TransformerBlock.Config]:
    """Build a list of per-layer TransformerBlock configs with depth-scaled inits."""
    inner_attention, mask_type = get_attention_config(attn_backend)
    layers = []
    for layer_id in range(n_layers):
        base_param_init = _LINEAR_INIT if depth_scaled_init else _PAPER_LINEAR_INIT
        output_param_init = (
            _depth_init(layer_id) if depth_scaled_init else _PAPER_LINEAR_INIT
        )
        layers.append(
            Llama3TransformerBlock.Config(
                attention_norm=_norm_config(dim),
                ffn_norm=_norm_config(dim),
                attention=make_gqa_config(
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    wqkv_param_init=base_param_init,
                    wo_param_init=output_param_init,
                    inner_attention=inner_attention,
                    fuse_qkv=fuse_qkv,
                    mask_type=mask_type,
                    rope_backend="complex",
                ),
                feed_forward=make_ffn_config(
                    dim=dim,
                    hidden_dim=hidden_dim,
                    w1_param_init=base_param_init,
                    w2w3_param_init=output_param_init,
                ),
            )
        )
    return layers


def _build_llama3_keel_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    hidden_dim: int,
    n_kv_heads: int | None = None,
    fuse_qkv: bool = False,
    attn_backend: str,
    residual_scale: float | None = None,
    block_loop_count: int = 1,
) -> list[TransformerBlock.Config]:
    """Build KEEL layers. ``n_layers`` is Transformer blocks, so sub-layers = 2x."""
    inner_attention, mask_type = get_attention_config(attn_backend)
    alpha = float(
        residual_scale
        if residual_scale is not None
        else 2 * n_layers * block_loop_count
    )
    layers = []
    for layer_id in range(n_layers):
        is_first_block = layer_id == 0
        use_runtime_first_block_rule = is_first_block and block_loop_count > 1
        residual_scale_for_block = (
            1.0 if is_first_block and not use_runtime_first_block_rule else alpha
        )
        layers.append(
            Llama3KeelTransformerBlock.Config(
                attention_norm=_norm_config(dim),
                ffn_norm=_norm_config(dim),
                attention_post_norm=(
                    None
                    if is_first_block and not use_runtime_first_block_rule
                    else _norm_config(dim)
                ),
                ffn_post_norm=_norm_config(dim),
                attention_residual_scale=residual_scale_for_block,
                ffn_residual_scale=residual_scale_for_block,
                attention=make_gqa_config(
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    wqkv_param_init=_PAPER_LINEAR_INIT,
                    wo_param_init=_PAPER_LINEAR_INIT,
                    inner_attention=inner_attention,
                    fuse_qkv=fuse_qkv,
                    mask_type=mask_type,
                    rope_backend="complex",
                ),
                feed_forward=make_ffn_config(
                    dim=dim,
                    hidden_dim=hidden_dim,
                    w1_param_init=_PAPER_LINEAR_INIT,
                    w2w3_param_init=_PAPER_LINEAR_INIT,
                ),
            )
        )
    return layers


def _debugmodel(attn_backend: str) -> Llama3Model.Config:
    dim = 256
    n_heads = 16
    n_layers = 6
    return Llama3Model.Config(
        dim=dim,
        vocab_size=2048,
        tok_embeddings=Embedding.Config(
            num_embeddings=2048, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim, out_features=2048, param_init=_output_linear_init(dim)
        ),
        rope=RoPE.Config(
            dim=dim // n_heads,
            max_seq_len=131072,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
        layers=_build_llama3_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            hidden_dim=compute_ffn_hidden_dim(dim, multiple_of=256),
            attn_backend=attn_backend,
        ),
    )


def _keel_debugmodel(attn_backend: str) -> Llama3Model.Config:
    dim = 256
    n_heads = 16
    n_layers = 6
    return Llama3Model.Config(
        dim=dim,
        vocab_size=2048,
        tok_embeddings=Embedding.Config(
            num_embeddings=2048, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=_norm_config(dim),
        lm_head=Linear.Config(
            in_features=dim, out_features=2048, param_init=_output_linear_init(dim)
        ),
        rope=RoPE.Config(
            dim=dim // n_heads,
            max_seq_len=131072,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
        layers=_build_llama3_keel_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            hidden_dim=compute_ffn_hidden_dim(dim, multiple_of=256),
            attn_backend=attn_backend,
        ),
    )


def _debugmodel_fused_qkv(attn_backend: str) -> Llama3Model.Config:
    dim = 256
    n_heads = 16
    n_layers = 6
    return Llama3Model.Config(
        dim=dim,
        vocab_size=2048,
        tok_embeddings=Embedding.Config(
            num_embeddings=2048, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim, out_features=2048, param_init=_output_linear_init(dim)
        ),
        rope=RoPE.Config(
            dim=dim // n_heads,
            max_seq_len=131072,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
        layers=_build_llama3_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            hidden_dim=compute_ffn_hidden_dim(dim, multiple_of=256),
            fuse_qkv=True,
            attn_backend=attn_backend,
        ),
    )


def _nanogpt_smoke_model(
    attn_backend: str,
    *,
    n_layers: int,
    dim: int = 768,
    n_heads: int = 12,
    use_keel: bool = False,
    block_loop_count: int = 1,
) -> Llama3Model.Config:
    # modded-nanogpt uses GPT-2 tokens with 50,257 ids padded to 50,304.
    # Use GPT-2-style tied input/output embeddings. Without tying, the padded
    # 50k vocab adds vocab_size * dim parameters.
    vocab_size = 50304
    build_layers = _build_llama3_keel_layers if use_keel else _build_llama3_layers
    layer_kwargs = {}
    if use_keel:
        layer_kwargs["block_loop_count"] = block_loop_count
    return Llama3Model.Config(
        dim=dim,
        vocab_size=vocab_size,
        enable_weight_tying=True,
        block_loop_count=block_loop_count,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_SKIP_INIT,
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=dim // n_heads,
            max_seq_len=131072,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
        layers=build_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            hidden_dim=compute_ffn_hidden_dim(dim, multiple_of=256),
            attn_backend=attn_backend,
            **layer_kwargs,
        ),
    )


def _nanogpt_smoke(attn_backend: str) -> Llama3Model.Config:
    return _nanogpt_smoke_model(attn_backend, n_layers=12)


def _nanogpt_smoke_24layer(attn_backend: str) -> Llama3Model.Config:
    return _nanogpt_smoke_model(attn_backend, n_layers=24)


def _nanogpt_smoke_48layer(attn_backend: str) -> Llama3Model.Config:
    return _nanogpt_smoke_model(attn_backend, n_layers=48)


def _nanogpt_smoke_384x192(attn_backend: str) -> Llama3Model.Config:
    return _nanogpt_smoke_model(attn_backend, n_layers=192, dim=384)


def _keel_gpt2_smoke(attn_backend: str) -> Llama3Model.Config:
    return _nanogpt_smoke_model(attn_backend, n_layers=12, use_keel=True)


def _keel_gpt2_512x256(attn_backend: str) -> Llama3Model.Config:
    return _nanogpt_smoke_model(
        attn_backend, n_layers=256, dim=512, n_heads=8, use_keel=True
    )


def _keel_gpt2_looped_512x128x2(attn_backend: str) -> Llama3Model.Config:
    return _nanogpt_smoke_model(
        attn_backend,
        n_layers=128,
        dim=512,
        n_heads=8,
        use_keel=True,
        block_loop_count=2,
    )


def _keel_gpt2_looped_512x16x16(attn_backend: str) -> Llama3Model.Config:
    return _nanogpt_smoke_model(
        attn_backend,
        n_layers=16,
        dim=512,
        n_heads=8,
        use_keel=True,
        block_loop_count=16,
    )


def _keel_gpt2_looped_768x32x16(attn_backend: str) -> Llama3Model.Config:
    return _nanogpt_smoke_model(
        attn_backend,
        n_layers=32,
        dim=768,
        n_heads=12,
        use_keel=True,
        block_loop_count=16,
    )


def _keel_gpt2_looped_768x2x32(attn_backend: str) -> Llama3Model.Config:
    return _nanogpt_smoke_model(
        attn_backend,
        n_layers=2,
        dim=768,
        n_heads=12,
        use_keel=True,
        block_loop_count=32,
    )


def _keel_gpt2_looped_768x8x8(attn_backend: str) -> Llama3Model.Config:
    return _nanogpt_smoke_model(
        attn_backend,
        n_layers=8,
        dim=768,
        n_heads=12,
        use_keel=True,
        block_loop_count=8,
    )


def _paper_depth_model(
    attn_backend: str,
    *,
    n_sublayers: int,
    dim: int,
    use_keel: bool,
) -> Llama3Model.Config:
    assert n_sublayers % 2 == 0, (
        "KEEL/Pre-LN depth counts Attention and FFN sub-layers"
    )
    n_blocks = n_sublayers // 2
    n_heads = 16
    n_kv_heads = 8
    vocab_size = 128256
    hidden_dim = 3 * dim
    build_layers = _build_llama3_keel_layers if use_keel else _build_llama3_layers
    layer_kwargs = {}
    if not use_keel:
        layer_kwargs["depth_scaled_init"] = False
    return Llama3Model.Config(
        dim=dim,
        vocab_size=vocab_size,
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_PAPER_EMBEDDING_INIT,
        ),
        norm=_norm_config(dim),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_PAPER_LINEAR_INIT,
        ),
        rope=RoPE.Config(
            dim=dim // n_heads,
            max_seq_len=4096,
            theta=10000,
            backend="complex",
            scaling=None,
        ),
        layers=build_layers(
            n_layers=n_blocks,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            hidden_dim=hidden_dim,
            attn_backend=attn_backend,
            **layer_kwargs,
        ),
    )


def _keel_64x1024(attn_backend: str) -> Llama3Model.Config:
    return _paper_depth_model(
        attn_backend, n_sublayers=64, dim=1024, use_keel=True
    )


def _keel_256x1024(attn_backend: str) -> Llama3Model.Config:
    return _paper_depth_model(
        attn_backend, n_sublayers=256, dim=1024, use_keel=True
    )


def _keel_512x1024(attn_backend: str) -> Llama3Model.Config:
    return _paper_depth_model(
        attn_backend, n_sublayers=512, dim=1024, use_keel=True
    )


def _keel_1024x1024(attn_backend: str) -> Llama3Model.Config:
    return _paper_depth_model(
        attn_backend, n_sublayers=1024, dim=1024, use_keel=True
    )


def _preln_64x1024(attn_backend: str) -> Llama3Model.Config:
    return _paper_depth_model(
        attn_backend, n_sublayers=64, dim=1024, use_keel=False
    )


def _preln_256x1024(attn_backend: str) -> Llama3Model.Config:
    return _paper_depth_model(
        attn_backend, n_sublayers=256, dim=1024, use_keel=False
    )


def _preln_512x1024(attn_backend: str) -> Llama3Model.Config:
    return _paper_depth_model(
        attn_backend, n_sublayers=512, dim=1024, use_keel=False
    )


def _preln_1024x1024(attn_backend: str) -> Llama3Model.Config:
    return _paper_depth_model(
        attn_backend, n_sublayers=1024, dim=1024, use_keel=False
    )


def _1b(attn_backend: str) -> Llama3Model.Config:
    dim = 2048
    n_heads = 32
    n_kv_heads = 8
    n_layers = 16
    vocab_size = 128256
    return Llama3Model.Config(
        dim=dim,
        vocab_size=vocab_size,
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_SKIP_INIT,
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=dim // n_heads,
            max_seq_len=131072,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
        layers=_build_llama3_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            hidden_dim=compute_ffn_hidden_dim(
                dim, multiple_of=1024, ffn_dim_multiplier=1.5
            ),
            attn_backend=attn_backend,
        ),
    )


def _3b(attn_backend: str) -> Llama3Model.Config:
    dim = 3072
    n_heads = 24
    n_kv_heads = 8
    n_layers = 28
    vocab_size = 128256
    return Llama3Model.Config(
        dim=dim,
        vocab_size=vocab_size,
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_SKIP_INIT,
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=dim // n_heads,
            max_seq_len=131072,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
        layers=_build_llama3_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            hidden_dim=compute_ffn_hidden_dim(
                dim, multiple_of=1024, ffn_dim_multiplier=1.0
            ),
            attn_backend=attn_backend,
        ),
    )


def _8b(attn_backend: str) -> Llama3Model.Config:
    dim = 4096
    n_heads = 32
    n_kv_heads = 8
    n_layers = 32
    vocab_size = 128256
    return Llama3Model.Config(
        dim=dim,
        vocab_size=vocab_size,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=dim // n_heads,
            max_seq_len=131072,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
        layers=_build_llama3_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            hidden_dim=compute_ffn_hidden_dim(
                dim, multiple_of=1024, ffn_dim_multiplier=1.3
            ),
            attn_backend=attn_backend,
        ),
    )


def _70b(attn_backend: str) -> Llama3Model.Config:
    dim = 8192
    n_heads = 64
    n_kv_heads = 8
    n_layers = 80
    vocab_size = 128256
    return Llama3Model.Config(
        dim=dim,
        vocab_size=vocab_size,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=dim // n_heads,
            max_seq_len=131072,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
        layers=_build_llama3_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            hidden_dim=compute_ffn_hidden_dim(
                dim, multiple_of=4096, ffn_dim_multiplier=1.3
            ),
            attn_backend=attn_backend,
        ),
    )


def _405b(attn_backend: str) -> Llama3Model.Config:
    dim = 16384
    n_heads = 128
    n_kv_heads = 8
    n_layers = 126
    vocab_size = 128256
    return Llama3Model.Config(
        dim=dim,
        vocab_size=vocab_size,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=dim // n_heads,
            max_seq_len=131072,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
        layers=_build_llama3_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            hidden_dim=compute_ffn_hidden_dim(
                dim, multiple_of=4096, ffn_dim_multiplier=1.2
            ),
            attn_backend=attn_backend,
        ),
    )


llama3_configs = {
    "debugmodel": _debugmodel,
    "keel_debugmodel": _keel_debugmodel,
    "debugmodel_fused_qkv": _debugmodel_fused_qkv,
    "nanogpt_smoke": _nanogpt_smoke,
    "nanogpt_smoke_24layer": _nanogpt_smoke_24layer,
    "nanogpt_smoke_48layer": _nanogpt_smoke_48layer,
    "nanogpt_smoke_384x192": _nanogpt_smoke_384x192,
    "keel_gpt2_smoke": _keel_gpt2_smoke,
    "keel_gpt2_512x256": _keel_gpt2_512x256,
    "keel_gpt2_looped_512x128x2": _keel_gpt2_looped_512x128x2,
    "keel_gpt2_looped_512x16x16": _keel_gpt2_looped_512x16x16,
    "keel_gpt2_looped_768x32x16": _keel_gpt2_looped_768x32x16,
    "keel_gpt2_looped_768x2x32": _keel_gpt2_looped_768x2x32,
    "keel_gpt2_looped_768x8x8": _keel_gpt2_looped_768x8x8,
    "keel_64x1024": _keel_64x1024,
    "keel_256x1024": _keel_256x1024,
    "keel_512x1024": _keel_512x1024,
    "keel_1024x1024": _keel_1024x1024,
    "preln_64x1024": _preln_64x1024,
    "preln_256x1024": _preln_256x1024,
    "preln_512x1024": _preln_512x1024,
    "preln_1024x1024": _preln_1024x1024,
    "1B": _1b,
    "3B": _3b,
    "8B": _8b,
    "70B": _70b,
    "405B": _405b,
}


def model_registry(
    flavor: str,
    attn_backend: str = "sdpa",
    quantization: list[QuantizationConverter.Config] | None = None,
) -> ModelSpec:
    config = llama3_configs[flavor](attn_backend=attn_backend)
    if quantization is not None:
        for q in quantization:
            q.build().convert(config)
    state_dict_adapter = (
        None if flavor.startswith(("keel_", "preln_")) else Llama3StateDictAdapter
    )
    return ModelSpec(
        name="llama3",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=None,
        state_dict_adapter=state_dict_adapter,
    )
