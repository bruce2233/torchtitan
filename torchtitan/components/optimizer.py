# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import re
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, overload, TypeVar

import torch
import torch.distributed.tensor
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.tensor import Replicate
from torch.optim import Optimizer
from torchtitan.config import Configurable
from torchtitan.distributed import ParallelDims
from torchtitan.tools.logging import logger

__all__ = [
    "OptimizersContainer",
    "OptimizersInBackwardContainer",
    "ParamGroupConfig",
    "register_moe_load_balancing_hook",
]


@dataclass(kw_only=True, slots=True)
class ParamGroupConfig:
    """Configuration for a parameter group with custom optimizer settings.

    Parameters matching the regex pattern will use lr and weight_decay values
    derived by multiplying the global optimizer values with the specified multipliers.
    """

    pattern: str
    """Regex pattern matched against parameter fully qualified names (FQNs).
    E.g. '.*bias$', '.*norm.*', '.*\\.embed_tokens\\..*'"""

    lr_multiplier: float = 1.0
    """Multiplied with the global optimizer lr to get this group's lr."""

    weight_decay_multiplier: float = 1.0
    """Multiplied with the global optimizer weight_decay to get this group's weight_decay."""

    beta1: float | None = None
    beta2: float | None = None
    """Override betas for this group. None means use the global optimizer betas.
    Each can be overridden independently."""


T = TypeVar("T", bound=Optimizer)


# TODO: Right now this class is biased towards AdamW. We should refactor to
# support mixed optimizers, including Muon.
class OptimizersContainer(Optimizer, Stateful, Configurable, Generic[T]):
    """A container for multiple optimizers.

    This class is used to wrap multiple optimizers into a single object that can be
    used to reduce the complexity of the training loop. This mimics the behavior of
    ``torch.optim.Optimizer``. This class currently only supports ``Adam`` and ``AdamW``.

    **Note**
    Users who want to customize the optimizer behavior can inherit from this class and
    extend the functionality as needed. The following methods must follow the same signature
    as ``torch.optim.Optimizer`` class: ``step()``, ``zero_grad()``, ``state_dict()``,
    ``load_state_dict()``.

    **Limitations**
    This class assumes that all the optimizers are the same type and have the same
    configurations. With this assumption, TorchTitan can support lr scheduler resharding
    (e.g., loading a checkpoint with a different number of GPUs and/or different
    parallelization strategy). Note that ``get_optimizer_state_dict`` already enables the
    resharding for the optimizer state but not for the lr scheduler state, hence the limitation.

    Args:
        model_parts (List[nn.Module]): List of model parts to be optimized.
        optimizer_kwargs (Dict[str, Any]): Keyword arguments for the optimizers.
        name (str): Name of the optimizers.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        name: str = "AdamW"
        """Optimizer to use"""

        lr: float = 8e-4
        """Learning rate to use"""

        beta1: float = 0.9
        beta2: float = 0.95
        """Exponential moving average hyperparameters to use"""

        eps: float = 1e-8
        """Epsilon value to use"""

        weight_decay: float = 0.1
        """Weight decay to use"""

        muon_momentum: float = 0.95
        """Momentum factor for Muon."""

        muon_nesterov: bool = True
        """Whether to use Nesterov momentum for Muon."""

        muon_ns_steps: int = 5
        """Number of Newton-Schulz iterations for Muon."""

        muon_ns_coefficients: tuple[float, float, float] = (3.4445, -4.775, 2.0315)
        """Newton-Schulz polynomial coefficients for Muon."""

        muon_eps: float = 1e-7
        """Numerical epsilon for Muon."""

        muon_adjust_lr_fn: Literal["original", "match_rms_adamw"] | None = (
            "match_rms_adamw"
        )
        """Muon learning-rate adjustment. 'match_rms_adamw' reuses AdamW-tuned LR."""

        muon_param_pattern: str | None = None
        """Optional regex selecting Muon parameters. Only matching 2D params use Muon."""

        muon_exclude_param_pattern: str = (
            r"(^|\.)(tok_embeddings|embed_tokens|embedding|embeddings|"
            r"lm_head|output)(\.|$)"
        )
        """Regex excluding 2D parameters from Muon when muon_param_pattern is unset."""

        implementation: Literal[
            "for-loop", "foreach", "fused", "fused_opt_states_bf16"
        ] = "fused"
        """
        Specify which optimizer implementation to use:
        - 'fused': Use fused implementation (CUDA only) for best performance.
        - 'foreach': Use some horizontal fusion of tensors for better performance.
        - 'for-loop': Use the default implementation for the optimizer (slowest).
        - 'fused_opt_states_bf16': Like 'fused', but initialize Adam/AdamW
          momentum and variance in bfloat16 via a step pre-hook so the fused
          CUDA kernel uses its mixed-precision path (fp32 params + bf16 states).
          Only supported for Adam/AdamW optimizer states, including the AdamW
          fallback groups when name='Muon'. Not supported with
          OptimizersInBackwardContainer. See docs/bf16_optimizer_states.md.
        - more info: https://pytorch.org/docs/stable/optim.html
        """

        param_groups: list[ParamGroupConfig] = field(default_factory=list)
        """Optional per-parameter-group overrides. Each entry specifies a regex
        pattern matching parameter FQNs and multipliers for lr and weight_decay.
        Parameters not matching any pattern use the global defaults.
        Patterns are checked in order; first match wins."""

        def __post_init__(self):
            if self.implementation == "fused_opt_states_bf16":
                if self.name not in ("Adam", "AdamW", "Muon"):
                    raise ValueError(
                        "implementation='fused_opt_states_bf16' is only supported "
                        f"for Adam/AdamW states, got optimizer '{self.name}'"
                    )

    optimizers: list[T]
    optimizers_by_model_part: list[list[T]]
    model_parts: list[nn.Module]

    @staticmethod
    def _resolve_optimizer_cls(name: str) -> type:
        optimizer_classes = {
            "Adam": torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
            "Muon": getattr(torch.optim, "Muon", None),
        }
        if name not in optimizer_classes:
            raise NotImplementedError(f"Optimizer {name} not added.")
        if optimizer_classes[name] is None:
            raise NotImplementedError(
                "Optimizer Muon is not available in this PyTorch build."
            )
        return optimizer_classes[name]

    @staticmethod
    def _build_optimizer_kwargs(config: Config) -> dict[str, Any]:
        return OptimizersContainer._build_adam_optimizer_kwargs(config)

    @staticmethod
    def _build_adam_optimizer_kwargs(config: Config) -> dict[str, Any]:
        assert config.implementation in [
            "fused",
            "foreach",
            "for-loop",
            "fused_opt_states_bf16",
        ]
        fused = config.implementation in ("fused", "fused_opt_states_bf16")
        return {
            "lr": config.lr,
            "betas": (config.beta1, config.beta2),
            "eps": config.eps,
            "weight_decay": config.weight_decay,
            "fused": fused,
            "foreach": config.implementation == "foreach",
        }

    @staticmethod
    def _build_muon_optimizer_kwargs(config: Config) -> dict[str, Any]:
        return {
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "momentum": config.muon_momentum,
            "nesterov": config.muon_nesterov,
            "ns_coefficients": config.muon_ns_coefficients,
            "eps": config.muon_eps,
            "ns_steps": config.muon_ns_steps,
            "adjust_lr_fn": config.muon_adjust_lr_fn,
        }

    @staticmethod
    def _build_param_groups_from_named_parameters(
        named_parameters: list[tuple[str, nn.Parameter]],
        config: Config,
        default_kwargs: dict[str, Any],
        *,
        warn_on_zero_matches: bool = True,
    ) -> list[dict[str, Any]]:
        """Build PyTorch param groups from model parameters and config.

        Each parameter is assigned to the first matching ParamGroupConfig pattern,
        or to the default group if no pattern matches. Returns a list of dicts
        with "params" key and optimizer kwargs, suitable for passing to an optimizer.
        """
        named_parameters = [
            (name, param) for name, param in named_parameters if param.requires_grad
        ]

        if not config.param_groups:
            params = [p for _, p in named_parameters]
            return [{"params": params, **default_kwargs}] if params else []

        compiled_patterns = [re.compile(pg.pattern) for pg in config.param_groups]

        # group_index -> list of params; None means default group
        grouped_params: dict[int | None, list[nn.Parameter]] = defaultdict(list)

        for name, param in named_parameters:
            matched_index = None
            for i, pat in enumerate(compiled_patterns):
                if pat.search(name):
                    matched_index = i
                    break
            grouped_params[matched_index].append(param)

        # Warn for patterns that matched nothing
        for i, pg in enumerate(config.param_groups):
            if warn_on_zero_matches and i not in grouped_params:
                logger.warning(
                    f"Optimizer param_groups pattern '{pg.pattern}' "
                    f"matched no parameters"
                )

        result = []
        # Default group first (unmatched params)
        if None in grouped_params:
            result.append({"params": grouped_params[None], **default_kwargs})

        # Then each matched group in pattern order
        for i, pg in enumerate(config.param_groups):
            if i not in grouped_params:
                continue
            group_kwargs = {**default_kwargs}
            group_kwargs["lr"] = default_kwargs["lr"] * pg.lr_multiplier
            group_kwargs["weight_decay"] = (
                default_kwargs["weight_decay"] * pg.weight_decay_multiplier
            )
            if (pg.beta1 is not None or pg.beta2 is not None) and (
                "betas" in default_kwargs
            ):
                default_beta1, default_beta2 = default_kwargs["betas"]
                group_kwargs["betas"] = (
                    pg.beta1 if pg.beta1 is not None else default_beta1,
                    pg.beta2 if pg.beta2 is not None else default_beta2,
                )
            result.append({"params": grouped_params[i], **group_kwargs})

        return result

    @staticmethod
    def _build_param_groups(
        model: nn.Module,
        config: Config,
        default_kwargs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return OptimizersContainer._build_param_groups_from_named_parameters(
            list(model.named_parameters()),
            config,
            default_kwargs,
        )

    @staticmethod
    def _split_muon_named_parameters(
        model: nn.Module, config: Config
    ) -> tuple[list[tuple[str, nn.Parameter]], list[tuple[str, nn.Parameter]]]:
        """Split trainable params into Muon and AdamW fallback groups.

        Muon is only valid for 2D hidden-layer parameters. By default we send 2D
        tensors to Muon while keeping embeddings, lm heads/output layers, bias,
        and normalization parameters on AdamW. Users can narrow Muon selection
        further with ``muon_param_pattern``.
        """
        muon_param_pattern = (
            re.compile(config.muon_param_pattern)
            if config.muon_param_pattern is not None
            else None
        )
        muon_exclude_param_pattern = (
            re.compile(config.muon_exclude_param_pattern, re.IGNORECASE)
            if config.muon_exclude_param_pattern
            else None
        )

        muon_named_parameters = []
        adam_named_parameters = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            use_muon = param.ndim == 2
            if muon_param_pattern is not None:
                use_muon = use_muon and bool(muon_param_pattern.search(name))
            elif muon_exclude_param_pattern is not None:
                use_muon = use_muon and not bool(
                    muon_exclude_param_pattern.search(name)
                )

            if use_muon:
                muon_named_parameters.append((name, param))
            else:
                adam_named_parameters.append((name, param))

        return muon_named_parameters, adam_named_parameters

    def __init__(self, config: Config, *, model_parts: list[nn.Module]) -> None:
        all_params = []
        self.optimizers = []
        self.optimizers_by_model_part = []
        self.model_parts = model_parts

        if config.name == "Muon":
            self._resolve_optimizer_cls("Muon")
            adam_optimizer_kwargs = self._build_adam_optimizer_kwargs(config)
            muon_optimizer_kwargs = self._build_muon_optimizer_kwargs(config)
            bf16_state_optimizers = []
            has_muon_params = False

            for model in self.model_parts:
                part_optimizers = []
                (
                    muon_named_parameters,
                    adam_named_parameters,
                ) = self._split_muon_named_parameters(model, config)

                muon_param_groups = self._build_param_groups_from_named_parameters(
                    muon_named_parameters,
                    config,
                    muon_optimizer_kwargs,
                    warn_on_zero_matches=False,
                )
                if muon_param_groups:
                    has_muon_params = True
                    muon_optimizer = torch.optim.Muon(muon_param_groups)
                    part_optimizers.append(muon_optimizer)
                    for group in muon_param_groups:
                        all_params.extend(group["params"])

                adam_param_groups = self._build_param_groups_from_named_parameters(
                    adam_named_parameters,
                    config,
                    adam_optimizer_kwargs,
                    warn_on_zero_matches=False,
                )
                if adam_param_groups:
                    adam_optimizer = torch.optim.AdamW(adam_param_groups)
                    part_optimizers.append(adam_optimizer)
                    bf16_state_optimizers.append(adam_optimizer)
                    for group in adam_param_groups:
                        all_params.extend(group["params"])

                self.optimizers.extend(part_optimizers)
                self.optimizers_by_model_part.append(part_optimizers)

            if not has_muon_params:
                raise ValueError(
                    "optimizer.name='Muon' selected no 2D hidden-layer parameters. "
                    "Check muon_param_pattern and muon_exclude_param_pattern."
                )
            if config.implementation == "fused_opt_states_bf16":
                self._register_bf16_optimizer_state_hook(bf16_state_optimizers)
            optimizer_kwargs = adam_optimizer_kwargs
            expected_num_optimizers = sum(
                len(part_optimizers)
                for part_optimizers in self.optimizers_by_model_part
            )
        else:
            optimizer_cls = self._resolve_optimizer_cls(config.name)
            optimizer_kwargs = self._build_optimizer_kwargs(config)
            for model in self.model_parts:
                param_groups = self._build_param_groups(
                    model, config, optimizer_kwargs
                )
                optimizer = optimizer_cls(param_groups)
                self.optimizers.append(optimizer)
                self.optimizers_by_model_part.append([optimizer])
                for group in param_groups:
                    all_params.extend(group["params"])
            expected_num_optimizers = len(self.model_parts)

        if config.implementation == "fused_opt_states_bf16":
            if config.name != "Muon":
                self._register_bf16_optimizer_state_hook()
        self._validate_length(expected_num_optimizers)
        self._post_init(all_params, optimizer_kwargs)

    def __iter__(self) -> Iterator[T]:
        return iter(self.optimizers)

    def __len__(self) -> int:
        return len(self.optimizers)

    @overload
    def step(self, closure: None = None) -> None:
        ...

    @overload
    def step(self, closure: Callable[[], float]) -> float:
        ...

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        assert closure is None, "OptimizersContainer does not support closures"
        for optimizer in self.optimizers:
            optimizer.step()
        return None

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        func = functools.partial(
            get_optimizer_state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        return {
            k: v
            for sd in map(func, self.model_parts, self.optimizers_by_model_part)
            for k, v in sd.items()
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        func = functools.partial(
            set_optimizer_state_dict,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        list(map(func, self.model_parts, self.optimizers_by_model_part))

    def _validate_length(self, expected_length: int) -> None:
        assert expected_length == len(self.optimizers), (
            "Must pass one optimizer per model part or per param if "
            "using OptimizersInBackwardContainer."
        )

    def _post_init(
        self, all_params: list[nn.Parameter], optimizer_kwargs: dict[str, Any]
    ) -> None:
        # We need to call Optimizer.__init__() to initialize some necessary optimizer
        # functionality such as hooks.
        Optimizer.__init__(self, all_params, optimizer_kwargs)

    def _register_bf16_optimizer_state_hook(
        self, optimizers: list[Optimizer] | None = None
    ) -> None:
        """Register a step pre-hook to create Adam optimizer states in bfloat16.

        The hook pre-populates optimizer state before Adam's lazy initialization
        runs, so that ``_init_group`` finds non-empty state and skips its own
        fp32 allocation. The fused CUDA kernel then sees the dtype mismatch
        between fp32 params and bf16 states, dispatching to the mixed-precision
        kernel (``FusedAdamMathFunctorMP``).
        """

        def _bf16_state_init_hook(
            optimizer: Optimizer, args: tuple, kwargs: dict
        ) -> None:
            for group in optimizer.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = optimizer.state[p]
                    if len(state) == 0:
                        state["step"] = (
                            torch.zeros((), dtype=torch.float32, device=p.device)
                            if group.get("capturable") or group.get("fused")
                            else torch.tensor(0.0, dtype=torch.float32)
                        )
                        state["exp_avg"] = torch.zeros_like(
                            p, dtype=torch.bfloat16, memory_format=torch.preserve_format
                        )
                        state["exp_avg_sq"] = torch.zeros_like(
                            p, dtype=torch.bfloat16, memory_format=torch.preserve_format
                        )
                        if group.get("amsgrad"):
                            state["max_exp_avg_sq"] = torch.zeros_like(
                                p,
                                dtype=torch.bfloat16,
                                memory_format=torch.preserve_format,
                            )

        for optim in (self.optimizers if optimizers is None else optimizers):
            optim.register_step_pre_hook(_bf16_state_init_hook)

    def init_cache_state_dict(self) -> None:
        """Initialize cached state dict for TorchFT. No-op for base class."""
        pass


class OptimizersInBackwardContainer(OptimizersContainer):
    """OptimizersContainer for executing ``optim.step()`` in backward pass.

    This class extend ``OptimizersContainer`` to support optimizer step in
    backward pass. ``step()`` and ``zero_grad()`` are no-op in this class.
    Instead, ``register_post_accumulate_grad_hook`` is used to register a hook to
    execute these methods when the gradient is accumulated.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        def __post_init__(self) -> None:
            if self.name == "Muon":
                raise ValueError(
                    "optimizer.name='Muon' is not supported with "
                    "OptimizersInBackwardContainer"
                )
            if self.implementation == "fused_opt_states_bf16":
                raise ValueError(
                    "implementation='fused_opt_states_bf16' is not supported with "
                    "OptimizersInBackwardContainer"
                )
            OptimizersContainer.Config.__post_init__(self)

    def __init__(self, config: Config, *, model_parts: list[nn.Module]) -> None:
        optimizer_cls = self._resolve_optimizer_cls(config.name)
        optimizer_kwargs = self._build_optimizer_kwargs(config)
        all_params = []
        self.model_parts = model_parts
        self.optimizers_by_model_part = []

        # Build a mapping from param -> effective kwargs using param group config
        param_to_kwargs: dict[nn.Parameter, dict[str, Any]] = {}
        for model in self.model_parts:
            param_groups = self._build_param_groups(model, config, optimizer_kwargs)
            for group in param_groups:
                group_kwargs = {k: v for k, v in group.items() if k != "params"}
                for p in group["params"]:
                    param_to_kwargs[p] = group_kwargs

        optim_dict = {}
        for model in self.model_parts:
            for p in model.parameters():
                if p.requires_grad:
                    optim_dict[p] = optimizer_cls([p], **param_to_kwargs[p])
                all_params.append(p)

        def optim_hook(param) -> None:
            optim_dict[param].step()
            optim_dict[param].zero_grad()

        for model in self.model_parts:
            for param in model.parameters():
                if param.requires_grad:
                    param.register_post_accumulate_grad_hook(optim_hook)

        self.optimizers = list(optim_dict.values())
        self.optimizers_by_model_part = [
            [optim_dict[p] for p in model.parameters() if p.requires_grad]
            for model in self.model_parts
        ]

        self._validate_length(
            sum(len(list(model.parameters())) for model in self.model_parts)
        )
        self._post_init(all_params, optimizer_kwargs)

    @overload
    def step(self, closure: None = None) -> None:
        ...

    @overload
    def step(self, closure: Callable[[], float]) -> float:
        ...

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        return None

    def zero_grad(self, set_to_none: bool = True) -> None:
        pass


def register_moe_load_balancing_hook(
    optimizers: OptimizersContainer,
    model_parts: list[nn.Module],
    parallel_dims: ParallelDims,
) -> None:
    """Register an optimizer step pre-hook for MoE auxiliary-loss-free load balancing.

    This function checks if MoE load balancing is enabled and, if so, registers
    a hook that updates expert biases before each optimizer step.

    Args:
        optimizers: The optimizers container to register the hook on.
        model_parts: List of model parts that may contain MoE layers.
        parallel_dims: Parallel dimensions for distributed communication.
    """

    def _should_register_moe_balancing_hook(model_parts: list[nn.Module]) -> bool:
        for model_part in model_parts:
            layers = model_part.get_submodule("layers")
            assert isinstance(layers, nn.ModuleDict)
            for transformer_block in layers.values():
                if transformer_block.moe_enabled:
                    # Assumption: load_balance_coeff is set universally on all moe blocks.
                    # pyrefly: ignore [missing-attribute]
                    return bool(transformer_block.moe.load_balance_coeff)
        return False

    # for MoE auxiliary-loss-free load balancing
    def _is_recomputation_enabled(module):
        return getattr(module, "checkpoint_impl", None) is CheckpointImpl.NO_REENTRANT

    def _update_expert_bias(
        model_parts: list[nn.Module],
        parallel_dims: ParallelDims,
    ):
        loss_mesh = parallel_dims.get_optional_mesh("loss")
        # TODO: Currently this sync is blocking (thus exposed) and happens on the
        # default compute stream. Need to assess if this is OK performance-wise.
        tokens_per_expert_list = []
        for model_part in model_parts:
            layers = model_part.get_submodule("layers")
            assert isinstance(layers, nn.ModuleDict)
            for transformer_block in layers.values():
                if not transformer_block.moe_enabled:
                    continue
                # pyrefly: ignore [missing-attribute]
                if transformer_block.moe.load_balance_coeff is None:
                    return
                # pyrefly: ignore [missing-attribute]
                tokens_per_expert = transformer_block.moe.tokens_per_expert
                if _is_recomputation_enabled(transformer_block):
                    # TODO: This is a hack, we assume with full AC, the tokens_per_expert is counted twice.
                    # This does not affect to expert choice, but affects the experts usage metrics.
                    # We divide by 2 to correct for this double-counting due to recomputation
                    # TODO: new API to help determine if AC is enabled https://github.com/pytorch/pytorch/pull/160888
                    tokens_per_expert = tokens_per_expert // 2
                tokens_per_expert_list.append(tokens_per_expert)

        tokens_per_expert_by_layer = torch.vstack(tokens_per_expert_list)

        if loss_mesh is not None:
            if isinstance(tokens_per_expert_by_layer, torch.distributed.tensor.DTensor):
                tokens_per_expert_by_layer = tokens_per_expert_by_layer.redistribute(
                    placements=[Replicate()]
                    * tokens_per_expert_by_layer.device_mesh.ndim
                )
            else:
                # Perform single all-reduce to get global statistics across all processes
                pg = loss_mesh.get_group()
                torch.distributed.all_reduce(
                    tokens_per_expert_by_layer,
                    group=pg,
                    op=torch.distributed.ReduceOp.SUM,
                )

        moe_layer_idx = 0
        with torch.no_grad():
            for model_part in model_parts:
                layers = model_part.get_submodule("layers")
                assert isinstance(layers, nn.ModuleDict)
                for transformer_block in layers.values():
                    if not transformer_block.moe_enabled:
                        continue
                    moe = transformer_block.moe

                    tokens_per_expert = tokens_per_expert_by_layer[
                        moe_layer_idx
                    ].float()
                    moe_layer_idx += 1

                    # update the expert bias
                    # this is not exactly the same as https://arxiv.org/pdf/2408.15664 proposed
                    # pyrefly: ignore [missing-attribute]
                    expert_bias_delta = moe.load_balance_coeff * torch.sign(
                        tokens_per_expert.mean() - tokens_per_expert
                    )
                    expert_bias_delta = expert_bias_delta - expert_bias_delta.mean()
                    # pyrefly: ignore [missing-attribute]
                    moe.expert_bias.add_(expert_bias_delta)
                    # pyrefly: ignore [missing-attribute]
                    moe.tokens_per_expert.zero_()

    if _should_register_moe_balancing_hook(model_parts):
        optimizers.register_step_pre_hook(
            lambda *args, **kwargs: _update_expert_bias(
                model_parts, parallel_dims=parallel_dims
            )
        )
