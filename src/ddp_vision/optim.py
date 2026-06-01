from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ddp_vision.config import TrainConfig

NORM_MODULES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.GroupNorm,
    nn.LayerNorm,
    nn.LocalResponseNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
)


def build_optimizer(model: nn.Module, config: TrainConfig) -> Optimizer:
    if config.bias_weight_decay and config.norm_weight_decay:
        return torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )

    decay_params, no_decay_params = _split_weight_decay_params(model, config)
    return torch.optim.SGD(
        [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        momentum=config.momentum,
    )


def build_scheduler(optimizer: Optimizer, config: TrainConfig) -> LRScheduler:
    step_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.lr_step_size,
        gamma=config.step_lr_decay,
    )
    if config.lr_warmup_epochs <= 0:
        return step_scheduler

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=config.lr_warmup_start_factor,
        total_iters=config.lr_warmup_epochs,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, step_scheduler],
        milestones=[config.lr_warmup_epochs],
    )


def _split_weight_decay_params(
    model: nn.Module, config: TrainConfig
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []
    norm_parameter_names = _norm_parameter_names(model)

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_bias = name.endswith(".bias")
        is_norm = name in norm_parameter_names
        if (is_bias and not config.bias_weight_decay) or (is_norm and not config.norm_weight_decay):
            no_decay_params.append(parameter)
        else:
            decay_params.append(parameter)

    return decay_params, no_decay_params


def _norm_parameter_names(model: nn.Module) -> set[str]:
    names: set[str] = set()
    for module_name, module in model.named_modules():
        if isinstance(module, NORM_MODULES):
            for parameter_name, _ in module.named_parameters(recurse=False):
                full_name = f"{module_name}.{parameter_name}" if module_name else parameter_name
                names.add(full_name)
    return names