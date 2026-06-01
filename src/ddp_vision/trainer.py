from __future__ import annotations

import re
from pathlib import Path

import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ddp_vision.config import TrainConfig
from ddp_vision.data import build_dataloaders
from ddp_vision.logging import LocalLogger
from ddp_vision.models import build_model
from ddp_vision.optim import build_optimizer, build_scheduler


def run_training(config: TrainConfig) -> None:
    config.experiment_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(
        project_dir=str(config.experiment_dir),
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with=None if config.log_with == "none" else config.log_with,
        mixed_precision=config.mixed_precision,
    )
    set_seed(config.seed)

    local_logger = LocalLogger(config.experiment_dir)
    model = build_model(config)
    train_loader, valid_loader = build_dataloaders(config)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    model, optimizer, train_loader, valid_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, valid_loader, scheduler
    )
    accelerator.register_for_checkpointing(scheduler)

    tracker_config = _tracker_config(config, accelerator.num_processes)
    if config.log_with != "none":
        accelerator.init_trackers(config.experiment_name, config=tracker_config)

    start_epoch = _resume_if_requested(accelerator, config)
    accelerator.print(
        f"Training {config.model_name} for {config.epochs} epochs on "
        f"{accelerator.num_processes} process(es). Effective batch size: "
        f"{tracker_config['effective_batch_size']}."
    )

    for epoch in range(start_epoch, config.epochs):
        train_metrics = _train_one_epoch(
            accelerator=accelerator,
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            config=config,
            epoch=epoch,
        )
        valid_metrics = _evaluate(
            accelerator=accelerator,
            model=model,
            loader=valid_loader,
            loss_fn=loss_fn,
            epoch=epoch,
        )

        scheduler.step()
        metrics = {
            "epoch": epoch,
            "learning_rate": scheduler.get_last_lr()[0],
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"valid_{key}": value for key, value in valid_metrics.items()},
        }

        accelerator.print(
            "Epoch "
            f"{epoch}: train_loss={metrics['train_loss']:.4f}, "
            f"train_acc={metrics['train_accuracy']:.4f}, "
            f"valid_loss={metrics['valid_loss']:.4f}, "
            f"valid_acc={metrics['valid_accuracy']:.4f}"
        )

        if accelerator.is_main_process:
            local_logger.log(metrics)
        if config.log_with != "none":
            accelerator.log(metrics, step=epoch)

        if _should_save_checkpoint(epoch, config):
            checkpoint_path = config.checkpoint_dir / f"checkpoint_{epoch}"
            accelerator.save_state(output_dir=str(checkpoint_path))

    if config.log_with != "none":
        accelerator.end_training()


def _train_one_epoch(
    accelerator: Accelerator,
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    epoch: int,
) -> dict[str, float]:
    model.train()
    totals = _MetricTotals()
    progress = tqdm(
        loader,
        desc=f"train epoch {epoch}",
        disable=not accelerator.is_local_main_process,
        leave=False,
    )

    for images, targets in progress:
        with accelerator.accumulate(model):
            outputs = model(images)
            loss = loss_fn(outputs, targets)
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        totals.update(accelerator, outputs.detach(), targets, loss.detach())
        if accelerator.is_local_main_process:
            progress.set_postfix(loss=f"{totals.average_loss:.4f}", acc=f"{totals.accuracy:.4f}")

    return totals.as_dict()


@torch.no_grad()
def _evaluate(
    accelerator: Accelerator,
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    epoch: int,
) -> dict[str, float]:
    model.eval()
    totals = _MetricTotals()
    progress = tqdm(
        loader,
        desc=f"valid epoch {epoch}",
        disable=not accelerator.is_local_main_process,
        leave=False,
    )

    for images, targets in progress:
        outputs = model(images)
        loss = loss_fn(outputs, targets)
        totals.update(accelerator, outputs, targets, loss)
        if accelerator.is_local_main_process:
            progress.set_postfix(loss=f"{totals.average_loss:.4f}", acc=f"{totals.accuracy:.4f}")

    return totals.as_dict()


class _MetricTotals:
    def __init__(self) -> None:
        self.loss_sum = 0.0
        self.correct = 0.0
        self.total = 0.0

    @property
    def average_loss(self) -> float:
        return self.loss_sum / max(self.total, 1.0)

    @property
    def accuracy(self) -> float:
        return self.correct / max(self.total, 1.0)

    def update(
        self,
        accelerator: Accelerator,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        loss: torch.Tensor,
    ) -> None:
        predictions = outputs.argmax(dim=1)
        batch_count = torch.tensor(targets.numel(), device=targets.device, dtype=torch.float32)
        batch_correct = (predictions == targets).sum(dtype=torch.float32)
        batch_loss = loss.float() * batch_count

        gathered = accelerator.gather_for_metrics(
            torch.stack([batch_loss, batch_correct, batch_count])
        ).reshape(-1, 3)
        self.loss_sum += gathered[:, 0].sum().item()
        self.correct += gathered[:, 1].sum().item()
        self.total += gathered[:, 2].sum().item()

    def as_dict(self) -> dict[str, float]:
        return {"loss": self.average_loss, "accuracy": self.accuracy}


def _resume_if_requested(accelerator: Accelerator, config: TrainConfig) -> int:
    if config.resume_from_checkpoint is None:
        return 0

    checkpoint_path = Path(config.resume_from_checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = config.checkpoint_dir / checkpoint_path
    accelerator.print(f"Resuming from checkpoint: {checkpoint_path}")
    accelerator.load_state(str(checkpoint_path))
    match = re.search(r"checkpoint_(\d+)$", checkpoint_path.name)
    return int(match.group(1)) + 1 if match else 0


def _tracker_config(config: TrainConfig, num_processes: int) -> dict[str, int | float | str | bool]:
    return {
        "model_name": config.model_name,
        "num_classes": config.num_classes,
        "epochs": config.epochs,
        "per_device_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "num_processes": num_processes,
        "effective_batch_size": config.batch_size * num_processes * config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "mixed_precision": config.mixed_precision,
        "pretrained": config.pretrained,
    }


def _should_save_checkpoint(epoch: int, config: TrainConfig) -> bool:
    is_interval = config.save_checkpoint_interval > 0 and epoch % config.save_checkpoint_interval == 0
    is_last_epoch = epoch == config.epochs - 1
    return is_interval or is_last_epoch
