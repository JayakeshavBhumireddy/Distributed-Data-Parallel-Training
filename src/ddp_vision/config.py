from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

LogBackend = Literal["none", "wandb", "tensorboard"]

@dataclass(frozen=True)
class TrainConfig:
    experiment_name: str
    data_dir: Path
    work_dir: Path
    epochs: int = 90
    save_checkpoint_interval: int = 10
    num_classes: int = 1000
    batch_size: int = 64
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    momentum: float = 0.9
    step_lr_decay: float = 0.1
    lr_step_size: int = 30
    lr_warmup_epochs: int = 0
    lr_warmup_start_factor: float = 0.1
    bias_weight_decay: bool = False
    norm_weight_decay: bool = False
    max_grad_norm: float = 1.0
    img_size: int = 124
    resize_size: int = 256
    num_workers: int = 8
    resume_from_checkpoint: str | None = None
    model_name: str = "resnet50"
    pretrained: bool = False
    log_with: LogBackend = "none"
    seed: int = 42
    mixed_precision: Literal["no", "fp16", "bf16", "fp8"] = "no"


    def __post_init__(self) -> None:
        object.__setattr__(self, "data_dir", Path(self.data_dir))
        object.__setattr__(self, "work_dir", Path(self.work_dir))
    
    @property
    def experiment_dir(self) -> Path:
        return self.work_dir / self.experiment_name
    
    @property
    def checkpoint_dir(self) -> Path:
        return self.experiment_dir / "checkpoints"



def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train an image classifier with DDP/Accelerate.")
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config file.")
    parser.add_argument("--experiment-name", "--experiment_name", type=str, default=None)
    parser.add_argument("--data-dir", "--path_to_data", type=Path, default=None)
    parser.add_argument("--work-dir", "--working_directory", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--save-checkpoint-interval", "--save_checkpoint_interval", type=int, default=None)
    parser.add_argument("--num-classes", "--num_classes", type=int, default=None)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=None)
    parser.add_argument(
        "--gradient-accumulation-steps", "--gradient_accumulation_steps", type=int, default=None
    )
    parser.add_argument("--learning-rate", "--learning_rate", type=float, default=None)
    parser.add_argument("--weight-decay", "--weight_decay", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--step-lr-decay", "--step_lr_decay", type=float, default=None)
    parser.add_argument("--lr-step-size", "--lr_step_size", type=int, default=None)
    parser.add_argument("--lr-warmup-epochs", "--lr_warmup_epochs", type=int, default=None)
    parser.add_argument(
        "--lr-warmup-start-factor", "--lr_warmup_start_factor", type=float, default=None
    )
    parser.add_argument("--bias-weight-decay", "--bias_weight_decay", action="store_true")
    parser.add_argument("--no-bias-weight-decay", dest="bias_weight_decay", action="store_false")
    parser.set_defaults(bias_weight_decay=None)
    parser.add_argument("--norm-weight-decay", "--norm_weight_decay", action="store_true")
    parser.add_argument("--no-norm-weight-decay", dest="norm_weight_decay", action="store_false")
    parser.set_defaults(norm_weight_decay=None)
    parser.add_argument("--max-grad-norm", "--max_grad_norm", type=float, default=None)
    parser.add_argument("--img-size", "--img_size", type=int, default=None)
    parser.add_argument("--resize-size", "--resize_size", type=int, default=None)
    parser.add_argument("--num-workers", "--num_workers", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", "--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--model-name", "--model_name", type=str, default=None)
    parser.add_argument("--pretrained", action="store_true", default=None)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--log-with", "--log_with", choices=["none", "wandb", "tensorboard"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mixed-precision", "--mixed_precision", choices=["no", "fp16", "bf16", "fp8"], default=None)
    args = parser.parse_args()

    values = _load_config_file(args.config)
    for key, value in vars(args).items():
        if key == "config":
            continue
        if value is not None:
            values[key] = value
    
    missing = [key for key in ("experiment_name", "data_dir", "work_dir") if values.get(key) is None]
    if missing:
        parser.error(f"Missing required arguments: {', '.join('--' + key.replace('_', '-') for key in missing)}")

    return TrainConfig(**values)

def _load_config_file(path: Path | None) -> dict:
    if path is None:
        return {}
    import yaml

    with path.open("r", encoding="utf-8") as file:
        values = yaml.safe_load(file) or {}

    return {key.replace("-", "_"): value for key, value in values.items()}
