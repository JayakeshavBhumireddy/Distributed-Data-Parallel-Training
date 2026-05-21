from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ddp_vision.config import TrainConfig

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(config: TrainConfig) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=(config.img_size, config.img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    valid_transform = transforms.Compose(
        [
            transforms.Resize((config.resize_size, config.resize_size)),
            transforms.CenterCrop((config.img_size, config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_transform, valid_transform


def build_dataloaders(config: TrainConfig) -> tuple[DataLoader, DataLoader]:
    train_transform, valid_transform = build_transforms(config)
    train_dir = config.data_dir / "train"
    valid_dir = _validation_dir(config.data_dir)

    _require_directory(train_dir)
    _require_directory(valid_dir)

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
    )
    return train_loader, valid_loader


def _validation_dir(data_dir: Path) -> Path:
    validation_dir = data_dir / "validation"
    if validation_dir.exists():
        return validation_dir
    return data_dir / "val"


def _require_directory(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Expected dataset directory does not exist: {path}")
