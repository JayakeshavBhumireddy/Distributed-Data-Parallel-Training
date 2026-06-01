from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image
from torchvision.datasets import CIFAR10
from tqdm.auto import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(description="Export CIFAR-10 as an ImageFolder dataset.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--download-dir", type=Path, default=Path("./data/raw"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    export_split(dataset=CIFAR10(args.download_dir, train=True, download=True), split="train", output_dir=args.output_dir)
    export_split(
        dataset=CIFAR10(args.download_dir, train=False, download=True),
        split="validation",
        output_dir=args.output_dir,
    )
    print(f"Demo dataset ready: {args.output_dir}")


def export_split(dataset: CIFAR10, split: str, output_dir: Path) -> None:
    split_dir = output_dir / split
    for class_name in dataset.classes:
        (split_dir / class_name).mkdir(parents=True, exist_ok=True)

    for index, (image, target) in enumerate(tqdm(dataset, desc=f"export {split}")):
        class_name = dataset.classes[target]
        path = split_dir / class_name / f"{index:06d}.png"
        if not path.exists():
            image = image if isinstance(image, Image.Image) else Image.fromarray(image)
            image.save(path)


if __name__ == "__main__":
    main()
