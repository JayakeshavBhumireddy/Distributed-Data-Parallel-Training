# Multi-GPU DDP Image Training with Accelerate

Production-style starter project for learning distributed data parallel (DDP) training on custom GPU machines such as RunPod, Lambda, vast.ai, or a local multi-GPU workstation.

The project trains a ResNet-50 image classifier from an `ImageFolder` dataset:

```text
data/
  train/
    class_a/*.jpg
    class_b/*.jpg
  validation/
    class_a/*.jpg
    class_b/*.jpg
```

It uses PyTorch, torchvision, Hugging Face Accelerate, torchmetrics, and optional Weights & Biases logging.

## Credit

This project is inspired by Priyam Mazumdar's YouTube video, [GPU go brrr: Reproducing ResNet50 using Huggingface Accelerate & Distributed Training](https://www.youtube.com/watch?v=cHKyhHu6WW0&list=PL16vydMdqFg838pOqpm6p8AqrYBLhPps6&index=8).

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

cp .env.example .env
# Edit .env if you want to use W&B or change default RunPod paths.

# Create a small demo ImageFolder dataset from CIFAR-10.
python scripts/prepare_demo_data.py --output-dir ./data/cifar10-imagefolder

# Single GPU or CPU smoke test.
accelerate launch train.py \
  --experiment-name demo-resnet50 \
  --data-dir ./data/cifar10-imagefolder \
  --work-dir ./runs \
  --num-classes 10 \
  --epochs 2 \
  --batch-size 64 \
  --num-workers 4 \
  --log-with none
```

For multi-GPU training, use the same command with more GPUs available:

```bash
accelerate launch --multi_gpu --num_processes 2 train.py \
  --experiment-name demo-resnet50-2gpu \
  --data-dir ./data/cifar10-imagefolder \
  --work-dir ./runs \
  --num-classes 10 \
  --epochs 10 \
  --batch-size 128 \
  --gradient-accumulation-steps 1 \
  --log-with wandb
```

Effective global batch size is:

```text
batch_size * number_of_gpus * gradient_accumulation_steps
```

## RunPod Setup

1. Create a GPU pod with a PyTorch image, for example an image that already includes CUDA and PyTorch.
2. Open the pod terminal or SSH into it.
3. Clone or upload this project.
4. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
cp .env.example .env
```

5. Prepare demo data:

```bash
python scripts/prepare_demo_data.py --output-dir /workspace/data/cifar10-imagefolder
```

6. Run multi-GPU training:

```bash
source .env
bash scripts/launch_runpod.sh
```

If you have ImageNet or a custom dataset, upload it to `/workspace/data/<name>` with `train` and `validation` folders, then update `DATA_DIR` in `.env`.

## Getting Data Onto a Pod

Good options for quick sessions:

- Demo learning: use `scripts/prepare_demo_data.py`. It downloads CIFAR-10 and writes an ImageFolder layout.
- Small custom dataset: zip your local `train/validation` folders, upload through the RunPod web UI, then run `unzip dataset.zip -d /workspace/data/my-dataset`.
- Larger dataset: store it in S3, GCS, Hugging Face Hub, or a mounted RunPod volume, then sync it into `/workspace/data`.

Example S3 sync:

```bash
aws s3 sync s3://your-bucket/imagenet /workspace/data/imagenet
```

Example Hugging Face dataset download:

```bash
huggingface-cli download <org>/<dataset> --repo-type dataset --local-dir /workspace/data/my-dataset
```

## Project Layout

```text
train.py                         # CLI entrypoint
src/ddp_vision/
  config.py                      # argparse and dataclass config
  data.py                        # ImageFolder datasets and transforms
  logging.py                     # local JSONL/CSV metrics logger
  models.py                      # model factory
  optim.py                       # optimizer and LR scheduler factories
  trainer.py                     # distributed training/evaluation loop
configs/resnet50_imagenet.yaml   # readable reference config
scripts/
  launch_local.sh
  launch_runpod.sh
  prepare_demo_data.py
```

## Resume Training

Checkpoints are saved under:

```text
runs/<experiment-name>/checkpoints/checkpoint_<epoch>
```

Resume with:

```bash
accelerate launch train.py \
  --experiment-name demo-resnet50 \
  --data-dir ./data/cifar10-imagefolder \
  --work-dir ./runs \
  --num-classes 10 \
  --resume-from-checkpoint checkpoint_4
```

## Weights & Biases

To use W&B on a pod:

```bash
cp .env.example .env
# Add your WANDB_API_KEY to .env, then:
source .env
wandb login
```

Then pass `--log-with wandb`. For offline/local-only runs, pass `--log-with none`.

## Notes

- This project intentionally uses `accelerate launch` so learners do not need to hand-write `torch.distributed` boilerplate.
- The training code still uses real DDP under the hood when launched with multiple GPUs.
- Local logs are always written to `metrics.jsonl` and `metrics.csv` in the experiment directory.
