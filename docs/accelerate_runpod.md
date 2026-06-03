# Accelerate Config on RunPod

You do not have to run `accelerate config` for this project if you use:

```bash
bash scripts/launch_runpod.sh
```

That script passes the important launch values directly:

```bash
accelerate launch --num_processes ... --multi_gpu train.py ...
```

Still, running `accelerate config` is useful for learning and for manual training commands.

## Before Config

Check how many GPUs the pod has:

```bash
nvidia-smi -L
```

Or:

```bash
python -c "import torch; print(torch.cuda.device_count())"
```

## Recommended Answers

Run:

```bash
accelerate config
```

Use these answers for a normal single RunPod machine with one or more GPUs:

```text
In which compute environment are you running?
This machine

Which type of machine are you using?
multi-GPU

How many different machines will you use?
1

Should distributed operations be checked while running for errors?
No

Do you wish to optimize your script with torch dynamo?
No

Do you want to use DeepSpeed?
No

Do you want to use FullyShardedDataParallel?
No

Do you want to use Megatron-LM?
No

How many GPU(s) should be used for distributed training?
Use the number from nvidia-smi -L. Example: 2

What GPU(s) should be used for training?
all

Do you wish to use FP16 or BF16 mixed precision?
bf16
```

Use `bf16` for modern GPUs such as A100, H100, L40, L40S, RTX 4090, and newer Ada/Hopper cards. If the pod GPU does not support BF16, use `fp16`. For debugging, use `no`.

## Manual Launch After Config

After config is saved, you can run:

```bash
accelerate launch train.py \
  --experiment-name resnet50-runpod-ddp \
  --data-dir /workspace/data/cifar10-imagefolder \
  --work-dir /workspace/runs \
  --num-classes 10 \
  --epochs 10 \
  --batch-size 128 \
  --mixed-precision bf16 \
  --log-with wandb
```

## Where Config Is Saved

Accelerate usually writes the config here:

```text
~/.cache/huggingface/accelerate/default_config.yaml
```

You can inspect it with:

```bash
cat ~/.cache/huggingface/accelerate/default_config.yaml
```

If the config gets messy, rerun:

```bash
accelerate config
```

## Simple Rule

For this project:

- Use `scripts/launch_runpod.sh` when you want the easiest path.
- Use `accelerate config` when you want to learn what Accelerate is doing under the hood.
