import os
import numpy as np
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torchvision.models import resnet50
from accelerate import Accelerator
from torchmetrics import Accuracy
from dotenv import load_dotenv

#from utils import LocalLogger

import warnings 
warnings.filterwarnings("ignore")

load_dotenv()

### Parse Training Arguments ###
parser = argparse.ArgumentParser(description="Arguments for Image Classification Training")
parser.add_argument("--experiment_name", 
                    help="Name of Experiment being Launched", 
                    required=True, 
                    type=str)
parser.add_argument("--path_to_data", 
                    help="Path to ImageNet root folder which should contain \train and \validation folders", 
                    required=True, 
                    type=str)
parser.add_argument("--working_directory", 
                    help="Working Directory where checkpoints and logs are stored, inside a \
                    folder labeled by the experiment name", 
                    required=True, 
                    type=str)
parser.add_argument("--epochs",
                    help="Number of Epochs to Train",
                    default=90, 
                    type=int)
parser.add_argument("--save_checkpoint_interval", 
                    help="After how many epochs to save model checkpoints",
                    default=10,
                    type=int)
parser.add_argument("--num_classes", 
                    help="How many classes is our network predicting?",
                    default=1000,
                    type=int)
parser.add_argument("--batch_size", 
                    help="Effective batch size. If split_batches is false, batch size is \
                         multiplied by number of GPUs utilized ", 
                    default=64, 
                    type=int)
parser.add_argument("--gradient_accumulation_steps", 
                    help="Number of Gradient Accumulation Steps for Training", 
                    default=1, 
                    type=int)
parser.add_argument("--learning_rate", 
                    help="Starting Learning Rate for StepLR", 
                    default=0.1,
                    type=float)
parser.add_argument("--weight_decay", 
                    help="Weight decay for optimizer", 
                    default=1e-4, 
                    type=float)
parser.add_argument("--momentum",
                    help="Momentum parameter for SGD optimizer",
                    default=0.9, 
                    type=float)
parser.add_argument("--step_lr_decay",
                    help="Decay for Step LR", 
                    default=0.1, 
                    type=float)
parser.add_argument("--lr_step_size",
                    help="Number of epochs for every step", 
                    default=30, 
                    type=int)
parser.add_argument("--lr_warmup_start_factor",
                    help="Learning rate start factor (i.e if learning rate is 0.1 and start factor is 0.01, then lr warm-up from 0.1*0.01 to 0.1)",
                    default=0.1, 
                    type=float)
parser.add_argument("--bias_weight_decay",
                    help="Apply weight decay to bias",
                    default=False, 
                    action=argparse.BooleanOptionalAction)
parser.add_argument("--norm_weight_decay",
                    help="Apply weight decay to normalization weight and bias",
                    default=False, 
                    action=argparse.BooleanOptionalAction)
parser.add_argument("--max_grad_norm", 
                    help="Maximum norm for gradient clipping", 
                    default=1.0, 
                    type=float)
parser.add_argument("--img_size", 
                    help="Width and Height of Images passed to model", 
                    default=224, 
                    type=int)
parser.add_argument("--num_workers", 
                    help="Number of workers for DataLoader", 
                    default=32, 
                    type=int)
parser.add_argument("--resume_from_checkpoint", 
                    help="Checkpoint folder for model to resume training from, inside the experiment folder", 
                    default=None, 
                    type=str)
args = parser.parse_args()

### Init Accelerator ###
path_to_experiment = os.path.join(args.working_directory, args.experiment_name)
accelerator = Accelerator(project_dir=path_to_experiment,
                          gradient_accumulation_steps=args.gradient_accumulation_steps,
                          log_with="wandb")

