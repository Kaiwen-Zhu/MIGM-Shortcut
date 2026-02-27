import random

import numpy as np
import argparse

import torch
from tqdm import tqdm

import webdataset as wds
from pathlib import Path

from Trainer.vit import MaskGIT


class Args(argparse.Namespace):
    data_folder="path/to/imagenet"                          
    vit_folder="./weights/pretrained_maskgit/MaskGIT/MaskGIT_ImageNet_512.pth"
    vqgan_folder="./weights/pretrained_maskgit/VQGAN/"
    writer_log=""
    data = "imagenet"
    mask_value = 1024                                                            # Value of the masked token; also the codebook size of vqgan
    img_size = 512                                                               # Size of the image
    vit_size = 'base'
    path_size = img_size // 16                                                   # Number of vizual token
    seed = 1                                                                     # Seed for reproducibility
    channel = 3                                                                  # Number of input channel
    num_workers = 4                                                              # Number of workers
    iter = 1_500_000                                                             # 750_000 at 256*256 + 750_000 at 512*512
    global_epoch = 380                                                           # 300 epoch w/ bsize 512 + 80 epoch with bsize 128
    lr = 1e-4                                                                    # Learning rate 
    drop_label = 0.1                                                             # Drop out label for cfg
    resume = True                                                                # Set to True for loading model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        # Device
    debug = True                                                                 # Load only the model (not the dataloader)
    test_only = False                                                            # Dont launch the testing
    is_master = True                                                             # Master machine 
    is_multi_gpus = False                                                        # set to False for colab demo

    #### sampling hyperparameters ####
    step = 15
    sched_mode = 'arccos'
    cfg_w = 2.8
    r_temp = 7
    sm_temp = 1
    randomize = "linear"
    #### sampling hyperparameters ####


args=Args()

parser = argparse.ArgumentParser()
parser.add_argument("--feat_save_dir", type=str, required=True)
parser.add_argument("--n_img", type=int, required=True)
parser.add_argument("--seed", type=int, required=True)
cmd_args = parser.parse_args()

args.feat_save_dir = cmd_args.feat_save_dir
args.n_img = cmd_args.n_img
args.seed = cmd_args.seed

print(f"feature save dir: {args.feat_save_dir}; #imgs: {args.n_img}")

# Fixe seed
if args.seed > 0:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.enable = False
    torch.backends.cudnn.deterministic = True

# Instantiate the MaskGIT
maskgit = MaskGIT(args)

sm_temp = args.sm_temp         # Softmax Temperature
r_temp = args.r_temp             # Gumbel Temperature
w = args.cfg_w                  # Classifier Free Guidance
randomize = args.randomize   # Noise scheduler
step = args.step              # Number of step
sched_mode = args.sched_mode  # Mode of the scheduler

Path(args.feat_save_dir).mkdir(exist_ok=True, parents=True)
output_pat = f"{args.feat_save_dir}/%06d.tar"
bs = 10
n_img = args.n_img
n_batch = n_img // bs
with wds.ShardWriter(output_pat, maxcount=40, maxsize=10*1024**3) as feat_sink:
    for b in tqdm(range(n_batch), desc=f"Generating features, {bs} images per batch"):
        labels = [random.randint(0, 999) for _ in range(bs)]
        labels = torch.LongTensor(labels).to(args.device)
        # Generate sample
        gen_sample, gen_code, l_mask = maskgit.sample_makedata(
            nb_sample=labels.size(0), 
            labels=labels, 
            sm_temp=sm_temp, 
            r_temp=r_temp, 
            w=w, 
            randomize=randomize, 
            sched_mode=sched_mode, 
            step=step,
            feat_sink=feat_sink,
            batch_id=b
        )
