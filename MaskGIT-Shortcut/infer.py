import random

import numpy as np
import argparse

import torch
import torchvision.utils as vutils
from torchvision.utils import save_image

from Trainer.vit import MaskGIT
from Network.shortcut_model import MaskGITShortcut
from pathlib import Path


def viz(x, nrow=10, pad=2, size=(18, 18), save_path='test.png'):
    """
    Visualize a grid of images.

    Args:
        x (torch.Tensor): Input images to visualize.
        nrow (int): Number of images in each row of the grid.
        pad (int): Padding between the images in the grid.
        size (tuple): Size of the visualization figure.

    """
    nb_img = len(x)
    min_norm = x.min(-1)[0].min(-1)[0].min(-1)[0].view(-1, 1, 1, 1)
    max_norm = x.max(-1)[0].max(-1)[0].max(-1)[0].view(-1, 1, 1, 1)
    x = (x - min_norm) / (max_norm - min_norm)

    x = vutils.make_grid(x.float().cpu(), nrow=nrow, padding=pad, normalize=False)
    save_image(x, save_path)


def main(args):
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

    labels, name = [1, 7, 282, 604, 724, 179, 681, 367, 850, random.randint(0, 999)] * 1, "r_row"
    # labels, name = [1] * 1, "r_row"
    labels = torch.LongTensor(labels).to(args.device)

    if args.shortcut:
        shortcut_model = MaskGITShortcut(
            hidden_dim=args.hidden_dim, num_heads=args.num_heads,
            n_block=args.n_block, bottleneck_ratio=args.bottleneck_ratio, 
            embed_t=not args.no_embed_t, dropout=args.dropout,
            weight_path=args.shortcut_path, 
            tok_emb=maskgit.vit.tok_emb, pos_emb=maskgit.vit.pos_emb
        )
        shortcut_model = shortcut_model.to(args.device)
    else:
        shortcut_model = None

    # Generate sample
    gen_sample, gen_code, l_mask, latency = maskgit.sample_shortcut(
        nb_sample=labels.size(0), 
        labels=labels, 
        sm_temp=sm_temp, 
        r_temp=r_temp, 
        w=w, 
        randomize=randomize, 
        sched_mode=sched_mode, 
        step=step,
        use_shortcut=args.shortcut, 
        shortcut_model=shortcut_model, 
        budget=args.budget, 
    )
    print("warmup done.")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    gen_sample, gen_code, l_mask, latency = maskgit.sample_shortcut(
        nb_sample=labels.size(0), 
        labels=labels, 
        sm_temp=sm_temp, 
        r_temp=r_temp, 
        w=w, 
        randomize=randomize, 
        sched_mode=sched_mode, 
        step=step,
        use_shortcut=args.shortcut, 
        shortcut_model=shortcut_model, 
        budget=args.budget, 
    )
    print(f"Sampling time for {labels.size(0)} images: {latency:.3f} seconds")

    if args.output_path is None:
        args.output_path = Path(f"test_s{step}_{f'b{args.budget}' if args.shortcut else 'vanilla'}_{latency:.2f}s.png")
    viz(gen_sample, nrow=5, size=(18, 18), save_path=Path(args.output_path))
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vit_folder", type=str, default="./weights/pretrained_maskgit/MaskGIT/MaskGIT_ImageNet_512.pth", help="Path to the pretrained MaskGIT weights")
    parser.add_argument("--vqgan_folder", type=str, default="./weights/pretrained_maskgit/VQGAN/", help="Path to the pretrained VQGAN weights")
    parser.add_argument("--writer_log", type=str, default="logs", help="Path to the pretrained VQGAN weights")
    parser.add_argument("--mask_value", type=int, default=1024, help="Value of the masked token")
    parser.add_argument("--img_size", type=int, default=512, help="Size of the image")
    parser.add_argument("--vit_size", type=str, default='base', help="Size of the ViT model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--seed", type=int, default=1, help="Seed for reproducibility")
    parser.add_argument("--step", type=int, default=15, help="Number of step")
    parser.add_argument("--sched_mode", type=str, default='arccos', help="Mode of the scheduler")
    parser.add_argument("--cfg_w", type=float, default=2.8, help="Classifier Free Guidance weight")
    parser.add_argument("--r_temp", type=float, default=7, help="Gumbel Temperature")
    parser.add_argument("--sm_temp", type=float, default=1, help="Softmax Temperature")
    parser.add_argument("--randomize", type=str, default="linear", help="Noise scheduler")
    parser.add_argument("--is_multi_gpus", type=bool, default=False)
    parser.add_argument("--is_master", type=bool, default=True)
    parser.add_argument("--iter", type=int, default=1_500_000)
    parser.add_argument("--global_epoch", type=int, default=380)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--debug", type=bool, default=True)
    parser.add_argument("--resume", type=bool, default=True)
    parser.add_argument("--data_folder", type=str, default=None)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--test_only", type=bool, default=False)
    parser.add_argument("--shortcut", action='store_true', help="Use shortcut model")
    parser.add_argument("--shortcut_path", type=str, default="weights/MaskGIT-Shortcut/maskgit-shortcut.pth")
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--n_block", type=int, default=1)
    parser.add_argument("--bottleneck_ratio", type=float, default=2)
    parser.add_argument("--hidden_dim", type=float, default=768)
    parser.add_argument("--tok_emb_size", type=float, default=2026)
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--no_embed_t", action='store_true')
    parser.add_argument("--output_path", type=str, default=None)

    args = parser.parse_args()

    main(args)