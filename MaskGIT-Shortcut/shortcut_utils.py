import torch

import webdataset as wds
import glob

import argparse
import math

from infer import main as infer_shortcut

mask_token_id = 1024

class InferArgs(argparse.Namespace):
    data_folder=None                         
    vit_folder="./weights/pretrained_maskgit/MaskGIT/MaskGIT_ImageNet_512.pth"
    vqgan_folder="./weights/pretrained_maskgit/VQGAN/"
    writer_log=""
    data = "imagenet"
    mask_value = 1024                                                            # Value of the masked token
    img_size = 512                                                               # Size of the image
    vit_size = 'base'
    patch_size = img_size // 16                                                   # Number of vizual token
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

    #### shortcut ####
    shortcut = True
    shortcut_path = None
    budget = 5
    n_block = 1
    bottleneck_ratio = 1
    hidden_dim = 768
    tok_emb_size = 2026
    dropout = 0.
    num_heads = 16
    no_embed_t = False
    #### shortcut ####


def get_samples_factory():
    def get_samples(img_iter):
        for k, feat, generation_order, teacher_vq_ids in img_iter:
            for step in range(feat.shape[0]-1):
                prev_feat = feat[step]
                tgt_feat = feat[step+1]
                new_token_map = generation_order == step
                new_token_ids = teacher_vq_ids[new_token_map]
                tgt_vq_ids = teacher_vq_ids.clone()
                cur_token_ids = teacher_vq_ids.clone()
                tgt_vq_ids[generation_order <= step] = -100
                cur_token_ids[generation_order > step] = mask_token_id
                yield k, step, prev_feat, new_token_ids, new_token_map, tgt_vq_ids, tgt_feat, cur_token_ids
    return get_samples


def build_dataloader(data_path, train=True, collate=True, batch_size=None, 
                     length=100_000, overfit=False):
    dataset = (
        wds.WebDataset(glob.glob(data_path) if isinstance(data_path, str) else data_path,
            resampled=train and not overfit, shardshuffle=False,
            nodesplitter=wds.split_by_node if train else wds.single_node_only,
            handler=wds.warn_and_continue
        )
        .decode("torch")
    )
    dataset = dataset.to_tuple("__key__", "feat.pth", "generation_order.pth", "teacher_vq_ids.pth")
    if collate:
        dataset = dataset.compose(get_samples_factory())
    ld = wds.WebLoader(dataset, batch_size=None, num_workers=8 if train and not overfit else 1, pin_memory=True, prefetch_factor=2)
    ld = ld.batched(batch_size)
    if train:
        if not overfit:
            ld = ld.shuffle(500 if collate else 1)
        ld = ld.with_length(length).with_epoch(length)
    return ld


def gen_images(args, ckpt_name, ckpt_path, save_dir):
    infer_args = InferArgs()
    infer_args.shortcut_path = ckpt_path
    infer_args.n_block = args.n_block
    infer_args.no_embed_t = args.no_embed_t
    infer_args.bottleneck_ratio = args.bottleneck_ratio
    infer_args.hidden_dim = args.hidden_dim
    infer_args.tok_emb_size = args.tok_emb_size
    infer_args.dropout = args.dropout
    infer_args.num_heads = args.num_heads

    infer_args.shortcut = True
    for budget in [10, 8, 5]:
        infer_args.budget = budget
        infer_args.output_path = save_dir / f"{ckpt_name}_T{infer_args.step}_budget{infer_args.budget}.png"
        infer_shortcut(infer_args)

    infer_args.timesteps = args.n_steps
    if not (vanilla_path := save_dir / 'ref_vanilla.png').exists():
        infer_args.shortcut = False
        infer_args.output_path = vanilla_path
        infer_shortcut(infer_args)


def feat2logits(feat, ff_out, vocab_size, compute_map=None):
    return ff_out(feat, compute_map=compute_map)[..., :vocab_size].view(-1, vocab_size)


def get_ema_avg_fn(decay):
    assert 0.0 < decay < 1.0, "Decay must be between 0 and 1"
    def ema_avg_fn(ema_param, new_param, num_averaged):
        return decay * ema_param + (1 - decay) * new_param
    return ema_avg_fn


def get_mask_ratio(total_step):
    r = torch.linspace(1, 0, total_step)
    val_to_mask = torch.arccos(r) / (math.pi * 0.5)
    sche = (val_to_mask / val_to_mask.sum()) * (32 * 32)
    sche = sche.round()
    sche[sche == 0] = 1
    sche[-1] += (32 * 32) - sche.sum()
    sche /= 1024
    sche = torch.cumsum(sche, dim=0)
    return 1 - sche[:-1]
