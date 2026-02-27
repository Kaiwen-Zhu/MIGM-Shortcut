import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.amp as amp
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.swa_utils import AveragedModel

import webdataset as wds
import glob
import os
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import wandb
import datetime

from Network.shortcut_model import MaskGITShortcut

from shortcut_utils import build_dataloader, gen_images, feat2logits, get_ema_avg_fn, mask_token_id, get_mask_ratio, InferArgs
from Trainer.vit import MaskGIT
import math


def parse_args():
    parser = argparse.ArgumentParser(description="Train MDM Shortcut Model")
    parser.add_argument("--exp_name", type=str, default='debug', help="Experiment name for logging")
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--model_path", type=str, default=None, help="Path to the pretrained model")
    parser.add_argument("--vocab_size", type=int, default=1024)
    parser.add_argument("--n_block", type=int, default=1, help="Number of blocks in the model; each block contains a cross attention block and a self attention block")
    parser.add_argument("--token_embedding_path", type=str, default='weights/token_embedding.pt', help="Path to the pretrained token embedding")
    parser.add_argument("--position_embedding_path", type=str, default='weights/position_embedding.pt', help="Path to the pretrained position embedding")
    parser.add_argument("--ff_out_path", type=str, default='weights/ff_out.pt', help="Path to the pretrained linear layer transforming feature to logits")
    parser.add_argument("--train_data_path", type=str, default='dataset/train/feat/*.tar', help="Path to the training data")
    parser.add_argument("--val_data_path", type=str, default='dataset/val/feat/*.tar', help="Path to the validation data")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training (per GPU)")
    parser.add_argument("--accum_steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--step_decay", action='store_true', help="Whether to use step decay for learning rate")
    parser.add_argument("--wd", type=float, default=1e-2, help="Weight decay for optimizer")
    parser.add_argument("--n_steps", type=int, default=15)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--log_interval", type=int, default=50, help="Logging interval")
    parser.add_argument("--val_interval", type=int, default=100_000, help="Validation interval")
    parser.add_argument("--mse_weight", type=float, default=1)
    parser.add_argument("--kl_weight", type=float, default=0)
    parser.add_argument("--T", type=float, default=1.0, help="Temperature for distillation KL loss")
    parser.add_argument("--overfit", action='store_true')
    parser.add_argument("--ema_decay", type=float, default=None, help="EMA decay rate")
    parser.add_argument("--max_rollout_step", type=int, default=1)
    parser.add_argument("--switch_rollout_step_interval", type=int, default=1)
    parser.add_argument("--val_max_rollout_step", type=int, default=None)
    parser.add_argument("--no_embed_t", action='store_true')
    parser.add_argument("--teacher_online_rollout", action='store_true')
    parser.add_argument("--bottleneck_ratio", type=float, default=1)
    parser.add_argument("--hidden_dim", type=float, default=768)
    parser.add_argument("--tok_emb_size", type=float, default=2026)
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.)
    parser.add_argument("--num_heads", type=int, default=16)
    return parser.parse_args()


def setup_dist():
    if 'LOCAL_RANK' not in os.environ:
        raise EnvironmentError("This script must be launched using torchrun or have DDP environment variables set.")

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=60))
    print(f"Initialized process rank {rank}/{world_size-1} (Local Rank: {local_rank})")

    return local_rank, world_size


def build_everything(args, local_rank, world_size):
    DEVICE = torch.device(f'cuda:{local_rank}')

    if local_rank == 0:
        print(f"Loading token embedding from {args.token_embedding_path}...")
    tok_emb = nn.Embedding(args.tok_emb_size, args.hidden_dim)
    state_dict = torch.load(args.token_embedding_path, map_location='cpu')
    tok_emb.load_state_dict(state_dict)
    for param in tok_emb.parameters():
        param.requires_grad = False
    tok_emb = tok_emb.to(DEVICE)

    if local_rank == 0:
        print(f"Loading position embedding from {args.position_embedding_path}...")
    pos_emb = nn.Parameter(torch.zeros(1, (args.patch_size*args.patch_size)+1, args.hidden_dim))
    state_dict = torch.load(args.position_embedding_path, map_location='cpu')
    pos_emb.data.copy_(state_dict)
    pos_emb.requires_grad = False
    pos_emb = pos_emb.to(DEVICE)

    if local_rank == 0:
        print(f"Loading model{f' from {args.model_path}' if args.model_path else ''}...")

    model = MaskGITShortcut(
        hidden_dim=args.hidden_dim, num_heads=args.num_heads,
        n_block=args.n_block, bottleneck_ratio=args.bottleneck_ratio, 
        embed_t=not args.no_embed_t, dropout=args.dropout,
        weight_path=args.model_path, 
        tok_emb=tok_emb, pos_emb=pos_emb
    )
    model = model.to(DEVICE)
    # model: MaskGITShortcut = torch.compile(model)
    model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)

    if args.ema_decay is not None and local_rank == 0:
        ema_avg_fn = get_ema_avg_fn(args.ema_decay)
        ema_model = AveragedModel(model, avg_fn=ema_avg_fn, device=DEVICE)
    else:
        ema_model = None

    if args.teacher_online_rollout:
        maskgit_args = InferArgs()
        maskgit_args.infer = True
        if local_rank == 0:
            print(f"Loading teacher model from {maskgit_args.vit_folder} for online rollout...")
        t_model = MaskGIT(maskgit_args)
        for param in t_model.vit.parameters():
            param.requires_grad = False
    else:
        t_model = None

    if local_rank == 0:
        print(f"Loading ff_out from {args.ff_out_path}...")
    ffout = torch.load(args.ff_out_path, map_location='cpu')
    ffout['weight'] = ffout['weight'].to(DEVICE).detach()
    ffout['bias'] = ffout['bias'].to(DEVICE).detach()
    def orig_ff_out(x, compute_map=None):
        bias = ffout['bias']
        if compute_map is not None:
            bias = bias[compute_map.view(1025)]
        return torch.matmul(x, ffout['weight']) + bias

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_sche = MultiStepLR(optim, milestones=[1000], gamma=0.1) if args.step_decay else None

    if args.debug:
        args.train_data_path = glob.glob(args.train_data_path)[0]
        args.val_data_path = glob.glob(args.val_data_path)[0]
        args.log_interval = 1
        args.val_interval = 10
    elif args.overfit:
        args.train_data_path = glob.glob(args.train_data_path)[0]
        args.log_interval = 10
        args.val_interval = 50000
        args.epochs = 5000
        assert world_size == 1
    if local_rank == 0:
        print(f"Loading training dataset from {args.train_data_path}...")
    ld_train = build_dataloader(args.train_data_path, train=True, collate=True, 
                                batch_size=args.batch_size, length=100_000, overfit=args.overfit)

    if local_rank == 0:
        print(f"Loading validation dataset from {args.val_data_path}...")
    if args.val_max_rollout_step is None:
        args.val_max_rollout_step = args.max_rollout_step
    ld_val = build_dataloader(args.val_data_path, train=False, collate=True, batch_size=1)
    
    args.mask_ratio_sche = get_mask_ratio(args.n_steps)

    return model, ema_model, orig_ff_out, optim, lr_sche, ld_train, ld_val, t_model


def cleanup_dist():
    dist.destroy_process_group()


def log(args, loss, loss_mse, loss_kl, prev_feat, est_feat, tgt_feat, 
        orig_ff_out, lr, global_iter, val_kl=False, mask_map=None):
    with torch.no_grad():
        with amp.autocast('cuda', enabled=True):
            wandb_log_dict = {}

            wandb_log_dict["loss/train/total"] = loss.item() * args.accum_steps
            wandb_log_dict["loss/train/MSE"] = loss_mse.item() * args.accum_steps
            if args.kl_weight or val_kl:
                wandb_log_dict["loss/train/KL"] = loss_kl.item() * args.accum_steps

            orig_cos_sim = F.cosine_similarity(prev_feat, tgt_feat, dim=-1)
            pred_cos_sim = F.cosine_similarity(est_feat, tgt_feat, dim=-1)
            wandb_log_dict["original metric/cosine similarity"] = orig_cos_sim.mean().item()
            wandb_log_dict["prediction metric/cosine similarity"] = pred_cos_sim.mean().item()
            wandb_log_dict["prediction metric/cosine similarity min"] = pred_cos_sim.min().item()
            wandb_log_dict["prediction metric/cosine similarity 10th pctl"] = torch.quantile(pred_cos_sim, 0.1).item()
            wandb_log_dict["improvement/cosine similarity"] = pred_cos_sim.mean().item() - orig_cos_sim.mean().item()
            orig_mse = F.mse_loss(prev_feat, tgt_feat).item()
            pred_mse = F.mse_loss(est_feat, tgt_feat).item()
            wandb_log_dict["original metric/MSE"] = orig_mse
            wandb_log_dict["prediction metric/MSE"] = pred_mse
            wandb_log_dict["improvement/MSE"] = orig_mse - pred_mse
            if args.kl_weight or val_kl:
                prev_logits = feat2logits(prev_feat, orig_ff_out, args.vocab_size, mask_map)
                est_logits = feat2logits(est_feat, orig_ff_out, args.vocab_size, mask_map)
                tgt_logits = feat2logits(tgt_feat, orig_ff_out, args.vocab_size, mask_map)
                orig_ce = F.kl_div(F.log_softmax(prev_logits / args.T, dim=-1), F.softmax(tgt_logits / args.T, dim=-1), 
                                reduction='batchmean').item()
                pred_ce = F.kl_div(F.log_softmax(est_logits / args.T, dim=-1), F.softmax(tgt_logits / args.T, dim=-1), 
                                reduction='batchmean').item()
                wandb_log_dict["original metric/KL"] = orig_ce
                wandb_log_dict["prediction metric/KL"] = pred_ce
                wandb_log_dict["improvement/KL"] = orig_ce - pred_ce

            wandb_log_dict["misc/learning rate"] = lr

            wandb.log(wandb_log_dict, step=global_iter)


def val(args, global_iter, it, epoch, world_size, device, 
        model: MaskGITShortcut, orig_ff_out: nn.Linear,
        ld_val: wds.WebLoader, weight_save_dir: Path, img_save_dir: Path, t_model, val_kl=False):
    model.eval()
    ckpt_name = f"{(global_iter+1) // args.val_interval:02d}_epoch{epoch:02d}_iter{it:06d}"
    ckpt_path = weight_save_dir / f"{ckpt_name}.pth"
    raw_model = model.module
    if isinstance(raw_model, torch._dynamo.eval_frame.OptimizedModule):
        torch.save(raw_model._orig_mod.state_dict(), ckpt_path) 
    else:
        torch.save(raw_model.state_dict(), ckpt_path)
    
    print(f"Model checkpoint saved to {ckpt_path}")
    if args.overfit:
        return

    os.environ["WORLD_SIZE"] = '1'
    with torch.no_grad():
        val_s_log = {}
        for val_it, batch in enumerate(tqdm(ld_val, desc='Validation', ncols=80)):
            rollout_step = (val_it // args.switch_rollout_step_interval) % args.val_max_rollout_step + 1
            loss, loss_mse, loss_kl, prev_feat, est_feat, tgt_feat, gt_vq_ids, mask_map = \
                run_step(args, batch, model, orig_ff_out, device, t_model, rollout_step, val_kl=val_kl)
            cos_sim = F.cosine_similarity(est_feat, tgt_feat, dim=-1).mean()
            if rollout_step not in val_s_log:
                val_s_log[rollout_step] = {'cosine similarity': [], 'MSE': [], 'KL': []}
            val_s_log[rollout_step]['cosine similarity'].append(cos_sim.item())
            val_s_log[rollout_step]['MSE'].append(loss_mse.item())
            val_s_log[rollout_step]['KL'].append(loss_kl.item())

        for s, val_log in val_s_log.items():
            cos_sim = sum(val_log['cosine similarity']) / len(val_log['cosine similarity'])
            mse = sum(val_log['MSE']) / len(val_log['MSE'])
            kl = sum(val_log['KL']) / len(val_log['KL'])
            wandb.log({
                f"val_s{s}/cosine similarity": cos_sim, 
                f"val_s{s}/MSE": mse, 
                f"val_s{s}/KL": kl
            }, step=global_iter)

    os.environ["WORLD_SIZE"] = str(world_size)
    model.train()

    gen_images(args, ckpt_name, ckpt_path, img_save_dir)


def loss_fn(args, est_feat, tgt_feat, orig_ff_out, val_kl=False, mask_map=None):
    loss_mse = F.mse_loss(est_feat, tgt_feat)
        
    if args.kl_weight or val_kl:
        est_logits = feat2logits(est_feat, orig_ff_out, args.vocab_size, mask_map)
        tgt_logits = feat2logits(tgt_feat, orig_ff_out, args.vocab_size, mask_map)
        loss_kl = F.kl_div(F.log_softmax(est_logits / args.T, dim=-1), F.softmax(tgt_logits / args.T, dim=-1), 
                            reduction='batchmean')
    else:
        loss_kl = torch.tensor(0.0, device=est_feat.device)
    
    loss = args.kl_weight * args.T * args.T * loss_kl + args.mse_weight * loss_mse
    loss = loss / args.accum_steps
    return loss, loss_mse, loss_kl


def sample_from_feat(feat, ff_out, token_ids, vocab_size, n_to_gen,
                     step=None, sm_temp=1, r_temp=7):
    B, T = token_ids.shape
    cur_mask = token_ids == mask_token_id
    assert torch.all(cur_mask[:-1])

    logits = ff_out(feat)[..., :vocab_size]
    prob = torch.softmax(logits * sm_temp, -1)
    distri = torch.distributions.Categorical(probs=prob)
    pred_code = distri.sample()
    conf = torch.gather(prob, 2, pred_code.view(B, T, 1))

    ratio = (step / (15-1))
    rand = r_temp * np.random.gumbel(size=(B, T)) * (1 - ratio)
    conf = torch.log(conf.squeeze()) + torch.from_numpy(rand).to(token_ids.device)

    conf[~cur_mask.bool()] = -math.inf
    
    thresh_conf, indice_mask = torch.topk(conf.view(B, -1), k=n_to_gen, dim=-1)
    thresh_conf = thresh_conf[:, -1]

    conf = (conf >= thresh_conf.unsqueeze(-1))
    f_mask = (cur_mask.float() * conf.float()).bool()
    token_ids[f_mask] = pred_code[f_mask]

    new_token_map = f_mask
    new_token_ids = pred_code[f_mask].view(B, n_to_gen)

    return token_ids, new_token_ids, new_token_map


def rollout(model, ff_out, feat, mask_ratio, token_ids=None, 
            next_token_ids=None, new_token_ids=None, new_token_map=None, 
            no_grad=True, vocab_size=8192, step=None, n_to_gen=None):
    """Sample hat{T}_{t+k-1} from hat{f}_{t+k-1}, merge with hat{T}_{<t+k-1} to get hat{T}_{<t+k}, and
    perform one student step: hat{f}_{t+k} = S(hat{f}_{t+k-1}, hat{T}_{t+k-1}). 
    For k=1, hat{T}_{<t+k} is directly given, so we only need to perform one student step with provided T_{t+k-1}.

    Args:
        feat: hat{f}_{t+k-1}
        token_ids: hat{T}_{<t+k-1}; masked tokens are marked by mask_token_id; not needed for k=1

    Returns:
        next_feat: hat{f}_{t+k}
        next_token_ids: hat{T}_{<t+k}; directly given for k=1
    """

    with amp.autocast('cuda'):
        if next_token_ids is None:
            next_token_ids, new_token_ids, new_token_map = sample_from_feat(
                feat, ff_out, token_ids, vocab_size, n_to_gen, step=step)
        else:
            assert token_ids is None
            assert new_token_ids is not None
            assert new_token_map is not None
        if no_grad:
            with torch.no_grad():
                next_feat = model(feat, new_token_ids, new_token_map, mask_ratio)
        else:
            next_feat = model(feat, new_token_ids, new_token_map, mask_ratio)
    return next_feat, next_token_ids


def run_step(args, batch, model, orig_ff_out, device, t_model, rollout_step, 
             val_kl=False, n_total=1024):
    img_key, step, prev_feat, new_token_ids, new_token_map, gt_vq_ids, tgt_feat, cur_token_ids = batch
    prev_feat = prev_feat.to(device)
    new_token_ids = new_token_ids.to(device)
    new_token_map = new_token_map.to(device)
    gt_vq_ids = gt_vq_ids.to(device)
    tgt_feat = tgt_feat.to(device)
    cur_token_ids = cur_token_ids.to(device)

    img_key = [k.split('_')[-2:] for k in img_key]
    uncond_lst = [k[1] == 'uncond' for k in img_key]
    drop_label = torch.tensor(uncond_lst).to(device)
    cls_lst = [int(k[0]) for k in img_key]
    cls_lst = torch.tensor(cls_lst, device=cur_token_ids.device)
    cls_lst[drop_label] = 2025
    assert torch.all(cur_token_ids[:,-1] == cls_lst), f"Class token mismatch! Expected {cls_lst}, got {cur_token_ids[:,-1]}"

    rollout_step = min(rollout_step, args.n_steps - step.max() - 1)

    cur_feat = prev_feat
    for k in range(1, rollout_step+1):
        mask_ratio = args.mask_ratio_sche[step + k-1]  # shape: (B,); the mask ratio before step+k
        prev_mask_ratio = args.mask_ratio_sche[step + k-2].item() if step+k-2>=0 else 1
        n_to_gen = round(n_total * prev_mask_ratio) - round(n_total * mask_ratio.item())
        # hat{f}_{t+k}, hat{T}_{<t+k}
        cur_feat, cur_token_ids = rollout(
            model, orig_ff_out, cur_feat, mask_ratio, 
            token_ids = cur_token_ids if k!=1 else None, 
            next_token_ids = None if k!=1 else cur_token_ids,
            new_token_ids = None if k!=1 else new_token_ids,
            new_token_map = None if k!=1 else new_token_map,
            no_grad = k < rollout_step, vocab_size=args.vocab_size,
            step=step+k-1, n_to_gen=n_to_gen)

    B, T, D = prev_feat.shape
    est_feat = cur_feat
    if rollout_step > 1:
        label = cur_token_ids[:,-1]
        img_token = cur_token_ids[:,:-1].reshape(B, 32, 32)
        assert torch.all(label >= 0) and torch.all(label < 1000 or label == 2025)
        _, tgt_feat = t_model.vit(img_token, label, drop_label, ret_last_feat=True)

    mask_map = cur_token_ids == mask_token_id
    prev_feat = prev_feat[mask_map].view(B, -1, D)
    est_feat = est_feat[mask_map].view(B, -1, D)
    tgt_feat = tgt_feat[mask_map].view(B, -1, D)
    with amp.autocast('cuda'):
        loss, loss_mse, loss_kl = loss_fn(args, est_feat, tgt_feat, orig_ff_out, val_kl, mask_map)

    return loss, loss_mse, loss_kl, prev_feat, est_feat, tgt_feat, gt_vq_ids, mask_map


def main_worker(args):
    local_rank, world_size = setup_dist()
    
    seed = args.seed + local_rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    (
        model, ema_model, orig_ff_out, 
        optim, lr_sche, ld_train, ld_val, t_model
    ) = build_everything(args, local_rank, world_size)
    
    is_main_process = (local_rank == 0)
    
    weight_save_dir = Path("weights/shortcut") / args.exp_name
    img_save_dir = Path("shortcut_val") / args.exp_name
    if is_main_process:
        os.environ["WANDB_MODE"] = "offline"
        wandb.init(project="maskgit-shortcut" if not args.debug else "maskgit-shortcut-debug", name=args.exp_name, config=vars(args))
        weight_save_dir.mkdir(parents=True, exist_ok=True)
        img_save_dir.mkdir(parents=True, exist_ok=True)
    
    scaler = amp.GradScaler()
    
    device = torch.device(f'cuda:{local_rank}')
    global_iter = 0
    model.train()
    for epoch in range(args.epochs):
        for it, batch in enumerate(tqdm(ld_train, disable=not is_main_process, 
                                        ncols=80, desc=f"Epoch {epoch}/{args.epochs-1}")):
            rollout_step = (global_iter // args.switch_rollout_step_interval) % args.max_rollout_step + 1
            loss, loss_mse, loss_kl, prev_feat, est_feat, tgt_feat, gt_vq_ids, mask_map = \
                run_step(args, batch, model, orig_ff_out, device, t_model, rollout_step, val_kl=True)
            
            scaler.scale(loss).backward()
            if (global_iter + 1) % args.accum_steps == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()
                if args.step_decay:
                    lr_sche.step()
                if ema_model is not None:
                    ema_model.update_parameters(model)
            
            if is_main_process:
                if (global_iter + 1) % args.log_interval == 0:
                    log(args, loss, loss_mse, loss_kl, prev_feat, est_feat, tgt_feat, 
                        orig_ff_out, optim.param_groups[0]['lr'], global_iter, 
                        val_kl=True, mask_map=mask_map)
                if (global_iter+1) % args.val_interval == 0:
                    val(args, global_iter, it, epoch, world_size, device, 
                        model if ema_model is None else ema_model.module,
                        orig_ff_out, ld_val, weight_save_dir, img_save_dir, t_model, val_kl=True)

            global_iter += 1
            dist.barrier()


def main():
    args = parse_args()
    assert args.batch_size == 1
    
    main_worker(args)

    if dist.is_initialized():
        cleanup_dist()


if __name__ == "__main__":
    main()
