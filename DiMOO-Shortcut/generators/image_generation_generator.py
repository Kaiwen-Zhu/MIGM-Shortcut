# -*- coding: utf-8 -*-
"""
Image generation generator
"""
import torch
import torch.nn.functional as F
import math
from pathlib import Path
from typing import Callable, Optional
from utils.generation_utils import cosine_schedule, gumbel_max_sample, mask_by_random_topk
from utils.generation_utils import setup_seed
from model import LLaDAForMultiModalGeneration


@torch.no_grad()
def generate_image(
    model,
    prompt: torch.LongTensor,
    *,
    seq_len: int = 1024,
    newline_every: int = 16,
    timesteps: int = 18,
    mask_token_id: int = 126336,
    newline_id: int = 126084,
    temperature: float = 1.0,
    cfg_scale: float = 0.0,
    uncon_ids: torch.LongTensor,
    code_start: Optional[int] = None,
    codebook_size: int = 8192,
    noise_schedule: Callable[[torch.Tensor], torch.Tensor] = cosine_schedule,
    text_vocab_size: Optional[int] = 126356,
    generator: Optional[torch.Generator] = None,
    use_cache=False,
    cache_ratio=0.9,
    refresh_interval=5,
    warmup_ratio=0.3,
) -> torch.LongTensor:
    """
    MaskGit parallel decoding to generate VQ tokens
    
    Args:
        model: Model
        prompt: Prompt tensor
        seq_len: Sequence length
        newline_every: Newline interval per row
        timesteps: Number of timesteps
        mask_token_id: Mask token id
        newline_id: Newline token id
        temperature: Temperature
        cfg_scale: CFG scale
        uncon_ids: Unconditional input
        code_start: Image token satrt index
        codebook_size: Codebook size
        noise_schedule: Noise schedule function
        text_vocab_size: Text vocabulary size
        generator: Random number generator
    
    Returns:
        Final VQ codes (1, seq_len)
    """
    device = next(model.parameters()).device
    prompt = prompt.to(device)
    B, P = prompt.shape
    assert B == 1, "batch>1 not supported – wrap in loop if needed"

    x = prompt
    
    vq_mask = x == mask_token_id
    unknown_cnt = vq_mask.sum(dim=1, keepdim=True)
    vq_len = unknown_cnt

    if isinstance(model, LLaDAForMultiModalGeneration):
        model.caching(use_cache)
    else:  # DDP
        model.module.caching(use_cache)

    warmup_step = int(timesteps * warmup_ratio)
    refresh_steps = torch.zeros(timesteps, dtype=torch.bool)
    for step in range(timesteps):
        if not use_cache or step <= warmup_step or (step-warmup_step) % refresh_interval == 0:
            refresh_steps[step] = True
    compute_ratio = 1 - cache_ratio

    # Infer text vocabulary size
    if text_vocab_size is None:
        vocab_total = model(torch.zeros(1, 1, dtype=torch.long, device=device), infer=True)[0].logits.size(-1)
        text_vocab_size = vocab_total - codebook_size
    vocab_offset = text_vocab_size

    for step in range(timesteps):
        if unknown_cnt.item() == 0:
            break

        # Calculate number of tokens to keep (continue masking) this round
        if step < timesteps - 1:
            frac = noise_schedule(torch.tensor([(step + 1) / timesteps], device=device))
            keep_n = (vq_len.float() * frac).floor().clamp_min(1).long()
        else:
            keep_n = torch.zeros_like(unknown_cnt)

        if use_cache and step and refresh_steps[step]:
            if isinstance(model, LLaDAForMultiModalGeneration):
                model.empty_cache()
            else:  # DDP
                model.module.empty_cache()

        # Forward pass (with/without CFG)
        if cfg_scale > 0:
            uncond = torch.cat((uncon_ids.to(x.device), x[:, code_start-2:]), axis=1)
            uncond_vq_mask = torch.cat((torch.zeros((1, uncon_ids.size()[1]), dtype=torch.bool).to(x.device), vq_mask[:, code_start-2:]), axis=1)
            cond_model_output, cond_feat = model(x, infer=True,
                    cat='cond', use_cache=use_cache, 
                    to_compute_mask = cond_to_compute_mask if not refresh_steps[step] else None,
                )
            cond_logits = cond_model_output.logits[..., vocab_offset : vocab_offset + codebook_size]
            cond_mask_logits = cond_logits[vq_mask].view(B, -1, codebook_size)
            uncond_model_output, uncond_feat = model(uncond, infer=True,
                    cat='uncond', use_cache=use_cache, 
                    to_compute_mask = uncond_to_compute_mask if not refresh_steps[step] else None,
                )
            uncond_logits = uncond_model_output.logits[..., vocab_offset : vocab_offset + codebook_size]
            uncond_mask_logits = uncond_logits[uncond_vq_mask].view(B, -1, codebook_size)
            logits = (1 + cfg_scale) * cond_mask_logits - cfg_scale * uncond_mask_logits
        else:
            logits = model(x, infer=True)[0].logits[:, vq_mask[0], vocab_offset : vocab_offset + codebook_size]

        sampled = gumbel_max_sample(logits, temperature, generator=generator)
        sampled_full = sampled + vocab_offset
        probs = torch.softmax(logits, dim=-1)
        conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

        flat_idx = vq_mask.nonzero(as_tuple=False)[:, 1]
        x.view(-1)[flat_idx] = sampled_full.view(-1)

        conf_map = torch.full_like(x, -math.inf, dtype=probs.dtype)
        conf_map.view(-1)[flat_idx] = conf.view(-1)

        mask_sel = mask_by_random_topk(keep_n.squeeze(1), conf, temperature=temperature, generator=generator)
        x.view(-1)[flat_idx[mask_sel.view(-1)]] = mask_token_id
        vq_mask = x == mask_token_id
        unknown_cnt = vq_mask.sum(dim=1, keepdim=True)

        if use_cache and step < timesteps - 1 and not refresh_steps[step+1]:
            cond_conf = cond_logits.max(dim=-1)[0]
            cond_conf_threshold = torch.quantile(cond_conf.to(torch.float), compute_ratio, dim=-1, keepdim=True)
            cond_to_compute_mask = cond_conf <= cond_conf_threshold

            uncond_conf = uncond_logits.max(dim=-1)[0]
            uncond_conf_threshold = torch.quantile(uncond_conf.to(torch.float), compute_ratio, dim=-1, keepdim=True)
            uncond_to_compute_mask = uncond_conf <= uncond_conf_threshold

    # Remove newline tokens
    vq_ids = x[0, code_start:-2]
    vq_ids = vq_ids[vq_ids != newline_id].view(1, seq_len)
    return vq_ids


@torch.no_grad()
def generate_image_makeshortcutdata(
    model,
    prompt: torch.LongTensor,
    *,
    seq_len: int = 1024,
    newline_every: int = 16,
    timesteps: int = 18,
    mask_token_id: int = 126336,
    newline_id: int = 126084,
    temperature: float = 1.0,
    cfg_scale: float = 0.0,
    uncon_ids: torch.LongTensor,
    code_start: Optional[int] = None,
    codebook_size: int = 8192,
    noise_schedule: Callable[[torch.Tensor], torch.Tensor] = cosine_schedule,
    text_vocab_size: Optional[int] = 126356,
    generator: Optional[torch.Generator] = None,
    use_cache=False,
    cache_ratio=0.9,
    refresh_interval=5,
    warmup_ratio=0.3,
    feat_sink = None,
    img_id = None,
) -> torch.LongTensor:
    """
    MaskGit parallel decoding to generate VQ tokens
    
    Args:
        model: Model
        prompt: Prompt tensor
        seq_len: Sequence length
        newline_every: Newline interval per row
        timesteps: Number of timesteps
        mask_token_id: Mask token id
        newline_id: Newline token id
        temperature: Temperature
        cfg_scale: CFG scale
        uncon_ids: Unconditional input
        code_start: Image token satrt index
        codebook_size: Codebook size
        noise_schedule: Noise schedule function
        text_vocab_size: Text vocabulary size
        generator: Random number generator
    
    Returns:
        Final VQ codes (1, seq_len)
    """
    device = next(model.parameters()).device
    prompt = prompt.to(device)
    B, P = prompt.shape
    assert B == 1, "batch>1 not supported – wrap in loop if needed"

    x = prompt
    
    vq_mask = x == mask_token_id
    unknown_cnt = vq_mask.sum(dim=1, keepdim=True)
    vq_len = unknown_cnt

    if isinstance(model, LLaDAForMultiModalGeneration):
        model.caching(use_cache)
    else:  # DDP
        model.module.caching(use_cache)

    warmup_step = int(timesteps * warmup_ratio)
    refresh_steps = torch.zeros(timesteps, dtype=torch.bool)
    for step in range(timesteps):
        if not use_cache or step <= warmup_step or (step-warmup_step) % refresh_interval == 0:
            refresh_steps[step] = True
    compute_ratio = 1 - cache_ratio

    # Infer text vocabulary size
    if text_vocab_size is None:
        vocab_total = model(torch.zeros(1, 1, dtype=torch.long, device=device), infer=True)[0].logits.size(-1)
        text_vocab_size = vocab_total - codebook_size
    vocab_offset = text_vocab_size

    all_cond_feat = []
    all_uncond_feat = []
    cond_generation_order = torch.full_like(x, -1, dtype=torch.int16)

    for step in range(timesteps):
        if unknown_cnt.item() == 0:
            break

        # Calculate number of tokens to keep (continue masking) this round
        if step < timesteps - 1:
            frac = noise_schedule(torch.tensor([(step + 1) / timesteps], device=device))
            keep_n = (vq_len.float() * frac).floor().clamp_min(1).long()
        else:
            keep_n = torch.zeros_like(unknown_cnt)

        if use_cache and step and refresh_steps[step]:
            if isinstance(model, LLaDAForMultiModalGeneration):
                model.empty_cache()
            else:  # DDP
                model.module.empty_cache()

        # Forward pass (with/without CFG)
        if cfg_scale > 0:
            uncond = torch.cat((uncon_ids.to(x.device), x[:, code_start-2:]), axis=1)
            uncond_vq_mask = torch.cat((torch.zeros((1, uncon_ids.size()[1]), dtype=torch.bool).to(x.device), vq_mask[:, code_start-2:]), axis=1)
            cond_model_output, cond_feat = model(x, infer=True,
                    cat='cond', use_cache=use_cache, 
                    to_compute_mask = cond_to_compute_mask if not refresh_steps[step] else None,
                    ret_last_feat=True
                )
            cond_logits = cond_model_output.logits[..., vocab_offset : vocab_offset + codebook_size]
            cond_mask_logits = cond_logits[vq_mask].view(B, -1, codebook_size)
            uncond_model_output, uncond_feat = model(uncond, infer=True,
                    cat='uncond', use_cache=use_cache, 
                    to_compute_mask = uncond_to_compute_mask if not refresh_steps[step] else None,
                    ret_last_feat=True
                )
            uncond_logits = uncond_model_output.logits[..., vocab_offset : vocab_offset + codebook_size]
            uncond_mask_logits = uncond_logits[uncond_vq_mask].view(B, -1, codebook_size)
            logits = (1 + cfg_scale) * cond_mask_logits - cfg_scale * uncond_mask_logits
        else:
            logits = model(x, infer=True)[0].logits[:, vq_mask[0], vocab_offset : vocab_offset + codebook_size]

        sampled = gumbel_max_sample(logits, temperature, generator=generator)
        sampled_full = sampled + vocab_offset
        probs = torch.softmax(logits, dim=-1)
        conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

        prev_x = x.clone()

        flat_idx = vq_mask.nonzero(as_tuple=False)[:, 1]
        x.view(-1)[flat_idx] = sampled_full.view(-1)

        conf_map = torch.full_like(x, -math.inf, dtype=probs.dtype)
        conf_map.view(-1)[flat_idx] = conf.view(-1)

        mask_sel = mask_by_random_topk(keep_n.squeeze(1), conf, temperature=temperature, generator=generator)
        x.view(-1)[flat_idx[mask_sel.view(-1)]] = mask_token_id
        

        ########## make data begin ##########
        all_cond_feat.append(cond_feat)
        all_uncond_feat.append(uncond_feat)
        cond_generation_order[x != prev_x] = step
        ########## make data end ##########

        vq_mask = x == mask_token_id
        unknown_cnt = vq_mask.sum(dim=1, keepdim=True)

        if use_cache and step < timesteps - 1 and not refresh_steps[step+1]:
            cond_conf = cond_logits.max(dim=-1)[0]
            cond_conf_threshold = torch.quantile(cond_conf.to(torch.float), compute_ratio, dim=-1, keepdim=True)
            cond_to_compute_mask = cond_conf <= cond_conf_threshold

            uncond_conf = uncond_logits.max(dim=-1)[0]
            uncond_conf_threshold = torch.quantile(uncond_conf.to(torch.float), compute_ratio, dim=-1, keepdim=True)
            uncond_to_compute_mask = uncond_conf <= uncond_conf_threshold
    
    ########## make data begin ##########
    all_cond_feat = torch.cat(all_cond_feat, dim=0)
    cond_trajectory = {
        "__key__": f"{img_id:06d}_cond",
        "feat.pth": all_cond_feat.cpu(),
        "generation_order.pth": cond_generation_order.cpu().squeeze(0),
        "teacher_vq_ids.pth": x.cpu().squeeze(0),
    }
    feat_sink.write(cond_trajectory)
    # torch.save(cond_trajectory, feat_save_dir/f"{img_id:06d}_cond.pth")

    all_uncond_feat = torch.cat(all_uncond_feat, dim=0)
    uncond_generation_order = torch.cat(
        (torch.full((1, uncon_ids.size()[1]), -1, dtype=cond_generation_order.dtype).to(x.device), 
        cond_generation_order[:, code_start-2:]), 
        axis=1
    )
    uncond_x = torch.cat(
        (torch.full((1, uncon_ids.size()[1]), -100, dtype=x.dtype).to(x.device), 
        x[:, code_start-2:]), 
        axis=1
    )
    uncond_trajectory = {
        "__key__": f"{img_id:06d}_uncond",
        "feat.pth": all_uncond_feat.cpu(),
        "generation_order.pth": uncond_generation_order.cpu().squeeze(0),
        "teacher_vq_ids.pth": uncond_x.cpu().squeeze(0),
    }
    feat_sink.write(uncond_trajectory)
    ########## make data end ##########

    # Remove newline tokens
    vq_ids = x[0, code_start:-2]
    vq_ids = vq_ids[vq_ids != newline_id].view(1, seq_len)
    return vq_ids


@torch.no_grad()
def generate_image_shortcut(
    model,
    prompt: torch.LongTensor,
    *,
    seq_len: int = 1024,
    newline_every: int = 16,
    timesteps: int = 18,
    mask_token_id: int = 126336,
    newline_id: int = 126084,
    temperature: float = 1.0,
    cfg_scale: float = 0.0,
    uncon_ids: torch.LongTensor,
    code_start: Optional[int] = None,
    codebook_size: int = 8192,
    noise_schedule: Callable[[torch.Tensor], torch.Tensor] = cosine_schedule,
    text_vocab_size: Optional[int] = 126356,
    generator: Optional[torch.Generator] = None,
    use_shortcut=False,
    budget=11,
    shortcut_model=None,
    no_prompt=False
) -> torch.LongTensor:
    """
    MaskGit parallel decoding to generate VQ tokens
    
    Args:
        model: Model
        prompt: Prompt tensor
        seq_len: Sequence length
        newline_every: Newline interval per row
        timesteps: Number of timesteps
        mask_token_id: Mask token id
        newline_id: Newline token id
        temperature: Temperature
        cfg_scale: CFG scale
        uncon_ids: Unconditional input
        code_start: Image token satrt index
        codebook_size: Codebook size
        noise_schedule: Noise schedule function
        text_vocab_size: Text vocabulary size
        generator: Random number generator
    
    Returns:
        Final VQ codes (1, seq_len)
    """
    device = next(model.parameters()).device
    prompt = prompt.to(device)
    B, P = prompt.shape
    assert B == 1, "batch>1 not supported – wrap in loop if needed"

    x = prompt
    
    vq_mask = x == mask_token_id
    unknown_cnt = vq_mask.sum(dim=1, keepdim=True)
    vq_len = unknown_cnt

    refresh_steps = torch.zeros(timesteps, dtype=torch.bool)
    if use_shortcut:
        left_budget = budget - 1
        assert left_budget > 0, f"{left_budget=}"
        left_timesteps = timesteps - 1

        refresh_steps[0] = True
        this_refresh_step = 0
        for k in range(1, left_budget+1):
            this_refresh_step = max(this_refresh_step+1, round((left_timesteps+1) * (k/(left_budget+1))))
            refresh_steps[this_refresh_step] = True

        assert refresh_steps.sum() == budget, f"{refresh_steps.sum()=} != {budget=}"

    if not use_shortcut:
        refresh_steps[:] = True

    # Infer text vocabulary size
    if text_vocab_size is None:
        vocab_total = model(torch.zeros(1, 1, dtype=torch.long, device=device), infer=True)[0].logits.size(-1)
        text_vocab_size = vocab_total - codebook_size
    vocab_offset = text_vocab_size

    ff_out = model.model.transformer.ff_out
    head_weight = ff_out.weight[vocab_offset : vocab_offset + codebook_size]
    head_bias = ff_out.bias
    head_bias = None if head_bias is None else head_bias[vocab_offset : vocab_offset + codebook_size]
    for step in range(timesteps):
        if unknown_cnt.item() == 0:
            break

        # Calculate number of tokens to keep (continue masking) this round
        if step < timesteps - 1:
            frac = noise_schedule(torch.tensor([(step + 1) / timesteps], device=device))
            keep_n = (vq_len.float() * frac).floor().clamp_min(1).long()
        else:
            keep_n = torch.zeros_like(unknown_cnt)

        if use_shortcut and step and refresh_steps[step]:
            if isinstance(model, LLaDAForMultiModalGeneration):
                model.empty_cache()
            else:  # DDP
                model.module.empty_cache()

        # Forward pass (with/without CFG)
        if cfg_scale > 0:
            uncond = torch.cat((uncon_ids.to(x.device), x[:, code_start-2:]), axis=1)
            uncond_vq_mask = torch.cat((torch.zeros((1, uncon_ids.size()[1]), dtype=torch.bool).to(x.device), vq_mask[:, code_start-2:]), axis=1)
            
            if not use_shortcut or refresh_steps[step]:
                cond_model_output, cond_last_feat = model(x, infer=True,
                        cat='cond', to_compute_mask = None, ret_last_feat = use_shortcut
                    )
                cond_logits = cond_model_output.logits
                uncond_model_output, uncond_last_feat = model(uncond, infer=True,
                        cat='uncond', to_compute_mask = None, ret_last_feat = use_shortcut
                    )
                uncond_logits = uncond_model_output.logits

                cond_logits = cond_logits[..., vocab_offset : vocab_offset + codebook_size]
                uncond_logits = uncond_logits[..., vocab_offset : vocab_offset + codebook_size]

                cond_mask_logits = cond_logits[vq_mask].view(B, -1, codebook_size)
                uncond_mask_logits = uncond_logits[uncond_vq_mask].view(B, -1, codebook_size)
            else:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    mask_ratio = torch.tensor([unknown_cnt / vq_len], device=device)
                    cond_last_feat = shortcut_model(cond_last_feat, cond_new_token, cond_new_token_mask, mask_ratio)
                    uncond_last_feat = shortcut_model(uncond_last_feat, uncond_new_token, uncond_new_token_mask, mask_ratio)
                    cond_mask_last_feat = cond_last_feat[vq_mask].view(B, -1, cond_last_feat.shape[-1])
                    uncond_mask_last_feat = uncond_last_feat[uncond_vq_mask].view(B, -1, uncond_last_feat.shape[-1])
                    cond_mask_logits = F.linear(cond_mask_last_feat, head_weight, head_bias)
                    uncond_mask_logits = F.linear(uncond_mask_last_feat, head_weight, head_bias)

            logits = (1 + cfg_scale) * cond_mask_logits - cfg_scale * uncond_mask_logits
        else:
            logits = model(x, infer=True)[0].logits[:, vq_mask[0], vocab_offset : vocab_offset + codebook_size]

        sampled = gumbel_max_sample(logits, temperature, generator=generator)
        sampled_full = sampled + vocab_offset
        probs = torch.softmax(logits, dim=-1)
        conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

        prev_x = x.clone()

        flat_idx = vq_mask.nonzero(as_tuple=False)[:, 1]
        x.view(-1)[flat_idx] = sampled_full.view(-1)

        conf_map = torch.full_like(x, -math.inf, dtype=probs.dtype)
        conf_map.view(-1)[flat_idx] = conf.view(-1)

        mask_sel = mask_by_random_topk(keep_n.squeeze(1), conf, temperature=temperature, generator=generator)
        x.view(-1)[flat_idx[mask_sel.view(-1)]] = mask_token_id
        vq_mask = x == mask_token_id
        unknown_cnt = vq_mask.sum(dim=1, keepdim=True)

        cond_new_token_mask = x != prev_x
        if not no_prompt:
            cond_new_token_mask[..., :code_start-2] = True
        cond_new_token_id = x[cond_new_token_mask].reshape(B, -1)
        cond_new_token = model.model.transformer.wte(cond_new_token_id)
        if cfg_scale > 0:
            uncond_new_token_mask = torch.cat(
                (torch.zeros((1, uncon_ids.size()[1]), dtype=torch.bool).to(x.device), 
                cond_new_token_mask[:, code_start-2:]), 
                axis=1)
            if no_prompt:
                uncond_new_token = cond_new_token
            else:
                uncond_new_token = cond_new_token[:, code_start-2:]

    # Remove newline tokens
    vq_ids = x[0, code_start:-2]
    vq_ids = vq_ids[vq_ids != newline_id].view(1, seq_len)
    return vq_ids