# -*- coding: utf-8 -*-
"""
Text-to-image inference script (DDP version)
"""
import os
import json
import argparse
import time
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoConfig, AutoTokenizer
from diffusers import VQModel
from pathlib import Path
import sys
from PIL import Image
import numpy as np
import datetime
import webdataset as wds

from config import SPECIAL_TOKENS
from model import LLaDAForMultiModalGeneration
from utils.image_utils import decode_vq_to_image, calculate_vq_params, add_break_line
from generators.image_generation_generator import generate_image_makeshortcutdata
from utils.generation_utils import setup_seed
from utils.prompt_utils import generate_text_to_image_prompt, create_prompt_templates


class PromptDataset(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


def is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

def encode_img_to_vq(vqvae, img_path, expected_size=None, target_size=None):
    img = Image.open(img_path).convert('RGB')
    if expected_size is not None:
        assert img.size == expected_size, f"Image size {img.size} does not match expected size {expected_size}"
        img = img.resize(target_size, resample=Image.BICUBIC)
    img = torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    # img = 2.0 * img - 1.0
    img = img.cuda()
    latents = vqvae.module.encode(img).latents
    lat_bsz, channels, lat_h, lat_w = latents.shape
    quantized = vqvae.module.quantize(latents)[2][2].reshape(lat_bsz, lat_h, lat_w)  # (1, 32, 32), torch.int64
    return quantized


def main():
    parser = argparse.ArgumentParser(description="Text-to-image inference (DDP version)")
    parser.add_argument("--checkpoint", type=str, default="weights/Lumina-DiMOO", help="Fine-tuned checkpoint path")
    parser.add_argument("--prompt_path", type=str, required=True, help="Prompt file path(.json/.jsonl/.txt)")
    parser.add_argument("--n_img", type=int, default=-1)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--timesteps", type=int, default=64, help="Number of timesteps")
    parser.add_argument("--cfg_scale", type=float, default=4.0, help="CFG scale")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--vae_ckpt", type=str, default="weights/Lumina-DiMOO", help="VAE checkpoint path")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--img_save_dir", type=str, default="shortcut_data/img", help="Output directory")
    parser.add_argument("--meta_save_dir", type=str, default="shortcut_data/meta", help="Output directory")
    parser.add_argument("--feat_save_dir", type=str, default="shortcut_data/feat")
    
    args = parser.parse_args()
    
    # Special tokens
    MASK = SPECIAL_TOKENS["mask_token"]
    NEW_LINE = SPECIAL_TOKENS["newline_token"]
    BOA = SPECIAL_TOKENS["answer_start"]  # Begin of Answer
    EOA = SPECIAL_TOKENS["answer_end"]    # End of Answer
    BOI = SPECIAL_TOKENS["boi"]           # Begin of Image
    EOI = SPECIAL_TOKENS["eoi"]           # End of Image

    # Initialize distributed
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=30))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    # Set random seed
    if int(args.seed) != 0:
        setup_seed(args.seed + local_rank)
    
    img_dir = Path(args.img_save_dir)
    json_dir = Path(args.meta_save_dir)
    if is_main_process():
        img_dir.mkdir(parents=True, exist_ok=True)
        json_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = LLaDAForMultiModalGeneration.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16,
    )
    
    # Wrap with DDP
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    vqvae = VQModel.from_pretrained(args.vae_ckpt, subfolder="vqvae").to(device)
    vqvae = DDP(vqvae, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    # Read prompts
    with open(args.prompt_path, "r", encoding="utf-8") as f:
        raw = f.read()
    prompts = [line.strip() for line in raw.splitlines() if line.strip()]

    # Create dataset and data loader
    prompts = prompts[args.start_idx:]
    if args.n_img != -1:
        prompts = prompts[:args.n_img]
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    n_prompt_per_proc = len(prompts) // world_size
    start_idx = rank * n_prompt_per_proc
    end_idx = (rank + 1) * n_prompt_per_proc if rank != world_size - 1 else len(prompts)
    prompts = prompts[start_idx:end_idx]
    
    # Calculate VQ parameters
    seq_len, newline_every, token_grid_height, token_grid_width = calculate_vq_params(args.height, args.width)
    
    if is_main_process():
        print(f"Generate image size: {args.height}x{args.width}")
        print(f"Calculated VQ sequence length: {seq_len}")
        print(f"Tokens per line (newline_every): {newline_every}")
    
    # Get prompt templates
    templates = create_prompt_templates()
    
    # rank-specific jsonl, avoid write conflicts
    rank = dist.get_rank()
    per_rank_json = json_dir / f"meta_rank{rank}.json"
    
    time_list = []
    
    Path(args.feat_save_dir).mkdir(exist_ok=True, parents=True)
    output_pat = f"{args.feat_save_dir}/{rank}_%06d.tar"
    with wds.ShardWriter(output_pat, maxcount=2, maxsize=10*1024**3) as feat_sink:
        # Main loop (each rank processes its own subset)
        all_res = []
        for i, prompt_text in enumerate(prompts):
            global_idx = start_idx + i
            
            if is_main_process():
                print(f"Processing prompt {i+1}/{len(prompts)}: {prompt_text}")

            # Generate filename
            words = prompt_text.split()
            filename_words = words[:10] if len(words) > 10 else words
            filename = "_".join(filename_words)
            filename = "".join(c for c in filename if c.isalnum() or c in ('_', '-'))
            filename = f"{global_idx}_{filename}_{args.height}x{args.width}_t{args.timesteps}_cfg{args.cfg_scale}_seed{args.seed}.png"
            save_path = os.path.join(img_dir, filename)

            # Generate prompts using utility function
            input_prompt, uncon_prompt = generate_text_to_image_prompt(prompt_text, templates)
            
            # build initial sequence
            con_prompt_token = tokenizer(input_prompt)["input_ids"]
            uncon_prompt_token = tokenizer(uncon_prompt)["input_ids"]
            
            # build image mask predition
            img_mask_token = add_break_line([MASK] * seq_len, token_grid_height, token_grid_width, new_number = NEW_LINE)
            img_pred_token = [BOA] + [BOI] + img_mask_token + [EOI] + [EOA]

            prompt_ids = torch.tensor(con_prompt_token + img_pred_token, device=device).unsqueeze(0)
            uncon_ids = torch.tensor(uncon_prompt_token, device=device).unsqueeze(0)

            # image satrt index
            code_start = len(con_prompt_token) + 2 
            
            # Generate VQ tokens
            start_time = time.time()
            vq_tokens = generate_image_makeshortcutdata(
                model,
                prompt_ids,
                seq_len=seq_len,
                newline_every=newline_every,
                timesteps=args.timesteps,
                temperature=args.temperature,
                cfg_scale=args.cfg_scale,
                uncon_ids=uncon_ids,
                code_start=code_start,
                feat_sink=feat_sink,
                img_id=global_idx,
            )
            
            # Decode VQ codes to PNG and save
            decode_vq_to_image(
                vq_tokens, 
                save_path=save_path, 
                vqvae=vqvae, 
                image_height=args.height, 
                image_width=args.width
            )
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            time_list.append(elapsed_time)
            
            if is_main_process():
                print(f"Time: {elapsed_time:.2f}s")
                print("-" * 50)
            
            # each rank's own JSON
            result = {
                "prompt": prompt_text,
                "elapsed_time": elapsed_time,
                "rank": rank
            }
            all_res.append(result)
            with open(per_rank_json, "w", encoding="utf-8") as f:
                json.dump(all_res, f, ensure_ascii=False, indent=2)
    
    barrier()
    
    # Merge JSONL (only on rank0)
    if is_main_process():
        merged = []
        
        # Collect all per-rank files
        world = dist.get_world_size()
        for r in range(world):
            fpath = json_dir / f"meta_rank{r}.json"
            if not os.path.exists(fpath):
                continue
            with open(fpath, "r", encoding="utf-8") as f:
                rank_res = json.load(f)
                merged.extend(rank_res)
        
        # Save merged json (list)
        all_json_path = json_dir / "meta.json"
        with open(all_json_path, "w", encoding="utf-8", buffering=1) as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        
        print(f"Average sampling time: {sum(time_list)/len(prompts):.2f}s")
    
    barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()