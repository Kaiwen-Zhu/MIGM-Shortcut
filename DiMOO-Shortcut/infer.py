# -*- coding: utf-8 -*-
"""
Text-to-image inference script
"""
import os
import json
import argparse
import time
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer
from pathlib import Path

from config import SPECIAL_TOKENS
from model import LLaDAForMultiModalGeneration
from model.modeling_llada import create_model_config_from_pretrained_config
from model.shortcut_model import MDMShortcut

from utils.generation_utils import setup_seed
from utils.image_utils import decode_vq_to_image, calculate_vq_params, add_break_line
from generators.image_generation_generator import generate_image_shortcut
from utils.prompt_utils import generate_text_to_image_prompt, create_prompt_templates



def main(args):
    # Special tokens
    MASK = SPECIAL_TOKENS["mask_token"]
    NEW_LINE = SPECIAL_TOKENS["newline_token"]
    BOA = SPECIAL_TOKENS["answer_start"]  # Begin of Answer
    EOA = SPECIAL_TOKENS["answer_end"]    # End of Answer
    BOI = SPECIAL_TOKENS["boi"]           # Begin of Image
    EOI = SPECIAL_TOKENS["eoi"]           # End of Image

    # Set Random seed
    # if args.seed != 0:
    setup_seed(args.seed)
    
    # Create Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and tokenizer
    device = 'cuda:0'
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = LLaDAForMultiModalGeneration.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16,
        # device_map="cuda:0",
    ).to(device)

    if args.shortcut:
        cfg = AutoConfig.from_pretrained(args.checkpoint)
        cfg.embedding_size = cfg.vocab_size = 134548
        cfg = create_model_config_from_pretrained_config(cfg)
        cfg.init_device = 'cpu'
        shortcut = MDMShortcut(cfg, args.shortcut_n_block, weight_path=args.shortcut_path, 
                               embed_t=not args.no_embed_t, bottleneck_ratio=args.bottleneck_ratio,
                               no_ca=args.no_ca).cuda()
    else:
        shortcut = None
    
    # Load VQ-VAE
    from diffusers import VQModel
    vqvae = VQModel.from_pretrained(args.vae_ckpt, subfolder="vqvae").to(device)
    # Calculate VQ parameters
    seq_len, newline_every, token_grid_height, token_grid_width = calculate_vq_params(args.height, args.width)
    
    print(f"Generate image size: {args.height}x{args.width}")
    print(f"Calculated VQ sequence length: {seq_len}")
    print(f"Tokens per line (newline_every): {newline_every}")
    
    # Get prompt templates
    templates = create_prompt_templates()

    # Get prompt
    prompt_text_lst = args.prompts

    meta = []
    for i, prompt_text in enumerate(prompt_text_lst):
        if any(output_dir.glob(f"{i:02d}_*.png")):
            print(f"Image for prompt {i} already exists, skipping...")
            continue

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
        vq_tokens = generate_image_shortcut(
            model,
            prompt_ids,
            seq_len=seq_len,
            newline_every=newline_every,
            timesteps=args.timesteps,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
            uncon_ids=uncon_ids,
            code_start=code_start,
            use_shortcut=args.shortcut,
            budget=args.budget,
            shortcut_model=shortcut,
            no_prompt=args.no_prompt
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Generate filename
        words = prompt_text.split()
        filename_words = words[:10] if len(words) > 10 else words
        filename = "_".join(filename_words)
        filename = "".join(c for c in filename if c.isalnum() or c in ('_', '-'))
        filename = f"{i:02d}_{elapsed_time:.2f}_{filename}.png"
        save_path = output_dir / filename
        
        # Decode VQ codes to PNG and save
        img = decode_vq_to_image(
            vq_tokens, 
            save_path=save_path,
            vae_ckpt=args.vae_ckpt, 
            image_height=args.height, 
            image_width=args.width,
            vqvae=vqvae
        )
        
        if hasattr(args, 'meta_path') and args.meta_path is not None:
            meta.append({
                "image_path": filename,
                "prompt": prompt_text,
                "latency": elapsed_time,
            })

            with open(args.meta_path, "w") as f:
                json.dump(meta, f, indent=4)
        
        print(f"Time: {elapsed_time:.2f}s")
       

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Text-to-image inference")
    parser.add_argument("--checkpoint", type=str, default="weights/Lumina-DiMOO", help="Fine-tuned checkpoint path")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--timesteps", type=int, default=64, help="Number of timesteps")
    parser.add_argument("--cfg_scale", type=float, default=4.0, help="CFG scale")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--vae_ckpt", type=str, default="weights/Lumina-DiMOO", help="VAE checkpoint path")
    parser.add_argument("--output_dir", type=str, default="dev", help="Output directory")
    parser.add_argument("--prompts", type=list, default=["A photo of a bench."])
    parser.add_argument("--shortcut", action='store_true', help="Enable caching for faster inference")
    parser.add_argument("--budget", type=int, default=11)
    parser.add_argument("--shortcut_path", type=str, default='weights/DiMOO-Shortcut/dimoo-shortcut.pth')
    parser.add_argument("--shortcut_n_block", type=int, default=1)
    parser.add_argument("--no_prompt", action='store_true')
    parser.add_argument("--no_embed_t", action='store_true')
    parser.add_argument("--prompt_path", type=str, default=None)
    parser.add_argument("--bottleneck_ratio", type=float, default=2)
    parser.add_argument("--no_ca", action='store_true')
    
    args = parser.parse_args()

    args.prompts = [
        "A striking photograph of a glass of orange juice on a wooden kitchen table, capturing a playful moment. The orange juice splashes out of the glass and forms the word \"Smile\" in a whimsical, swirling script just above the glass. The background is softly blurred, revealing a cozy, homely kitchen with warm lighting and a sense of comfort.",
        """A plush toy resembling a white dog with large ears and a pink bow tie sits in the center of a snowy landscape. The toy wears a pink and white hat and is surrounded by small pink heart-shaped objects on the snow. The word "Loveing" is written in the snow in front of the toy. The background features a vast expanse of snow with bare trees and a pale, overcast sky. The scene is serene and whimsical, with soft natural lighting and a pastel color palette.
        """,
        """A close-up of a woman's face, framed slightly off-center, showcases her attentive expression. Her head is tilted slightly right, allowing the light to highlight the contours of her cheekbones. Her eyes are wide open, looking past the camera, with well-groomed eyebrows arching gracefully. Her lips form a subtle, relaxed line. Her curly, auburn hair falls in loose tendrils framing her face, drawing focus to the clear texture of her skin under soft lighting.
        """,
        """Close-up photo of a gourmet dish featuring grilled chicken wraps on a white rectangular plate. The wraps are cut in half, revealing a filling of chicken, herbs, and red peppers, garnished with fresh parsley. Three small white bowls containing different sauces—mustard, red sauce, and a spicy red-brown sauce—are placed to the left of the plate. The background includes a blurred bowl of fries and a white cloth with red stripes. Red peppercorns and parsley leaves are scattered around the plate.""",
        """A serene photograph of a young man standing on a sandy beach at dusk, facing the ocean. He is positioned in the lower center of the frame, wearing a grey sweater and black shorts. The horizon line divides the image, with gentle waves rolling in the midground and a calm sea extending to the right. The sky transitions from soft pink to pale blue, indicating sunset. A small strip of land is visible on the far right. The scene is tranquil and contemplative.
        """,
        """A whimsical scene featuring a plush toy bear wearing a blue sweater, positioned in the foreground, holding a butterfly on its raised arm. The bear is surrounded by a field of vibrant blue flowers, likely nemophila, creating a lush and colorful foreground. In the background, Mount Fuji rises majestically, its snow-capped peak sharply contrasting against a clear blue sky. The mountain is framed by fluffy white clouds and a line of dark green trees at its base. The butterfly, with its intricate black and orange wings, adds a touch of realism to the playful composition.
        """,
        """Modern living room with a minimalist design, featuring a beige sofa, facing a wooden table with a metal frame in the center. A TV is placed on a low wooden console with white drawers, beneath a framed artwork on the wall. A tripod floor lamp with a white shade stands next to the TV. Two large windows with white frames allow natural light to flood the room. A grey rug covers the wooden floor.
        """,
        """Vibrant autumn landscape photograph of a serene river winding through a forest. The river flows from the foreground to the background, reflecting the vivid colors of the surrounding trees. On the left bank, trees display a mix of deep reds and bright yellows, while the right bank is dominated by fiery reds and golden yellows. The background features a dense forest with a mix of evergreen and deciduous trees, their leaves in rich autumn hues. The sky is a clear blue with a few fluffy white clouds, adding contrast to the warm colors below. The foreground includes a few branches and shrubs with red and green foliage.
        """
    ]

    if args.prompt_path is not None:
        prompts = []
        args.meta_path = os.path.join(args.output_dir, 'metadata.json')
        args.output_dir = os.path.join(args.output_dir, 'images')
        if args.prompt_path.endswith(".jsonl"):
            with open(args.prompt_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        obj = json.loads(line)
                        prompt_text = obj.get("prompt", "").strip()
                        if prompt_text:
                            prompts.append(prompt_text)
        elif args.prompt_path.endswith(".json"):
            with open(args.prompt_path, "r", encoding="utf-8") as f:
                prompts = json.load(f)
        elif args.prompt_path.endswith(".txt"):
            with open(args.prompt_path, "r", encoding="utf-8") as f:
                raw = f.read()
            prompts = [line.strip() for line in raw.splitlines() if line.strip()]
        else:
            raise ValueError("Unsupported file format, please use .json/.jsonl/.txt")
        args.prompts = prompts

    main(args)
