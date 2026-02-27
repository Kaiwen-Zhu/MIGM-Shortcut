# -*- coding: utf-8 -*-
"""
Image processing utilities
"""
import torch
import PIL
import random
from PIL import Image
from diffusers import VQModel
from diffusers.image_processor import VaeImageProcessor

def decode_vq_to_image(
    vq_codes: torch.LongTensor, 
    image_height: int, 
    image_width: int,
    vae_ckpt: str = None, 
    save_path: str = None,
    vqvae: VQModel = None
) -> Image.Image:
    """
    Decode VQ codes to image
    
    Args:
        vq_codes: VQ codes
        save_path: Save path
        vae_ckpt: VAE checkpoint path
        image_height: Image height
        image_width: Image width
        vqvae: VQ-VAE model, if None will load from vae_ckpt
    
    Returns:
        PIL image
    """
    device = vq_codes.device
    if vqvae is None:
        vqvae = VQModel.from_pretrained(vae_ckpt, subfolder="vqvae").to(device)
    
    if isinstance(vqvae, VQModel):
        scale = 2 ** (len(vqvae.config.block_out_channels) - 1)
    else:
        scale = 2 ** (len(vqvae.module.config.block_out_channels) - 1)
    img_proc = VaeImageProcessor(vae_scale_factor=scale, do_normalize=False)

    # Calculate latent space grid size
    latent_height = image_height // scale
    latent_width = image_width // scale

    # Ensure VQ codes length matches
    if vq_codes.shape[1] != latent_height * latent_width:
        raise ValueError(
            f"VQ codes length mismatch: {vq_codes.shape[1]} != {latent_height * latent_width} "
            f"for image size ({image_height},{image_width}) with scale {scale}"
        )

    latents = (vq_codes.view(1, latent_height, latent_width) - 126356).long()
    
    if isinstance(vqvae, VQModel):
        recon = vqvae.decode(
            latents,
            force_not_quantize=True,
            shape=(1, latent_height, latent_width, vqvae.config.latent_channels),
        ).sample.clip(0, 1)
    else:  # DDP
        recon = vqvae.module.decode(
            latents,
            force_not_quantize=True,
            shape=(1, latent_height, latent_width, vqvae.module.config.latent_channels),
        ).sample.clip(0, 1)

    img = img_proc.postprocess(recon.detach(), output_type="pil")[0]
    if save_path is not None:
        img.save(save_path)
        print(f"[✓] Saved {save_path}")
    return img


def preprocess_image(image_path: str, target_size: tuple = (512, 512)):
    """
    Preprocess image: load, crop, resize
    
    Args:
        image_path: Image path
        target_size: Target size (width, height)
    
    Returns:
        Processed PIL image
    """
    img = Image.open(image_path).convert("RGB")
    crop_size_list = generate_crop_size_list((target_size[0] // 32) ** 2, 32)
    processed_img = var_center_crop(img, crop_size_list=crop_size_list)
    return processed_img


def calculate_vq_params(image_height: int, image_width: int, vae_scale: int = 16):
    """
    Calculate VQ related parameters
    
    Args:
        image_height: Image height
        image_width: Image width
        vae_scale: VAE scale factor
    
    Returns:
        seq_len, newline_every, token_grid_height, token_grid_width
    """
    token_grid_height = image_height // vae_scale
    token_grid_width = image_width // vae_scale
    seq_len = token_grid_height * token_grid_width
    newline_every = token_grid_width
    return seq_len, newline_every, token_grid_height, token_grid_width

def center_crop(pil_image, crop_size):
    while pil_image.size[0] >= 2 * crop_size[0] and pil_image.size[1] >= 2 * crop_size[1]:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = max(crop_size[0] / pil_image.size[0], crop_size[1] / pil_image.size[1])
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    crop_left = random.randint(0, pil_image.size[0] - crop_size[0])
    crop_upper = random.randint(0, pil_image.size[1] - crop_size[1])
    crop_right = crop_left + crop_size[0]
    crop_lower = crop_upper + crop_size[1]
    return pil_image.crop(box=(crop_left, crop_upper, crop_right, crop_lower))


def var_center_crop(pil_image, crop_size_list, random_top_k=1):
    w, h = pil_image.size
    rem_percent = [min(cw / w, ch / h) / max(cw / w, ch / h) for cw, ch in crop_size_list]
    crop_size = random.choice(
        sorted(((x, y) for x, y in zip(rem_percent, crop_size_list)), reverse=True)[:random_top_k]
    )[1]
    return center_crop(pil_image, crop_size)


def generate_crop_size_list(num_patches, patch_size, max_ratio=4.0):
    assert max_ratio >= 1.0
    crop_size_list = []
    wp, hp = num_patches, 1
    while wp > 0:
        if max(wp, hp) / min(wp, hp) <= max_ratio:
            crop_size_list.append((wp * patch_size, hp * patch_size))
        if (hp + 1) * wp <= num_patches:
            hp += 1
        else:
            wp -= 1
    return crop_size_list

def add_break_line(sequence: list, H: int, W: int, new_number: int = 0) -> list:
    """Add newline characters to sequence"""
    result = []
    for i in range(H):
        start = i * W
        end = start + W
        row = sequence[start:end]
        result.extend(row + [new_number])
    return result

def encode_img_with_breaks(img, vqvae, vae_scale_factor: int = 16):
    """Encode image and add newline characters"""
    from diffusers.image_processor import VaeImageProcessor
    
    orig = img.convert("RGB")
    orig_resized = orig
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_normalize=False)
    x = image_processor.preprocess(orig_resized).to(vqvae.device)
    latents = vqvae.encode(x).latents
    latents_bsz, channels, lat_h, lat_w = latents.shape
    quantized = vqvae.quantize(latents)[2][2] + 126356
    quantized = quantized.reshape(latents_bsz, lat_h, lat_w).flatten().tolist()
    img_token = add_break_line(quantized, lat_h, lat_w, new_number=126084)
    img_token = [126349] + img_token + [126350]
    return img_token
