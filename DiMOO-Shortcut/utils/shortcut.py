import torch
import webdataset as wds
import glob
import argparse

from infer import main as infer_shortcut


H, W = 1024, 1024
vae_scale = 16
lat_h, lat_w = H // vae_scale, W // vae_scale
special = 4  # boa, boi, eoi, eoa
img_seq_len = lat_h * lat_w + lat_h + special

mask_token_id = 126336


def get_samples_factory(no_prompt):
    def get_samples(img_iter):
        for k, feat, generation_order, teacher_vq_ids in img_iter:
            for step in range(feat.shape[0]-1):
                prev_feat = feat[step]
                tgt_feat = feat[step+1]
                new_token_map = generation_order == step
                if not no_prompt and 'uncond' not in k:
                    new_token_map[:-img_seq_len] = True  # prompt tokens always visible
                new_token_ids = teacher_vq_ids[new_token_map]
                n_total = (generation_order != -1).sum().item()
                cur_token_ids = teacher_vq_ids.clone()
                cur_token_ids[generation_order > step] = mask_token_id
                if 'uncond' in k:
                    # When generating the samples, the prompt in uncondition branch is set to -100;
                    # manually recover them so that the tokens can be fed into the vanilla model.
                    # This is useful only in rollout.
                    cur_token_ids[:14] = torch.tensor([
                        126332, 36289, 289, 3972, 4631, 297, 268, 3019, 12166, 13, 126333, 126334, 126351, 126335], 
                        device=cur_token_ids.device, dtype=cur_token_ids.dtype)
                yield k, step, prev_feat, new_token_ids, new_token_map, n_total, tgt_feat, cur_token_ids
    return get_samples


def build_dataloader(data_path, no_prompt=False, train=True, collate=True, batch_size=None, 
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
        dataset = dataset.compose(get_samples_factory(no_prompt))
    ld = wds.WebLoader(dataset, batch_size=None, num_workers=8 if train and not overfit else 1, pin_memory=True, prefetch_factor=2)
    ld = ld.batched(batch_size)
    if train:
        if not overfit:
            ld = ld.shuffle(500 if collate else 1)
        ld = ld.with_length(length).with_epoch(length)
    return ld


def gen_images(args, ckpt_name, ckpt_path, save_dir, global_iter):
    infer_args = argparse.Namespace()
    infer_args.checkpoint = args.config_dir
    infer_args.vae_ckpt = args.config_dir
    infer_args.seed = 1
    infer_args.height, infer_args.width = args.height, args.width
    infer_args.timesteps = args.n_steps
    infer_args.cfg_scale = 4
    infer_args.temperature = 1
    infer_args.shortcut_path = ckpt_path
    infer_args.shortcut_n_block = args.n_block
    infer_args.no_prompt = args.no_prompt
    infer_args.no_embed_t = args.no_embed_t
    infer_args.bottleneck_ratio = args.bottleneck_ratio
    infer_args.no_ca = args.no_ca

    infer_args.prompts = [
        "A heroic figure stands tall amidst snow-capped peaks, wearing golden armor, holding a staff and orb, while winged creatures soar behind him; all beneath the magical glow of Aurora Borealis.",
        "One person writing code.",
        "A vibrant cosmic dance floor filled with colorful lights and energetic dancers.",

        "a photo of a bench",
        "a photo of a bicycle",
        "a photo of a clock",

        """A plush toy resembling a white dog with large ears and a pink bow tie sits in the center of a snowy landscape. The toy wears a pink and white hat and is surrounded by small pink heart-shaped objects on the snow. The word "Loveing" is written in the snow in front of the toy. The background features a vast expanse of snow with bare trees and a pale, overcast sky. The scene is serene and whimsical, with soft natural lighting and a pastel color palette.
        """,
        """A close-up of a woman's face, framed slightly off-center, showcases her attentive expression. Her head is tilted slightly right, allowing the light to highlight the contours of her cheekbones. Her eyes are wide open, looking past the camera, with well-groomed eyebrows arching gracefully. Her lips form a subtle, relaxed line. Her curly, auburn hair falls in loose tendrils framing her face, drawing focus to the clear texture of her skin under soft lighting.
        """,
        """Close-up photo of a gourmet dish featuring grilled chicken wraps on a white rectangular plate. The wraps are cut in half, revealing a filling of chicken, herbs, and red peppers, garnished with fresh parsley. Three small white bowls containing different sauces—mustard, red sauce, and a spicy red-brown sauce—are placed to the left of the plate. The background includes a blurred bowl of fries and a white cloth with red stripes. Red peppercorns and parsley leaves are scattered around the plate.""",

        "The image depicts a traditional Japanese torii gate, a symbol of Shinto shrines, standing prominently in the center of a paved pathway. The gate is constructed from dark wood with two large stone pillars supporting it, adorned with golden ornaments at the top. The background reveals lush greenery and trees, suggesting a serene, natural setting. The pathway leads into an open area where a few people can be seen walking, indicating the shrine's accessibility to visitors. The overall atmosphere is peaceful and inviting, typical of a Shinto sanctuary.",
        "The image depicts a picturesque coastal town under a clear blue sky. The foreground features calm turquoise waters with a weathered, rusted boat docked at the pier. Behind the water, a row of charming, multi-story buildings with traditional architecture lines the street. These buildings, painted in warm tones of yellow and beige, have pitched roofs and are adorned with shutters and balconies. The street is bustling with activity; people can be seen walking along the sidewalk, and cars are parked nearby. A red flag flutters atop one of the buildings, adding a vibrant touch to the scene. The overall atmosphere is serene and inviting, typical of a quaint seaside town.",
        "The image captures a serene moment of two people standing close together, gazing out through large floor-to-ceiling windows that offer a panoramic view of a tranquil body of water and distant land. The couple is silhouetted against the bright daylight, creating a peaceful and intimate atmosphere. The scene outside includes calm waters, a few boats, and lush greenery, suggesting a coastal or lakeside location. Inside, minimalistic furniture, including a chair on each side of the frame, complements the modern and open design of the space. The overall mood is one of quiet contemplation and connection with nature.",

        "A woman in elaborate armor and a red cloak stands against the backdrop of an autumnal medieval village, where falling leaves decorate the cobbled streets beneath a gloomy sky.",
        "A black & white photo shows a classic-style fountain in a city square, bearing the inscription 'NEVER AGAIN', while a modern car is submerged within its waters. Grand ornate buildings surround this somber scene.",
        "One person greeting pets.",

        "In an astral scene, a divine form radiates amidst swirling galaxies, while two figures observe from earthly terrain with a mystical panorama of mountains, ruins, and a reflective lake.",
        "The image showcases a breathtaking sunset at a beach. The sun is positioned near the horizon, casting a golden hue over the scene. Rays of sunlight pierce through the clouds, creating a dramatic display of light and shadow. The clouds are dense and fluffy, with some appearing dark and others illuminated by the sun's rays. The beach has gentle waves rolling in, and the wet sand reflects the sun's glow. In the distance, there's a silhouette of a city or town with some structures and possibly a bridge.",
        "The image depicts a grand scene set in what appears to be a cathedral or a large hall with arches. In the foreground, a group of women dressed in elaborate, fantasy-inspired gowns of various colors and designs are walking in a procession. These women have distinct elf-like features, such as pointed ears. They are adorned with intricate jewelry and accessories. The gowns are detailed with lace, embroidery, and ruffles. In the background, there's a crowd of onlookers, including men and women, who seem to be watching the procession with awe. Some of the onlookers are dressed in simple clothing, contrasting with the opulence of the women in the foreground. On the right side, there are three elderly men with long beards and hats, sitting and observing the scene. The overall atmosphere of the image is one of grandeur, fantasy, and reverence."
    ]

    infer_args.shortcut = True
    for budget in [14, 11, 9]:
        infer_args.budget = budget
        infer_args.output_dir = save_dir / f"{ckpt_name}_T{infer_args.timesteps}_budget{infer_args.budget}"
        infer_shortcut(infer_args)

    infer_args.timesteps = args.n_steps
    if not (save_dir / 'ref_vanilla').exists():
        infer_args.shortcut = False
        infer_args.output_dir = save_dir / 'ref_vanilla'
        infer_shortcut(infer_args)


def get_tgt_vq_ids(tgt_feat, ff_out, vocab_size, orig_gt_vq_ids):
    tgt_logits = feat2logits(tgt_feat, ff_out, vocab_size)
    tgt_vq_ids = torch.argmax(tgt_logits, dim=-1).view_as(orig_gt_vq_ids)
    tgt_vq_ids[orig_gt_vq_ids == -100] = -100
    return tgt_vq_ids.view(-1)


def feat2logits(feat, ff_out, vocab_size):
    return ff_out(feat)[..., -vocab_size:].view(-1, vocab_size)


def get_ema_avg_fn(decay):
    assert 0.0 < decay < 1.0, "Decay must be between 0 and 1"
    def ema_avg_fn(ema_param, new_param, num_averaged):
        return decay * ema_param + (1 - decay) * new_param
    return ema_avg_fn
