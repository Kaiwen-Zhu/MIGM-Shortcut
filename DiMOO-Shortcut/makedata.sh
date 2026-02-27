set -ex

train_n_img=2000
val_n_img=50

torchrun --nproc_per_node=4 makedata.py \
    --prompt_path dataset/blip3o_prompts.txt \
    --checkpoint weights/Lumina-DiMOO \
    --seed 1 \
    --timesteps 64 \
    --height 1024 \
    --width 1024 \
    --img_save_dir dataset/train/img \
    --meta_save_dir dataset/train/meta \
    --feat_save_dir dataset/train/feat \
    --n_img ${train_n_img}

torchrun --nproc_per_node=4 makedata.py \
    --prompt_path dataset/blip3o_prompts.txt \
    --checkpoint weights/Lumina-DiMOO \
    --seed 1 \
    --timesteps 64 \
    --height 1024 \
    --width 1024 \
    --img_save_dir dataset/val/img \
    --meta_save_dir dataset/val/meta \
    --feat_save_dir dataset/val/feat \
    --n_img ${val_n_img} \
    --start_idx ${train_n_img}