# MaskGIT-Shortcut
For the implementation of MaskGIT, we adopt [valeoai/MaskGIT PyTorch](https://github.com/valeoai/Halton-MaskGIT/tree/v1.0).


## Installation
Run `prepare_weights.py` to
+ download the pretrained MaskGIT model and MaskGIT-Shortcut model
+ extract the token embeddings, position embeddings, and the classification head weights from the pretrained MaskGIT


## Training
+ Curate the dataset
   ```bash
   bash makedata.sh
   ```
+ Train MaskGIT-Shortcut
   ```bash
   torchrun --nproc_per_node=4 train_shortcut.py \
      --exp_name 000 --bottleneck_ratio 2 --mse_weight 1 --kl_weight 0 \
      --ema_decay 0.9999 --lr 1e-4 --step_decay \
      --max_rollout_step 1 --val_max_rollout_step 1 
   ```


## Inference
+ Infer without shortcut (vanilla MaskGIT)
   ```bash
   python infer.py
   ```
+ Infer with shortcut
   ```bash
   python infer.py --shortcut --budget 12 --step 32 --shortcut_path weights/MaskGIT-Shortcut/maskgit-shortcut.pth
   ```


## Evaluation
```bash
bash eval.sh weights/MaskGIT-Shortcut/maskgit-shortcut.pth path/to/imagenet
```
Here, `path/to/imagenet` should be like
```
path/to/imagenet
└── val/
    ├── n01440764/
    ├── n01443537/
    ├── ...
```