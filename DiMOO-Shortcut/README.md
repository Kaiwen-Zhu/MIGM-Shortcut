# DiMOO-Shortcut

## Installation
+ Download the pretrained Lumina-DiMOO model
    ```bash
    huggingface-cli download Alpha-VLLM/Lumina-DiMOO --local-dir weights/Lumina-DiMOO
    ```
+ Download the pretrained DiMOO-Shortcut model
    ```bash
    huggingface-cli download Kaiwen-Zhu/MIGM-Shortcut DiMOO-Shortcut/dimoo-shortcut.pth --local-dir weights/
    ```
+ Run `prepare_weights.py` to extract the token embeddings and the classification head weights from the pretrained Lumina-DiMOO


## Training
+ Curate the dataset
   ```bash
   bash makedata.sh
   ```
+ Train DiMOO-Shortcut
   ```bash
   torchrun --nproc_per_node=4 train_shortcut.py \
      --exp_name 000 --bottleneck_ratio 2 --mse_weight 1 --kl_weight 0 \
      --ema_decay 0.9999 --lr 1e-4 --step_decay \
      --max_rollout_step 1 --val_max_rollout_step 1 
   ```

## Inference
+ Infer without shortcut (vanilla Lumina-DiMOO)
   ```bash
   python infer.py
   ```
+ Infer with shortcut
   ```bash
   python infer.py --shortcut --budget 11 --timesteps 64 --shortcut_path weights/DiMOO-Shortcut/dimoo-shortcut.pth
   ```
