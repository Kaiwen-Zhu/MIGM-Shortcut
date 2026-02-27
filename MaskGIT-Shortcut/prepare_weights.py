from huggingface_hub import hf_hub_download

# Download MaskGIT; script from https://github.com/valeoai/Halton-MaskGIT/blob/v1.0/download_models.py
hf_hub_download(repo_id="llvictorll/Maskgit-pytorch", filename="pretrained_maskgit/VQGAN/last.ckpt", local_dir="./weights")
hf_hub_download(repo_id="llvictorll/Maskgit-pytorch", filename="pretrained_maskgit/VQGAN/model.yaml", local_dir="./weights")
hf_hub_download(repo_id="llvictorll/Maskgit-pytorch", filename="pretrained_maskgit/MaskGIT/MaskGIT_ImageNet_512.pth", local_dir="./weights")

# Download MaskGIT-Shortcut
hf_hub_download(repo_id="Kaiwen-Zhu/MIGM-Shortcut", filename="MaskGIT-Shortcut/maskgit-shortcut.pth", local_dir="./weights")


# Extract weights
import torch
from Trainer.vit import MaskGIT
from shortcut_utils import InferArgs

maskgit = MaskGIT(InferArgs())

backbone = maskgit.vit
torch.save(backbone.tok_emb.cpu().state_dict(), 'weights/token_embedding.pt')
torch.save(backbone.pos_emb.cpu(), 'weights/position_embedding.pt')
torch.save({
    "weight": backbone.tok_emb.weight.T.cpu(),  # (768, 2026)
    "bias": backbone.bias.cpu()  # (1025, 2026)
}, 'weights/ff_out.pt')  # this is not for a regular Linear!!!!
