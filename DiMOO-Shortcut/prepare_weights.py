from huggingface_hub import hf_hub_download

# Download DiMOO-Shortcut
hf_hub_download(repo_id="Kaiwen-Zhu/MIGM-Shortcut", filename="DiMOO-Shortcut/dimoo-shortcut.pth", local_dir="./weights")


# Extract weights
import torch
from model import LLaDAForMultiModalGeneration

model = LLaDAForMultiModalGeneration.from_pretrained(
    'weights/Lumina-DiMOO', torch_dtype=torch.bfloat16,
)

backbone = model.model.transformer
torch.save(backbone.wte.state_dict(), "weights/token_embedding.pt")
torch.save(backbone.ff_out.state_dict(), "weights/ff_out.pt")
