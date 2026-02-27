import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import PreNorm, FeedForward, Attention
import math


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar mask ratio into vector representation.
    Adapted from https://github.com/facebookresearch/DiT/blob/main/models.py#L27-L64
    """
    def __init__(self, hidden_dim, frequency_embedding_size=256):
        super().__init__()
        mid_size = round(math.sqrt(hidden_dim * frequency_embedding_size))
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, mid_size, bias=True),
            nn.SiLU(),
            nn.Linear(mid_size, hidden_dim, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ShortcutAttnBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.):
        super().__init__()
        self.cross_attn = PreNorm(
            hidden_dim, 
            # nn.MultiheadAttention(
            #     embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, bias=True, dropout=dropout
            # )
            Attention(hidden_dim, num_heads, dropout=dropout)
        )
        self.cross_attn_ff = PreNorm(
            hidden_dim, 
            FeedForward(hidden_dim, hidden_dim * 4, dropout=dropout)
        )
        self.self_attn = PreNorm(
            hidden_dim, 
            # nn.MultiheadAttention(
            #     embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, bias=True, dropout=dropout
            # )
            Attention(hidden_dim, num_heads, dropout=dropout)
        )
        self.self_attn_ff = PreNorm(
            hidden_dim, 
            FeedForward(hidden_dim, hidden_dim * 4, dropout=dropout)
        )

    def forward(self, x, new_token):
        attn_value, _ = self.cross_attn(x, key=new_token, value=new_token)
        x = x + attn_value
        x = x + self.cross_attn_ff(x)

        attn_value, _ = self.self_attn(x, key=x, value=x)
        x = x + attn_value
        x = x + self.self_attn_ff(x)
        return x


class MaskGITShortcut(nn.Module):
    def __init__(self, hidden_dim: int, n_block: int = 1, weight_path=None, embed_t=True, bottleneck_ratio=2,
                 tok_emb: nn.Embedding = None, pos_emb: nn.Parameter = None,
                 num_heads=8, dropout=0.1):
        super().__init__()
        orig_hidden_dim = hidden_dim
        hidden_dim = round(hidden_dim / bottleneck_ratio)
        if bottleneck_ratio != 1:
            self.proj_in = nn.Linear(orig_hidden_dim, hidden_dim)
            self.proj_out = nn.Linear(hidden_dim, orig_hidden_dim)
            self.new_token_proj_in = nn.Linear(orig_hidden_dim, hidden_dim)
        else:
            self.proj_in = nn.Identity()
            self.proj_out = nn.Identity()
            self.new_token_proj_in = nn.Identity()

        self.blocks = nn.ModuleList([
            ShortcutAttnBlock(hidden_dim, num_heads, dropout) for _ in range(n_block)
        ])
        self.last_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12),
            # nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim, eps=1e-12),
        )

        if embed_t:
            self.time_embedder = TimestepEmbedder(hidden_dim=hidden_dim)
            self.adaLN_modulation = nn.Linear(hidden_dim, hidden_dim * 2)
        else:
            self.time_embedder = None
            self.adaLN_modulation = None

        assert tok_emb is not None, "tok_emb must be provided"
        assert pos_emb is not None, "pos_emb must be provided"
        self.register_buffer("wte", tok_emb.weight.detach().clone(), persistent=False)
        self.register_buffer("wpe", pos_emb.detach().clone(), persistent=False)

        if weight_path is not None:
            print(f"Load shortcut model weights from {weight_path}")
            state_dict = torch.load(weight_path, map_location='cpu')
            self.load_state_dict(state_dict)

    def forward(self, prev_feat: torch.Tensor, new_token_ids: torch.Tensor, new_token_map: torch.Tensor, 
                mask_ratio: torch.Tensor = None):
        B, T, C = prev_feat.shape
        newT = new_token_ids.shape[1]
        assert new_token_map.shape == (B, T), f"Expected new_token_map shape {(B, T)}, got {new_token_map.shape}"
        assert new_token_map.dtype == torch.bool, f"Expected new_token_map dtype torch.bool, got {new_token_map.dtype}"
        assert mask_ratio.shape == (B,), f"Expected mask_ratio shape {(B,)}, got {mask_ratio.shape}"
        assert torch.all(new_token_map.sum(dim=-1) == newT), f"Expected {new_token_map.sum(dim=-1)} new tokens, got {newT}"

        new_token = F.embedding(new_token_ids, self.wte)
        pos_emb = self.wpe.expand(B, -1, -1)
        pos_emb = pos_emb[new_token_map].reshape(B, newT, C)
        new_token = new_token + pos_emb

        new_token = self.new_token_proj_in(new_token)
        
        x = prev_feat
        x = self.proj_in(x)

        for block in self.blocks:
            x = block(x, new_token)
        x = self.last_layer(x)
        
        if self.time_embedder is not None:
            # mask_ratio: (B,)
            assert torch.all((mask_ratio >= 0) & (mask_ratio <= 1)), f"mask_ratio should be in [0, 1], got {mask_ratio}"
            mask_ratio = mask_ratio * 1000
            time_emb = self.time_embedder(mask_ratio)  # (B, C)
            scale, shift = self.adaLN_modulation(time_emb).unsqueeze(1).chunk(2, dim=-1)  # (B, 1, C)
            x = x * (1 + scale) + shift

        x = self.proj_out(x)
        x = x + prev_feat
        return x


if __name__ == "__main__":
    hidden_dim = 768
    bottleneck_ratio = 2
    tok_emb = nn.Embedding(1024+1+1000+1, hidden_dim)
    pos_emb = nn.init.trunc_normal_(nn.Parameter(torch.zeros(1, 1025, hidden_dim)), 0., 0.02)
    shortcut = MaskGITShortcut(hidden_dim=hidden_dim, n_block=2, embed_t=True, 
                               bottleneck_ratio=bottleneck_ratio,
                               tok_emb=tok_emb, pos_emb=pos_emb)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    shortcut = shortcut.to(device)
    num_params = sum(p.numel() for p in shortcut.parameters())
    for name, param in shortcut.named_parameters():
        print(f"{name}: {param.numel() / 1e6:.3f} M parameters")
    print(f"Number of shortcut model parameters: {num_params / 1e6:.3f} M")
    print(f"Data type of shortcut model: {next(shortcut.parameters()).dtype}")

    feat = torch.randn(10, 1025, hidden_dim).to(device)
    new_token_ids = torch.randint(0, 1024, (10, 100)).to(device)
    new_token_map = torch.zeros(10, 1025, dtype=torch.bool).to(device)
    new_token_map[:, -100:] = True
    mask_ratio = torch.rand(10).to(device)
    out = shortcut(feat, new_token_ids, new_token_map, mask_ratio)
    print(f"Input: {feat.shape=}, {new_token_ids.shape=}, {new_token_map.shape=}, {mask_ratio.shape=}")
    print(f"Output: {out.shape=}")
