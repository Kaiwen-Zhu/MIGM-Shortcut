import math
from typing import (
    Optional,
    Tuple
)

import torch
import torch.nn as nn

import copy

from .configuration_llada import ModelConfig

from .modeling_llada import LLaDALlamaBlock, BufferCache, LayerNorm, RotaryEmbedding


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar mask ratio into vector representation.
    Adapted from https://github.com/facebookresearch/DiT/blob/main/models.py#L27-L64
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        mid_size = round(math.sqrt(hidden_size * frequency_embedding_size))
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, mid_size, bias=True),
            nn.SiLU(),
            nn.Linear(mid_size, hidden_size, bias=True),
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
    def __init__(self, config: ModelConfig, cache, no_ca=False):
        super().__init__()
        if no_ca:
            self.cross_attn = LLaDALlamaBlock(layer_id=None, config=config, cache=cache)
        else:
            self.cross_attn = ShortcutCrossAttention(config=config, cache=cache)
        self.self_attn = LLaDALlamaBlock(layer_id=None, config=config, cache=cache)
        self.no_ca = no_ca
    
    def reset_parameters(self):
        self.cross_attn.reset_parameters()
        if hasattr(self.self_attn, 'reset_parameters'):
            self.self_attn.reset_parameters()

    def forward(self, x: torch.Tensor, new_token: torch.Tensor, new_token_map: torch.Tensor):
        if not self.no_ca:
            x, _ = self.cross_attn(x, new_token, new_token_map)
        else:
            x, _ = self.cross_attn(x)
        x, _ = self.self_attn(x)
        return x


class RotaryEmbedding4CrossAttn(RotaryEmbedding):
    def __init__(self, config: ModelConfig, cache: BufferCache):
        super().__init__(config=config, cache=cache)

    def forward(self, q: torch.Tensor, k: torch.Tensor, 
                new_token_map: torch.Tensor  # newly added
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.config.rope_full_precision:
            q_, k_ = q.float(), k.float()
        else:
            q_, k_ = q, k

        with torch.autocast(q.device.type, enabled=False):
            query_len = q_.shape[-2]
            pos_sin, pos_cos = self.get_rotary_embedding(query_len, q_.device)
            pos_sin = pos_sin.type_as(q_)
            pos_cos = pos_cos.type_as(q_)
            q_ = self.apply_rotary_pos_emb(pos_sin, pos_cos, q_)
            new_token_idx = new_token_map.nonzero(as_tuple=True)[1]  # get indices of new tokens
            k_ = self.apply_rotary_pos_emb(
                pos_sin[:, :, new_token_idx, :],  # only apply to new tokens
                pos_cos[:, :, new_token_idx, :], 
                k_)
        return q_.type_as(q), k_.type_as(k)

class ShortcutCrossAttention(LLaDALlamaBlock):
    def __init__(self, config: ModelConfig, cache):
        super().__init__(layer_id=None, config=config, cache=cache)
        self.__cache = cache
        if self.config.rope:
            self.rotary_emb = RotaryEmbedding4CrossAttn(config, self.__cache)  # newly added

    def forward(self, prev_feat: torch.Tensor, new_token: torch.Tensor, new_token_map: torch.Tensor, 
                attention_bias=None, layer_past=None, use_cache=False
                ):
        x_normed = self.attn_norm(prev_feat)
        new_token_normed = self.attn_norm(new_token)  # newly added
        q = self.q_proj(x_normed)
        k = self.k_proj(new_token_normed)  # newly added
        v = self.v_proj(new_token_normed)  # newly added

        # Get attention scores.
        if self._activation_checkpoint_fn is not None:
            att, cache = self._activation_checkpoint_fn(  # type: ignore
                self.attention, q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache, new_token_map=new_token_map
            )
        else:
            att, cache = self.attention(q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache,
                                         new_token_map=new_token_map)  # newly added

        # Add attention scores.
        # shape: (B, T, C)
        x = prev_feat + self.dropout(att)

        # Add feed-forward projection.
        # shape: (batch_size, seq_len, d_model)
        og_x = x
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.ff_norm, x)  # type: ignore
        else:
            x = self.ff_norm(x)
        x, x_up = self.ff_proj(x), self.up_proj(x) # new add
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.act, x)  # type: ignore
        else:
            x = self.act(x)
        x = x * x_up # new add
        x = self.ff_out(x)
        x = self.dropout(x)
        x = og_x + x

        return x, cache
    
    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        new_token_map = None  # newly added
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = q.size()  # batch size, sequence length, d_model
        new_T = k.size(1)  # new token sequence length  # newly added
        dtype = k.dtype

        # Optionally apply layer norm to keys and queries.
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q).to(dtype=dtype)
            k = self.k_norm(k).to(dtype=dtype)

        q = q.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        k = k.view(B, new_T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)
        v = v.view(B, new_T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)

        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        present = (k, v) if use_cache else None
        query_len, key_len = q.shape[-2], k.shape[-2]  # could be different if layer_past not None

        if self.config.rope:
            # Apply rotary embeddings.
            q, k = self.rotary_emb(q, k, new_token_map=new_token_map)  # newly added

        if attention_bias is not None:
            raise NotImplementedError("Not implemented attention bias for MDMShortcut")
            # Resize and cast attention bias.
            # The current dtype of the attention bias might not match the dtype that the SDP attn function will
            # run in if AMP is enabled, and this can be a problem if some tokens are masked out due to padding
            # as down-casting the attention bias to the autocast precision will result in -infs, which will
            # cause the SDP attn function to produce NaNs.
            attention_bias = self._cast_attn_bias(
                attention_bias[:, :, key_len - query_len : key_len, :key_len], dtype
            )

        # Get the attention scores.
        # shape: (B, nh, T, hs)
        att = self._scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_bias,
            dropout_p=0.0 if not self.training else self.config.attention_dropout,
            is_causal=False,
        )

        # Re-assemble all head outputs side-by-side.
        att = att.transpose(1, 2).contiguous().view(B, T, C)

        # Apply output projection.
        return self.attn_out(att), present


class MDMShortcut(nn.Module):
    def __init__(self, config: ModelConfig, n_block: int = 1, weight_path=None, 
                 embed_t=True, bottleneck_ratio=2, no_ca=False):
        super().__init__()
        self.config = config
        self.__cache = BufferCache()

        if bottleneck_ratio != 1:
            config = copy.copy(config)
            orig_d = config.d_model
            config.d_model = round(orig_d / bottleneck_ratio)
            self.proj_in = nn.Linear(orig_d, config.d_model)
            self.proj_out = nn.Linear(config.d_model, orig_d)
            if no_ca:
                self.new_token_proj_in = nn.Identity()
            else:
                self.new_token_proj_in = nn.Linear(orig_d, config.d_model)
        else:
            self.proj_in = nn.Identity()
            self.proj_out = nn.Identity()
            self.new_token_proj_in = nn.Identity()

        self.blocks = nn.ModuleList([
            ShortcutAttnBlock(config=config, cache=self.__cache, no_ca=no_ca) for _ in range(n_block)
        ])
        self.ln_f = LayerNorm.build(config)
        if embed_t:
            self.time_embedder = TimestepEmbedder(hidden_size=config.d_model)
            self.adaLN_modulation = nn.Linear(config.d_model, config.d_model * 2)
        else:
            self.time_embedder = None
            self.adaLN_modulation = None

        if weight_path is not None:
            print(f"Load shortcut model weights from {weight_path}")
            state_dict = torch.load(weight_path, map_location='cpu')
            self.load_state_dict(state_dict)
        else:
            self.reset_parameters()

    def reset_parameters(self):
        for block in self.blocks:
            block.reset_parameters()
        self.ln_f.reset_parameters()

    def forward(self, prev_feat: torch.Tensor, new_token: torch.Tensor, new_token_map: torch.Tensor, mask_ratio: torch.Tensor = None):
        B, T, C = prev_feat.shape
        newT = new_token.shape[1]
        assert new_token_map.shape == (B, T), f"Expected new_token_map shape {(B, T)}, got {new_token_map.shape}"
        assert torch.all(new_token_map.sum(dim=-1) == newT), f"Expected {new_token_map.sum(dim=-1).item()} new tokens, got {newT}"

        new_token = self.new_token_proj_in(new_token)
        
        x = prev_feat
        x = self.proj_in(x)

        for block in self.blocks:
            x = block(x, new_token, new_token_map)
        x = self.ln_f(x)
        
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
    from transformers import AutoConfig
    from model.modeling_llada import create_model_config_from_pretrained_config
    cfg = AutoConfig.from_pretrained('weights/Lumina-DiMOO')
    cfg = create_model_config_from_pretrained_config(cfg)
    cfg.init_device = 'cpu'
    device = torch.device('cuda')
    shortcut = MDMShortcut(cfg, 1, bottleneck_ratio=2)
    shortcut = shortcut.to(device)

    num_params = sum(p.numel() for p in shortcut.parameters())
    for name, param in shortcut.named_parameters():
        print(f"{name}: {param.numel() / 1e6:.3f} M parameters")
    print(f"Number of shortcut model parameters: {num_params / 1e6:.3f} M")
    print(f"Data type of shortcut model: {next(shortcut.parameters()).dtype}")
