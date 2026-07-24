from __future__ import annotations


import math

import torch
import torch.nn.functional as F
from torch import nn


class CrossAttention(nn.Module):
    """Multi-head cross-attention, V-JEPA2 style: separate q and kv linear
    projections, no output projection."""

    def __init__(self, dim: int, num_heads: int = 12, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, 2 * dim, bias=qkv_bias)

    def forward(self, q: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        B, n, C = q.shape
        h = self.num_heads
        q = self.q(q).reshape(B, n, h, C // h).transpose(1, 2)
        kv = self.kv(x).reshape(B, -1, 2, h, C // h).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, kv[0], kv[1])
        return out.transpose(1, 2).reshape(B, n, C)


class SelfAttentionBlock(nn.Module):
    """Standard pre-LN transformer block, used on the tokens before pooling
    when ``depth > 1``."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, qkv_bias: bool):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, bias=qkv_bias, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        x = x + self.attn(y, y, y, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class CrossAttentionPool(nn.Module):
    """V-JEPA2 attentive pooler, adapted to take the learned queries as an
    input (so two modalities can share the weights but not the queries) and
    to project the pooled vector into the shared contrastive space. 
    Args:
        embed_dim: dimension of the AION tokens (and of everything else).
        num_heads: attention heads (V-JEPA2 uses head_dim 64, i.e. 12 @ 768).
        mlp_ratio: MLP expansion ratio in the blocks.
        depth: total blocks; ``depth - 1`` self-attention blocks run over the
            tokens before the final cross-attention pooling.
        qkv_bias: bias in the attention projections.
        init_std: std of the truncated-normal weight init.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        depth: int = 1,
        qkv_bias: bool = True,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            SelfAttentionBlock(embed_dim, num_heads, mlp_ratio, qkv_bias)
            for _ in range(depth - 1)
        )

        # Cross-attention block: LayerNorm on the context only, residuals on
        # the query path (queries enter un-normalised — they are learned
        # parameters, free to adopt whatever scale is useful).
        self.norm_context = nn.LayerNorm(embed_dim)
        self.xattn = CrossAttention(embed_dim, num_heads, qkv_bias)
        self.norm_mlp = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, hidden), nn.GELU(), nn.Linear(hidden, embed_dim))

        self.head = nn.Linear(embed_dim, embed_dim)

        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        """V-JEPA2 depth-scaled init: divide the last linear of each residual
        branch by sqrt(2 * layer_id) so that the residual stream variance
        stays controlled as depth grows."""
        layer_id = 0
        for layer_id, block in enumerate(self.blocks, start=1):
            block.attn.out_proj.weight.data.div_(math.sqrt(2.0 * layer_id))
            block.mlp[-1].weight.data.div_(math.sqrt(2.0 * layer_id))
        self.mlp[-1].weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))

    def forward(self, tokens: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        # tokens : [B, L, D] AION token sequence (any L — that is the point!)
        # queries: [Q, D] learned, modality-specific
        for block in self.blocks:
            tokens = block(tokens)

        q = queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)  # [B, Q, D]
        q = q + self.xattn(q, self.norm_context(tokens))
        q = q + self.mlp(self.norm_mlp(q))

        # Average the Q query outputs into one summary vector, then project
        # into the shared contrastive space.
        return self.head(q.mean(dim=1))
