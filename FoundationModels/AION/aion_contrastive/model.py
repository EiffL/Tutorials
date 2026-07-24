"""Contrastive alignment of AION image and spectrum embeddings.

AION gives us, for every galaxy, a *sequence* of tokens per modality
(576 tokens for the image, 273 for the spectrum). To compare modalities we
need one vector per galaxy per modality. The module below learns exactly
that reduction:

                       learned queries (per modality)
                                  |
        AION tokens ──> [ attentive pooler (V-JEPA2) ] ──> vector in R^768

A small set of *learned query vectors* cross-attends to the token sequence:
the queries decide, via attention, which tokens matter and summarise them.
The pooler weights are shared between modalities — only the queries differ —
so the same module can pool either an image or a spectrum.

The pooler follows the attentive-pooler architecture of V-JEPA 2
(github.com/facebookresearch/vjepa2, ``src/models/attentive_pooler.py``):

  - optional ``depth - 1`` self-attention blocks over the *tokens* before
    pooling (their probes use up to 4);
  - a cross-attention block in which only the context tokens are
    LayerNormed — the queries are free parameters and enter raw — and the
    attention itself has no output projection;
  - truncated-normal init (std 0.02) everywhere, zero-init queries, and
    residual-branch weights rescaled by 1/sqrt(2 * layer_id).

The two resulting vectors are then aligned with the standard CLIP/InfoNCE
objective: in a batch of N galaxies, the image embedding of galaxy i must
be more similar to *its own* spectrum embedding than to the N-1 others (and
vice versa). Nothing about the physics is hand-coded — the shared structure
between images and spectra is discovered purely from co-occurrence.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
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
    to project the pooled vector into the shared contrastive space. There is
    a single dimension throughout: the AION token dimension.

    Args:
        embed_dim: dimension of the AION tokens (and of everything else).
        num_heads: attention heads (V-JEPA2 uses head_dim 64, i.e. 12 @ 768).
        mlp_ratio: MLP expansion ratio in the blocks.
        depth: total blocks; ``depth - 1`` self-attention blocks run over the
            tokens before the final cross-attention pooling (V-JEPA2 probes
            use up to 4).
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


def clip_loss(z_image: torch.Tensor, z_spectrum: torch.Tensor, logit_scale: torch.Tensor):
    """Symmetric InfoNCE loss (CLIP). Embeddings must be L2-normalised."""
    logits = logit_scale * z_image @ z_spectrum.T  # [N, N] cosine similarities
    labels = torch.arange(len(logits), device=logits.device)
    # Row i of `logits` are the similarities of image i to every spectrum:
    # cross-entropy pushes the diagonal (the true pairing) to dominate.
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


class ContrastiveAlignment(LightningModule):
    """LightningModule tying everything together.

    The optimization recipe follows the V-JEPA2 attentive-probe evals:
    AdamW with a small weight decay (their sweeps use 5e-4 to 8e-3, far
    below the 0.05 common for full transformers), linear warmup from a low
    start LR, and cosine decay to 0.

    Args:
        embed_dim: dimension of the AION tokens (768 for AION-base), used
            throughout — including for the shared contrastive space.
        num_heads: attention heads in the pooler.
        num_queries: how many learned query vectors per modality.
        depth: pooler depth (``depth - 1`` self-attention blocks + pooling).
        lr: peak learning rate.
        start_lr: warmup starting learning rate.
        warmup_epochs: linear warmup duration.
        weight_decay: AdamW weight decay.
        init_temperature: initial softmax temperature of the CLIP loss; it is
            learned during training (as in CLIP) and logged as a diagnostic.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_queries: int = 8,
        depth: int = 1,
        lr: float = 5e-4,
        start_lr: float = 2e-4,
        warmup_epochs: int = 1,
        weight_decay: float = 2e-3,
        init_temperature: float = 0.07,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.pool = CrossAttentionPool(embed_dim, num_heads, depth=depth)
        # One set of learned queries per modality — this is the only
        # modality-specific part of the model. V-JEPA2 inits them as
        # trunc_normal(0.02) around zero.
        self.query_image = nn.Parameter(torch.zeros(num_queries, embed_dim))
        self.query_spectrum = nn.Parameter(torch.zeros(num_queries, embed_dim))
        nn.init.trunc_normal_(self.query_image, std=0.02)
        nn.init.trunc_normal_(self.query_spectrum, std=0.02)
        # Learned (log) inverse temperature, clamped to at most 100 as in CLIP.
        self.log_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_temperature)))

    # ------------------------------------------------------------------ #
    # Embedding                                                          #
    # ------------------------------------------------------------------ #

    def embed_image(self, tokens: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.pool(tokens.float(), self.query_image), dim=-1)

    def embed_spectrum(self, tokens: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.pool(tokens.float(), self.query_spectrum), dim=-1)

    def forward(self, batch):
        return self.embed_image(batch["image"]), self.embed_spectrum(batch["spectrum"])

    # ------------------------------------------------------------------ #
    # Training                                                           #
    # ------------------------------------------------------------------ #

    def training_step(self, batch, batch_idx):
        z_image, z_spectrum = self(batch)
        scale = self.log_scale.clamp(max=math.log(100.0)).exp()
        loss = clip_loss(z_image, z_spectrum, scale)

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/temperature", 1.0 / scale)
        # A random-chance batch would give loss = log(batch_size): useful to
        # know whether we are doing better than chance.
        self.log("train/chance_loss", math.log(len(z_image)))
        return loss

    # ------------------------------------------------------------------ #
    # Validation: loss + retrieval metrics + embedding-space diagnostics #
    # ------------------------------------------------------------------ #

    def on_validation_epoch_start(self):
        self._val_outputs = []

    def validation_step(self, batch, batch_idx):
        z_image, z_spectrum = self(batch)
        scale = self.log_scale.clamp(max=math.log(100.0)).exp()
        loss = clip_loss(z_image, z_spectrum, scale)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)

        out = {"z_image": z_image, "z_spectrum": z_spectrum}
        for key in ("Z_HP_provabgs", "LOG_MSTAR_provabgs"):
            if key in batch:
                out[key] = batch[key]
        self._val_outputs.append({k: v.cpu() for k, v in out.items()})

    def on_validation_epoch_end(self):
        if not self._val_outputs:
            return
        z_image = torch.cat([o["z_image"] for o in self._val_outputs]).float()
        z_spectrum = torch.cat([o["z_spectrum"] for o in self._val_outputs]).float()

        # Cross-modal retrieval over the *full* validation set: given an
        # image, can we find its own spectrum among all N candidates?
        similarity = z_image @ z_spectrum.T  # [N, N]
        target = torch.arange(len(similarity))
        for k in (1, 10):
            img2spec = (similarity.topk(k, dim=1).indices == target[:, None]).any(1)
            spec2img = (similarity.topk(k, dim=0).indices == target[None, :]).any(0)
            self.log(f"val/retrieval_image_to_spectrum_top{k}", img2spec.float().mean())
            self.log(f"val/retrieval_spectrum_to_image_top{k}", spec2img.float().mean())

        self._log_wandb_figures(z_image, z_spectrum, similarity)
        self._val_outputs = []

    def _log_wandb_figures(self, z_image, z_spectrum, similarity):
        """Two qualitative wandb diagnostics: the similarity matrix (a bright
        diagonal = the model matches pairs) and a PCA of the embeddings
        colored by redshift (structure = physics is encoded)."""
        from lightning.pytorch.loggers import WandbLogger

        if not isinstance(self.logger, WandbLogger):
            return

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import wandb

        n = min(200, len(similarity))
        fig_sim, ax = plt.subplots(figsize=(5, 4.5))
        im = ax.imshow(similarity[:n, :n], cmap="magma")
        ax.set(xlabel="spectrum index", ylabel="image index", title="cosine similarity")
        fig_sim.colorbar(im, ax=ax)

        figures = {"val/similarity_matrix": wandb.Image(fig_sim)}

        redshift = torch.cat(
            [o["Z_HP_provabgs"] for o in self._val_outputs if "Z_HP_provabgs" in o]
        ) if any("Z_HP_provabgs" in o for o in self._val_outputs) else None
        if redshift is not None:
            # Shared PCA basis so both modalities land in the same 2D plane.
            joint = torch.cat([z_image, z_spectrum])
            _, _, basis = torch.pca_lowrank(joint, q=2)
            fig_pca, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
            for ax, z, name in zip(axes, (z_image, z_spectrum), ("images", "spectra")):
                xy = z @ basis
                sc = ax.scatter(xy[:, 0], xy[:, 1], c=redshift, s=3, cmap="viridis")
                ax.set_title(f"{name} (PCA of shared space)")
            fig_pca.colorbar(sc, ax=axes, label="redshift")
            figures["val/embedding_pca"] = wandb.Image(fig_pca)

        self.logger.experiment.log(figures)
        plt.close("all")

    # ------------------------------------------------------------------ #
    # Optimization                                                       #
    # ------------------------------------------------------------------ #

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        # V-JEPA2 probe schedule: linear warmup from start_lr, cosine to 0.
        warmup = self.hparams.warmup_epochs
        schedulers = [
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=self.hparams.start_lr / self.hparams.lr,
                total_iters=warmup,
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, self.trainer.max_epochs - warmup)
            ),
        ]
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers, milestones=[warmup]
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
