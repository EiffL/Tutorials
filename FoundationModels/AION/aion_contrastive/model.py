"""Contrastive alignment of AION image and spectrum embeddings.

AION gives us, for every galaxy, a *sequence* of tokens per modality
(576 tokens for the image, 273 for the spectrum). Here we build a model
that pulls the information from this sequence into a single vector:

                       learned queries (per modality)
                                  |
        AION tokens ──> [ attentive pooler ] ──> vector in R^768

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
from .networks import CrossAttentionPool

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
        # modality-specific part of the model. 
        self.query_image = nn.Parameter(torch.zeros(num_queries, embed_dim))
        nn.init.trunc_normal_(self.query_image, std=0.02)
        self.query_spectrum = nn.Parameter(torch.zeros(num_queries, embed_dim))
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
        return loss

    # ------------------------------------------------------------------ #
    # Validation                                                         #
    # ------------------------------------------------------------------ #
    
    def validation_step(self, batch, batch_idx):
        z_image, z_spectrum = self(batch)
        scale = self.log_scale.clamp(max=math.log(100.0)).exp()
        loss = clip_loss(z_image, z_spectrum, scale)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    # ------------------------------------------------------------------ #
    # Optimization                                                       #
    # ------------------------------------------------------------------ #

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        # Probe schedule: linear warmup from start_lr, cosine to 0.
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
