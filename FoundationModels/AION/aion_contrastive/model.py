"""Contrastive alignment of AION image and spectrum embeddings.

AION gives us, for every galaxy, a *sequence* of tokens per modality
(576 tokens for the image, 273 for the spectrum). To compare modalities we
need one vector per galaxy per modality. The module below learns exactly
that reduction:

                       learned queries (per modality)
                                  |
        AION tokens ──> [ cross-attention pooling ] ──> vector in R^1024

A small set of *learned query vectors* cross-attends to the token sequence:
the queries decide, via attention, which tokens matter and summarise them.
The attention weights themselves are shared between modalities — only the
queries differ — so the same module can pool either an image or a spectrum.

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


class CrossAttentionPool(nn.Module):
    """Pools a token sequence [B, L, D] into a single vector [B, output_dim]
    using a set of learned queries passed in by the caller.

    This is one pre-LayerNorm transformer block where the queries attend to
    the tokens (cross-attention) instead of to themselves, followed by a
    linear head to the shared embedding space.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        output_dim: int = 1024,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.norm_query = nn.LayerNorm(embed_dim)
        self.norm_tokens = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        hidden = int(embed_dim * mlp_ratio)
        self.norm_mlp = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.GELU(), nn.Linear(hidden, embed_dim)
        )
        self.head = nn.Linear(embed_dim, output_dim)

    def forward(self, tokens: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        # tokens : [B, L, D] AION token sequence (any L — that is the point!)
        # queries: [Q, D] learned, modality-specific
        queries = queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)  # [B, Q, D]

        # Queries attend to the tokens, with residual connections as in a
        # standard transformer block.
        attended, _ = self.attention(
            self.norm_query(queries), self.norm_tokens(tokens), self.norm_tokens(tokens)
        )
        x = queries + attended
        x = x + self.mlp(self.norm_mlp(x))

        # Average the Q query outputs into one summary vector, then project
        # into the shared contrastive space.
        return self.head(x.mean(dim=1))


def clip_loss(z_image: torch.Tensor, z_spectrum: torch.Tensor, logit_scale: torch.Tensor):
    """Symmetric InfoNCE loss (CLIP). Embeddings must be L2-normalised."""
    logits = logit_scale * z_image @ z_spectrum.T  # [N, N] cosine similarities
    labels = torch.arange(len(logits), device=logits.device)
    # Row i of `logits` are the similarities of image i to every spectrum:
    # cross-entropy pushes the diagonal (the true pairing) to dominate.
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


class ContrastiveAlignment(LightningModule):
    """LightningModule tying everything together.

    Args:
        embed_dim: dimension of the AION tokens (768 for AION-base).
        num_heads: attention heads in the pooling block.
        output_dim: dimension of the shared contrastive space.
        num_queries: how many learned query vectors per modality.
        lr: peak learning rate (AdamW, cosine-annealed).
        weight_decay: AdamW weight decay.
        init_temperature: initial softmax temperature of the CLIP loss; it is
            learned during training (as in CLIP) and logged as a diagnostic.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        output_dim: int = 1024,
        num_queries: int = 8,
        lr: float = 3e-4,
        weight_decay: float = 0.05,
        init_temperature: float = 0.07,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.pool = CrossAttentionPool(embed_dim, num_heads, output_dim)
        # One set of learned queries per modality — this is the only
        # modality-specific part of the model.
        self.query_image = nn.Parameter(0.02 * torch.randn(num_queries, embed_dim))
        self.query_spectrum = nn.Parameter(0.02 * torch.randn(num_queries, embed_dim))
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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
