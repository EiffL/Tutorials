# Aligning AION image and spectrum embeddings with contrastive learning

This tutorial trains a *small cross-attention head* on top of frozen
[AION](https://github.com/PolymathicAI/AION) embeddings, so that the image
of a galaxy and its spectrum end up close together in a shared 768-d space.
It is the same idea as [AstroCLIP](https://arxiv.org/abs/2310.03024) — but
instead of training two large encoders, we reuse a multimodal foundation
model's embeddings and only learn a tiny pooling module (~10M parameters,
trains in minutes per epoch on a single consumer GPU).

It is also a compact, idiomatic example of a
[PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) (2.6+)
codebase: a `LightningDataModule`, a `LightningModule`, a YAML config, and a
`LightningCLI` entrypoint with wandb logging.

## 1. The data

`/local/home/flanusse/repo/FoundationModelTutorial/aion_embeddings/desi_x_ls_aion`
contains ~96,000 galaxies from the DESI×Legacy Surveys cross-match, stored as
477 parquet shards (~117 GB). Each row was produced by running the frozen
AION-base encoder on both modalities and keeping the **full token sequences**:

| column | shape | meaning |
|---|---|---|
| `embedding_image` | `[576, 768]` float16 | one token per Legacy Surveys image patch |
| `embedding_spectrum` | `[273, 768]` float16 | one token per DESI spectrum segment |
| `*_provabgs` | scalars | PROVABGS catalog properties (redshift, stellar mass, …) |

Because the dataset does not fit in memory, `aion_contrastive/data.py` loads
it with [HuggingFace `datasets`](https://huggingface.co/docs/datasets): on
first use the parquet shards are converted into the Arrow cache format (a
one-time cost — **the cache is as large as the dataset, ~117 GB, so make
sure the disk has room**; it lives in `~/.cache/huggingface/datasets` unless
you set `data.cache_dir`). After that, every row is a memory-mapped read and
`with_format("torch")` converts it to torch tensors on the fly — so we get a
fully shuffled map-style dataset without ever holding 117 GB in RAM. The
last `num_val_files` shards are held out for validation.

## 2. The model

AION gives us a *sequence* of tokens per galaxy per modality; contrastive
learning needs *one vector* per galaxy per modality. The reduction is learned
by a single cross-attention block (`CrossAttentionPool` in
`aion_contrastive/model.py`):

```
learned queries [Q, 768]  ──┐
                            ├──> [depth-1 self-attn blocks over tokens]
AION tokens   [L, 768]    ──┘    ──> cross-attention + MLP ──> mean over Q
                                                             ──> Linear ──> R^768
```

The pooler follows the **attentive pooler of V-JEPA 2**
([facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2),
`src/models/attentive_pooler.py`), whose details are worth reading closely:

- Only the *context tokens* are LayerNormed before attention — the learned
  queries enter raw (they are free parameters, so a norm on them would be
  redundant), and the cross-attention has no output projection.
- Optionally, `depth - 1` self-attention blocks process the *token sequence*
  before pooling (V-JEPA2 probes use up to 4; we default to 1, i.e. pooling
  only).
- Initialization: truncated normal (std 0.02) everywhere, zero-init queries,
  and the last linear of every residual branch rescaled by
  `1/sqrt(2 * layer_id)`.
- The optimization recipe also follows their probe evals: AdamW with a
  *small* weight decay (~2e-3, not the 0.05 typical of full transformers),
  linear warmup from a low start LR, cosine decay to 0.

Two design choices of our own worth noticing:

- **The pooling block is shared between modalities.** It never needs to know
  whether it is looking at 576 image tokens or 273 spectrum tokens — attention
  is naturally length-agnostic.
- **Only the queries are modality-specific.** `query_image` and
  `query_spectrum` are two small learned matrices `[num_queries, 768]`; they
  are what "asks" the token sequence for the information relevant to each
  modality. This is the perceiver/attention-pooling pattern used in CLIP,
  Flamingo, and AION itself.

### What exactly is in the stored tokens?

A subtle but important detail: the embeddings were generated with `AION.encode()`, which does
**not** return the raw encoder output. Mirroring how AION's own decoder consumes the encoder,
it returns `decoder_proj_context(encoder_output) + pos_emb + mod_emb` — the encoder output
passed through a linear projection, with the fixed sin-cos **positional embeddings** (2D over
the 24×24 patch grid for images, a degenerate 273×1 grid for spectra) and the learned
**modality embedding** added back in. You can verify this from the data alone: averaging the
tokens over many galaxies washes out the galaxy-specific content but not the additive
constants, and the per-position mean correlates strongly (r ≈ 0.5) with the sin-cos pattern.

The practical consequence: our probe cross-attends to **exactly the same tensor AION's own
decoder cross-attends to**, positions included — so the learned queries can select tokens by
position (central patches, specific wavelength regions) and we do not need to add any
positional information ourselves.

### The contrastive (InfoNCE / CLIP) loss

Both pooled vectors are L2-normalised, so their dot product is a cosine
similarity. For a batch of $N$ galaxies we build the $N \times N$ similarity
matrix between all images and all spectra, and treat matching as an
$N$-way classification problem in both directions:

$$
\mathcal{L} = \tfrac{1}{2}\left[
\underbrace{-\tfrac{1}{N}\sum_i \log
  \frac{e^{z^{img}_i \cdot z^{spec}_i / \tau}}{\sum_j e^{z^{img}_i \cdot z^{spec}_j / \tau}}}_{\text{image} \to \text{spectrum}}
+ \underbrace{(\dots)}_{\text{spectrum} \to \text{image}}
\right]
$$

The temperature $\tau$ is learned (initialised at 0.07, as in CLIP). Two
useful intuitions:

- A model that has learned nothing gives $\mathcal{L} = \log N$
  (≈ 5.55 for a batch of 256) — we log this as `train/chance_loss`.
- Every other galaxy in the batch acts as a negative example, which is why
  contrastive learning likes **large batches**.

Nothing about galaxy physics is hand-coded: the only supervision is
*"this image and this spectrum belong to the same object"*, and the shared
structure (redshift, stellar populations, morphology…) emerges by itself.

## 3. The code

```
AION/
├── main.py                    # LightningCLI entrypoint — the whole "script"
├── config/
│   └── contrastive.yaml       # every knob lives here
├── aion_contrastive/
│   ├── data.py                # AIONEmbeddingDataModule (HF datasets over the parquet shards)
│   └── model.py               # CrossAttentionPool + ContrastiveAlignment
├── evaluate_probe.ipynb       # after training: k-NN physics benchmark vs mean pooling
├── requirements.txt
└── README.md
```

`main.py` is deliberately three lines: `LightningCLI` reads the `__init__`
signatures of `ContrastiveAlignment` and `AIONEmbeddingDataModule` and
auto-generates the config schema and command-line interface from them. If
you add an argument to a constructor, it appears in the config — no argparse
boilerplate ever.

## 4. Running it

```bash
cd FoundationModels/AION
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
source .venv/bin/activate

wandb login          # once, if you have not already

# Full training run
python main.py fit -c config/contrastive.yaml

# Quick debugging run on a subset of shards, without syncing to wandb
WANDB_MODE=offline python main.py fit -c config/contrastive.yaml \
    --data.max_files 30 --trainer.max_epochs 2

# Any config entry can be overridden from the command line
python main.py fit -c config/contrastive.yaml --model.lr 1e-4 --data.batch_size 128
```

## 5. What to look at on wandb

- **`train/loss` vs `train/chance_loss`** — the gap to $\log N$ is your
  signal that the model beats random matching.
- **`val/retrieval_*_top1` / `top10`** — cross-modal retrieval over the full
  validation set (~2,000 galaxies): given an image, does its own spectrum
  rank first (top-1) or in the top ten? This is the honest end-task metric;
  random chance for top-1 is ~0.05%.
- **`train/temperature`** — the learned $\tau$; it typically drops as the
  embeddings sharpen.
- **`val/similarity_matrix`** — the image×spectrum cosine similarity matrix;
  training progress appears as an emerging bright diagonal.
- **`val/embedding_pca`** — both modalities projected onto the same 2-d PCA
  plane, colored by redshift. Physical structure appearing here (a redshift
  gradient) is emergent: redshift was never used in training.

## 6. Things to try

1. **Batch size.** Halve and double `data.batch_size` — how do the retrieval
   metrics respond? Remember `chance_loss` moves too, so compare retrieval,
   not raw loss.
2. **Number of queries.** `model.num_queries: 1` makes this classic attention
   pooling; does more than one query help?
3. **Unshared pooling.** Give each modality its own `CrossAttentionPool`
   (roughly twice the parameters). Is sharing helping or hurting?
4. **Mean-pool baseline.** Replace the cross-attention with a simple mean
   over tokens followed by a linear layer. How much does attention actually
   buy you?
5. **Physics from embeddings.** Load the best checkpoint, embed the
   validation set, and fit a k-NN regressor from the *image* embedding to
   redshift or stellar mass — the classic AstroCLIP result: spectra teach the
   image encoder spectroscopy.
