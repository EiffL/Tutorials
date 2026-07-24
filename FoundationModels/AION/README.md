# Training a Multimodal Contrastive Embedding Model 

This tutorial trains a *small cross-attention head* on top of frozen
[AION](https://github.com/PolymathicAI/AION) embeddings, so that the image
of a galaxy and its spectrum end up close together in a shared 768-d space.
It is the same idea as [AstroCLIP](https://arxiv.org/abs/2310.03024) — but
instead of training two large encoders, we reuse a multimodal foundation
model's embeddings and only learn a small pooling module (~10M parameters,
trains in minutes per epoch on a single consumer GPU).


### Learning objectives

- **MLOps**: Practical experience training a foundation model with torch, Lightning, 
and Weights and Biases on a national supercomputer.
- **Contstrastive Learning**: [to complete]
- **Adapting Foundation Models for Downstream Tasks**: [to complete]
- **Probabilistic Inference from Foundation Model Outputs**: [to complete]


It is also a compact, idiomatic example of a
[PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) (2.6+)
codebase: a `LightningDataModule`, a `LightningModule`, a YAML config, and a
`LightningCLI` entrypoint with wandb logging.


## 1. Preparing the project 

Once you are logged in on Leonardo, you can run the following commands to prepare
the project:
```bash
git clone https://github.com/EiffL/Tutorials.git
module load python
cd Tutorials/FoundationModels/AION
python -m venv --system-site-packages .venv
.venv/bin/activate
pip install -r requirements.txt
```

Once this is done, you should have all the dependencies installed, and the project
is ready to run. To be able to see the results of your runs online, please also 
log into Weights and Biases:

```bash
wandb login
```

## 2. Understanding the model

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
`src/models/attentive_pooler.py`).

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

The temperature $\tau$ is learned (initialised at 0.07, as in CLIP). Every other galaxy in the batch acts as 
a negative example, which is why contrastive learning likes **large batches**.

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
