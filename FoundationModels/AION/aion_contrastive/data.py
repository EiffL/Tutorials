"""Data loading for pre-computed AION embeddings.

The dataset is `EiffL/desi_x_ls_aion` on the HuggingFace Hub (~117 GB,
~96k galaxies). Every row holds the *full token sequences* produced by the
frozen AION encoder:

  - ``embedding_image``    : [576, 768]  (24x24 image patches)
  - ``embedding_spectrum`` : [273, 768]  (spectrum patches)

plus scalar galaxy properties from the PROVABGS catalog (redshift, stellar
mass, ...) that we carry along purely for diagnostics.

We let HuggingFace ``datasets`` do all the heavy lifting: on first use the
parquet shards are fetched and converted into the Arrow cache format (a
one-time cost — make sure the disk has room), after which every access is a
memory-mapped read. ``with_format("torch")`` then converts rows to torch
tensors on the fly, so the dataset behaves exactly like an in-memory
map-style dataset without ever loading 117 GB into RAM.
"""

from __future__ import annotations

import glob
import os
from typing import Sequence

from datasets import load_dataset
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

SPLIT_SEED = 42  # fixed so that train/val is reproducible everywhere


class AIONEmbeddingDataModule(LightningDataModule):
    """LightningDataModule over the AION embedding dataset.

    Args:
        dataset: a HuggingFace Hub dataset id (default: the public
            ``EiffL/desi_x_ls_aion``), or a local directory of ``*.parquet``
            shards — handy to skip the download on a machine that already
            has the files.
        batch_size: contrastive learning benefits from large batches (more
            negatives per positive), so use the largest that fits in memory.
        num_workers: dataloader workers.
        val_fraction: fraction of galaxies held out for validation, split
            at the row level with a fixed seed — so training and evaluation
            agree on the split no matter where they run.
        properties: scalar catalog columns to carry through to the batch for
            diagnostics (e.g. coloring embedding plots by redshift).
        num_proc: parallel processes for the one-time Arrow conversion.
    """

    def __init__(
        self,
        dataset: str = "EiffL/desi_x_ls_aion",
        batch_size: int = 256,
        num_workers: int = 16,
        val_fraction: float = 0.02,
        properties: Sequence[str] = ("Z_HP_provabgs", "LOG_MSTAR_provabgs"),
        num_proc: int = 8,
    ):
        super().__init__()
        self.save_hyperparameters()

    def _load(self):
        source = self.hparams.dataset
        if os.path.isdir(source):
            files = sorted(glob.glob(os.path.join(source, "*.parquet")))
            dataset = load_dataset(
                "parquet", data_files=files, split="train", num_proc=self.hparams.num_proc
            )
        else:
            dataset = load_dataset(source, split="train", num_proc=self.hparams.num_proc)

        splits = dataset.train_test_split(
            test_size=self.hparams.val_fraction, seed=SPLIT_SEED
        )
        # Only decode the columns we actually train on, as torch tensors.
        splits = splits.rename_columns(
            {"embedding_image": "image", "embedding_spectrum": "spectrum"}
        )
        return splits.with_format(
            "torch", columns=["image", "spectrum", *self.hparams.properties]
        )

    def prepare_data(self):
        # Runs on a single process before setup(): triggers the one-time
        # download + Arrow conversion so that workers only ever read the cache.
        self._load()

    def setup(self, stage: str | None = None):
        splits = self._load()  # instant after prepare_data(): cache hit
        self.train_dataset = splits["train"]
        self.val_dataset = splits["test"]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
            drop_last=True,  # partial batches distort the contrastive loss scale
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=min(2, self.hparams.num_workers),
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
        )
