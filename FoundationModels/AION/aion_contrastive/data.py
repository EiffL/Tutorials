"""Data loading for pre-computed AION embeddings.

The dataset on disk is a collection of parquet shards (~117 GB total,
~96k galaxies). Every row holds the *full token sequences* produced by the
frozen AION encoder:

  - ``embedding_image``    : [576, 768]  (24x24 image patches)
  - ``embedding_spectrum`` : [273, 768]  (spectrum patches)

plus scalar galaxy properties from the PROVABGS catalog (redshift, stellar
mass, ...) that we carry along purely for diagnostics.

We let HuggingFace ``datasets`` do all the heavy lifting: on first use it
converts the parquet shards into its Arrow cache format (a one-time cost —
the cache is as large as the dataset, so make sure the disk has room), after
which every access is a memory-mapped read. ``with_format("torch")`` then
converts rows to torch tensors on the fly, so the dataset behaves exactly
like an in-memory map-style dataset without ever loading 117 GB into RAM.
"""

from __future__ import annotations

import glob
import os
from typing import Sequence

from datasets import load_dataset
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader


class AIONEmbeddingDataModule(LightningDataModule):
    """LightningDataModule over the AION embedding dataset.

    Args:
        data_dir: directory containing the ``*.parquet`` embedding shards.
        batch_size: contrastive learning benefits from large batches (more
            negatives per positive), so use the largest that fits in memory.
        num_workers: dataloader workers.
        num_val_files: the *last* N shards (in sorted order) are held out for
            validation, the rest are used for training. A file-level split
            keeps things simple and is perfectly adequate here.
        max_files: optionally cap the total number of shards — handy to make
            a quick debugging run on a small subset of the data.
        properties: scalar catalog columns to carry through to the batch for
            diagnostics (e.g. coloring embedding plots by redshift).
        cache_dir: where HuggingFace datasets stores its Arrow cache
            (defaults to ``~/.cache/huggingface/datasets``).
        num_proc: parallel processes for the one-time parquet -> Arrow
            conversion.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 256,
        num_workers: int = 4,
        num_val_files: int = 10,
        max_files: int | None = None,
        properties: Sequence[str] = ("Z_HP_provabgs", "LOG_MSTAR_provabgs"),
        cache_dir: str | None = None,
        num_proc: int = 8,
    ):
        super().__init__()
        self.save_hyperparameters()

    def _load(self):
        files = sorted(glob.glob(os.path.join(self.hparams.data_dir, "*.parquet")))
        if not files:
            raise FileNotFoundError(f"No parquet shards found in {self.hparams.data_dir}")
        if self.hparams.max_files is not None:
            files = files[: self.hparams.max_files]

        n_val = self.hparams.num_val_files
        dataset = load_dataset(
            "parquet",
            data_files={"train": files[:-n_val], "validation": files[-n_val:]},
            cache_dir=self.hparams.cache_dir,
            num_proc=self.hparams.num_proc,
        )
        # Only decode the columns we actually train on, as torch tensors.
        dataset = dataset.rename_columns(
            {"embedding_image": "image", "embedding_spectrum": "spectrum"}
        )
        return dataset.with_format(
            "torch", columns=["image", "spectrum", *self.hparams.properties]
        )

    def prepare_data(self):
        # Runs on a single process before setup(): triggers the one-time
        # parquet -> Arrow conversion so that workers only ever read the cache.
        self._load()

    def setup(self, stage: str | None = None):
        dataset = self._load()  # instant after prepare_data(): cache hit
        self.train_dataset = dataset["train"]
        self.val_dataset = dataset["validation"]

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
