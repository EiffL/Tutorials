"""LightningCLI entrypoint.

Everything — model hyperparameters, data settings, trainer options, the
wandb logger — is driven by the YAML config:

    python main.py fit -c config/contrastive.yaml

Any config value can be overridden on the command line, e.g.:

    python main.py fit -c config/contrastive.yaml --model.lr 1e-4 --data.batch_size 128
"""

from lightning.pytorch.cli import LightningCLI

from aion_contrastive import AIONEmbeddingDataModule, ContrastiveAlignment


def cli_main():
    # LightningCLI inspects the __init__ signatures of the model and the
    # datamodule and exposes every argument in the config/CLI. Note that we
    # never call trainer.fit() ourselves — the CLI subcommand does it.
    # (overwrite=True: wandb's save dir is not versioned per run, so the
    # saved config.yaml would otherwise block the second launch.)
    LightningCLI(
        ContrastiveAlignment,
        AIONEmbeddingDataModule,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    cli_main()
