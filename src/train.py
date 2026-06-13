import argparse

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from .dataset.mesh_dataset import MeshDataModule
from .training.module import MeshTransformerModule
from .training.logger.wandb import build_logger
from .utils.cfg import load_cfg


def main(args: argparse.Namespace) -> None:
    model_cfg, data_cfg, training_cfg, wandb_cfg, raw_cfg = load_cfg(args.config)

    module = MeshTransformerModule(model_cfg, training_cfg)
    datamodule = MeshDataModule(data_cfg, batch_size=training_cfg.batch_size, train_dir=args.train_dir, val_dir=args.val_dir)

    logger = build_logger(wandb_cfg, output_dir=args.output_dir, hparams=raw_cfg)

    trainer = Trainer(
        strategy="ddp_find_unused_parameters_true" if torch.cuda.device_count() > 1 else "auto",
        max_steps=training_cfg.max_steps,
        precision="16-mixed" if training_cfg.mixed_precision else "32-true",
        gradient_clip_val=training_cfg.grad_clip,
        limit_val_batches=0,
        default_root_dir=args.output_dir,
        logger=logger,
        callbacks=[
            ModelCheckpoint(
                dirpath=args.output_dir,
                every_n_train_steps=training_cfg.save_every,
                save_last=True,
                filename="step-{step}",
            ),
        ],
    )
    trainer.fit(module, datamodule=datamodule, ckpt_path=args.ckpt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--val_dir", required=True)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--output_dir", default="runs")
    parser.add_argument("--ckpt", default=None, help="Resume from checkpoint (.ckpt)")
    main(parser.parse_args())
