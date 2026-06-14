import os
import random

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers import WandbLogger

from ..dataset.types import Batch
from ..model.mesh_transformer import MeshTransformer, MeshTransformerCfg
from .config import TrainingCfg
from .loss import compute_metrics, decompose_loss, face_type_acc, reconstruction_loss, valid_row_mask
from .optimizer import build_optimizer
from .strategy import MaskingPipeline, OrderingPipeline, TargetBuilder
from ..utils.viz import save_generation_video, save_prediction_grid


class MeshTransformerModule(LightningModule):

    def __init__(self, model_cfg: MeshTransformerCfg, training_cfg: TrainingCfg):
        super().__init__()
        self.model_cfg      = model_cfg
        self.training_cfg   = training_cfg
        self.model          = MeshTransformer(model_cfg)
        self.masking        = MaskingPipeline.from_cfg(training_cfg.masking)
        self.target_builder = TargetBuilder.from_cfg(training_cfg.target_builder)
        self.ordering = OrderingPipeline.from_cfg(training_cfg.ordering)

    def setup(self, stage: str) -> None:
        if stage == "fit" and self.model_cfg.encoder.weights_path:
            ckpt = torch.load(self.model_cfg.encoder.weights_path, map_location="cpu", weights_only=False)
            self.model.encoder.load_state_dict(ckpt["model"])
        self.model.encoder.requires_grad_(False)

    def on_train_epoch_start(self) -> None:
        self.model.encoder.eval()

    def training_step(self, batch: Batch, batch_idx: int):
        batch      = self.ordering.apply_to_batch(batch)
        token_mask = self.masking.compute_mask(batch)
        use_edge_cond     = self.model_cfg.decoder.use_edge_cond
        targets, query_edges = self.target_builder.compute_targets(batch, token_mask, use_edge_cond=use_edge_cond)
        pc         = self._maybe_pc(batch)
        logits     = self.model(pc, batch["faces"], token_mask=token_mask, query_edges=query_edges)

        loss    = reconstruction_loss(logits, targets, faces=batch["faces"],
                                      eos_weight=self.training_cfg.eos_weight)
        metrics = compute_metrics(logits, targets, faces=batch["faces"])
        self.log("train/loss", loss, prog_bar=False, on_step=True, on_epoch=False)
        self.log_dict({f"train/{k}": v for k, v in metrics.items()}, prog_bar=False, on_step=True, on_epoch=False)

        # Phase-4 metrics: loss components + face-type accuracy
        decomposed    = decompose_loss(logits, targets, faces=batch["faces"])
        ftype_acc     = face_type_acc(logits, targets, faces=batch["faces"])
        self.log_dict({f"train/{k}": v for k, v in decomposed.items()}, prog_bar=False, on_step=True, on_epoch=False)
        self.log("train/face_type_acc", ftype_acc, prog_bar=False, on_step=True, on_epoch=False)

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        step = self.global_step
        if step == 0 or step % self.training_cfg.viz_every_n_steps != 0:
            return

        batch      = self.ordering.apply_to_batch(batch)
        token_mask = self.masking.compute_mask(batch)
        use_edge_cond     = self.model_cfg.decoder.use_edge_cond
        targets, query_edges = self.target_builder.compute_targets(batch, token_mask, use_edge_cond=use_edge_cond)
        pc         = self._maybe_pc(batch)
        with torch.no_grad():
            logits = self.model(pc, batch["faces"], token_mask=token_mask, query_edges=query_edges)

        wandb_run = self.logger.experiment if isinstance(self.logger, WandbLogger) else None
        out_dir   = None if wandb_run else os.path.join(self.trainer.log_dir, "train_images")
        save_prediction_grid(
            logits, batch["faces"], targets,
            valid_row_mask(batch["faces"]), token_mask,
            out_dir, step, wandb_run=wandb_run,
            query_edges=query_edges,
        )

        ctx = batch["faces"][:1, :self.training_cfg.gen_max_ctx]
        pc_gen = batch["pc"][:1] if random.random() < self.training_cfg.pc_cond_prob else None
        with torch.no_grad():
            gen_results, intermediates, eos_snapshots, _, boundary_snapshots, query_snapshots = self.model.generate(
                ctx,
                pc=pc_gen,
                max_steps=self.training_cfg.gen_max_steps,
                return_intermediates=True,
            )

        # Truncation watchdog: if upweighting EOS makes the model emit EOS too early,
        # the generated mesh collapses to a handful of faces.  Log the final face count
        # so the eos_weight side-effect is visible in Wandb alongside eos_acc.
        if gen_results:
            self.log("train/gen_final_faces", float(gen_results[0].shape[0]),
                     prog_bar=False, on_step=True, on_epoch=False)
        gen_out_dir = None if wandb_run else os.path.join(self.trainer.log_dir, "gen_videos")
        save_generation_video(
            intermediates[0], gen_out_dir, wandb_run=wandb_run,
            eos_snapshots=eos_snapshots[0],
            boundary_snapshots=boundary_snapshots[0],
            query_snapshots=query_snapshots[0],
        )

    def configure_optimizers(self):
        return build_optimizer(self.model, self.training_cfg)

    def _maybe_pc(self, batch: Batch):
        if random.random() < self.training_cfg.pc_cond_prob:
            return batch["pc"]
        return None
