"""Google Drive checkpoint sync callback for Colab training.

Colab sessions disconnect after ~12 hours.  This callback mirrors every
checkpoint that Lightning writes to a local output directory into a
corresponding subdirectory on Google Drive so that training can be resumed
after a reconnect without data loss.

Usage in the training launcher
------------------------------
from src.training.callbacks import DriveCheckpointCallback

trainer = Trainer(
    ...
    callbacks=[
        ModelCheckpoint(...),
        DriveCheckpointCallback(
            drive_dir="/content/drive/MyDrive/mesh_genai/runs",
            sync_every_n_steps=1000,
        ),
    ],
)

The Drive dir layout mirrors the local output_dir:
  /content/drive/MyDrive/mesh_genai/runs/<run_name>/
      step-1000.ckpt
      step-2000.ckpt
      last.ckpt
      hparams.yaml           ← synced from the logger
      metrics.csv            ← synced if present
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path

from lightning.pytorch import Callback, LightningModule, Trainer

log = logging.getLogger(__name__)

_DRIVE_MOUNT = "/content/drive/MyDrive"


class DriveCheckpointCallback(Callback):
    """Mirror local checkpoints to Google Drive every N training steps.

    Parameters
    ----------
    drive_dir:
        Absolute path under Google Drive root.  E.g.
        ``/content/drive/MyDrive/mesh_genai/runs``.
        If Drive is not mounted (not running in Colab) the callback logs a
        warning and becomes a no-op, so the same code runs locally without errors.
    sync_every_n_steps:
        How often to trigger a full sync of the checkpoint directory.
        Defaults to 1000.  Lightning's ModelCheckpoint fires ``save_every``
        (typically 10 000) — we sync more often so partial progress is safe.
    also_sync_logs:
        If True, copy ``*.csv`` and ``*.yaml`` files from the logger directory
        to Drive alongside checkpoints (useful for offline metric inspection).
    run_name:
        Sub-directory under ``drive_dir`` for this run.  Defaults to the
        Lightning logger name or the current timestamp.
    """

    def __init__(
        self,
        drive_dir: str = f"{_DRIVE_MOUNT}/mesh_genai/runs",
        sync_every_n_steps: int = 1000,
        also_sync_logs: bool = True,
        run_name: str | None = None,
    ):
        super().__init__()
        self.drive_dir          = drive_dir
        self.sync_every_n_steps = sync_every_n_steps
        self.also_sync_logs     = also_sync_logs
        self._run_name          = run_name
        self._drive_available   = False
        self._last_sync_step    = -1
        self._sync_times: list[float] = []

    # ── Lifecycle hooks ───────────────────────────────────────────────────────

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if not os.path.isdir(_DRIVE_MOUNT):
            log.warning(
                "Google Drive not mounted at %s — DriveCheckpointCallback is disabled. "
                "Mount Drive with: from google.colab import drive; drive.mount('/content/drive')",
                _DRIVE_MOUNT,
            )
            return

        self._drive_available = True
        run_name = self._run_name or self._infer_run_name(trainer)
        self._dst = Path(self.drive_dir) / run_name
        self._dst.mkdir(parents=True, exist_ok=True)
        log.info("DriveCheckpointCallback: syncing to %s every %d steps",
                 self._dst, self.sync_every_n_steps)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        if not self._drive_available:
            return
        step = trainer.global_step
        if step - self._last_sync_step >= self.sync_every_n_steps:
            self._sync(trainer)
            self._last_sync_step = step

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self._drive_available:
            log.info("Training ended — performing final Drive sync.")
            self._sync(trainer)

    def on_exception(self, trainer: Trainer, pl_module: LightningModule, exception: BaseException) -> None:
        if self._drive_available:
            log.warning("Exception caught — emergency Drive sync before teardown.")
            self._sync(trainer)

    # ── Sync logic ────────────────────────────────────────────────────────────

    def _sync(self, trainer: Trainer) -> None:
        """Copy all .ckpt files (and optionally logs) from output_dir to Drive."""
        t0     = time.perf_counter()
        src    = Path(trainer.default_root_dir)
        n_ckpt = 0
        n_log  = 0

        # Copy checkpoint files
        for ckpt in src.rglob("*.ckpt"):
            rel  = ckpt.relative_to(src)
            dst  = self._dst / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists() or ckpt.stat().st_mtime > dst.stat().st_mtime:
                shutil.copy2(ckpt, dst)
                n_ckpt += 1

        # Optionally copy logs
        if self.also_sync_logs:
            for pattern in ("*.csv", "*.yaml", "*.json"):
                for f in src.rglob(pattern):
                    rel = f.relative_to(src)
                    dst = self._dst / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dst)
                    n_log += 1

        elapsed = time.perf_counter() - t0
        self._sync_times.append(elapsed)
        log.info(
            "Drive sync @ step %d: %d checkpoint(s), %d log file(s) — %.1fs",
            trainer.global_step, n_ckpt, n_log, elapsed,
        )

    @staticmethod
    def _infer_run_name(trainer: Trainer) -> str:
        """Use wandb run name if available, otherwise fall back to timestamp."""
        try:
            from lightning.pytorch.loggers import WandbLogger
            for logger in trainer.loggers:
                if isinstance(logger, WandbLogger):
                    return logger.experiment.name or logger.experiment.id
        except Exception:
            pass
        from datetime import datetime
        return datetime.now().strftime("run_%Y%m%d_%H%M%S")

    @property
    def avg_sync_time_s(self) -> float:
        return sum(self._sync_times) / max(len(self._sync_times), 1)
