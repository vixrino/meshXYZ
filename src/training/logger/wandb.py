from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WandbCfg:
    enabled: bool = False
    project: str = "mesh_genai"
    name: Optional[str] = None
    tags: list = field(default_factory=list)


def build_logger(cfg: WandbCfg, output_dir: str, hparams: dict):
    if not cfg.enabled:
        return True
    from lightning.pytorch.loggers import WandbLogger
    logger = WandbLogger(
        project=cfg.project,
        name=cfg.name,
        tags=cfg.tags,
        save_dir=output_dir,
    )
    logger.log_hyperparams(hparams)
    return logger
