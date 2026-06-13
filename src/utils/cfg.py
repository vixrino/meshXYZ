import dacite
import yaml

from ..dataset.mesh_dataset import DataCfg
from ..model.mesh_transformer import MeshTransformerCfg
from ..training.config import TrainingCfg
from ..training.logger.wandb import WandbCfg


def load_cfg(config_path: str) -> tuple[MeshTransformerCfg, DataCfg, TrainingCfg, WandbCfg, dict]:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    model_cfg = dacite.from_dict(MeshTransformerCfg, {"encoder": raw["encoder"], "decoder": raw["decoder"]})
    data_cfg = dacite.from_dict(DataCfg, raw.get("data", {}))
    training_cfg = dacite.from_dict(TrainingCfg, raw.get("training", {}))
    wandb_cfg = dacite.from_dict(WandbCfg, raw.get("wandb", {}))
    return model_cfg, data_cfg, training_cfg, wandb_cfg, raw
