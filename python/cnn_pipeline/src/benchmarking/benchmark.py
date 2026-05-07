import sys
from pathlib import Path

import torch

ROOT_PATH = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_PATH))
from src.model.model import get_model
from src.utils.config import Config, load_config


def load_model(config: Config, model_path: Path):
    model_name = config.model.name
    num_classes = config.model.num_classes
    pretrained = config.model.pretrained
    dropout = config.model.dropout

    print(model_path)
    model = get_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
    )
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)


if __name__ == "__main__":
    load_model(
        config=load_config("./configs/xray-config1.yaml"),
        model_path=Path("./models/chest-xray-own-best_model_0.2604.pt").resolve(),
    )
