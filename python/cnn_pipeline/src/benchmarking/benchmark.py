import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT_PATH = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_PATH))
from src.model.model import get_model
from src.pipeline.data_pipeline import ImageDataset, get_transforms, load_statistics
from src.utils.config import Config, load_config


def load_model(config: Config, model_path: Path) -> nn.Module:
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
    return model


def load_benchmark_set(path: Path):
    dataset = ImageDataset(root_dir=path)
    return dataset


def set_transform(
    statistics_path: Path,
    dataset,
    img_size: tuple,
    train: bool,
    three_gray_channels: bool,
):
    statistics = load_statistics(statistics_path)
    transform = get_transforms(
        statistics.get("mean"),
        statistics.get("std"),
        img_size=img_size,
        train=train,
        three_gray_channels=three_gray_channels,
    )
    dataset.transform = transform
    return dataset


if __name__ == "__main__":
    config = load_config("./configs/xray-config1.yaml")
    model = load_model(
        config=config,
        model_path=Path("./models/chest-xray-own-best_model_0.2604.pt").resolve(),
    )
    set = load_benchmark_set(
        Path("../../datasets/chest_xray_data/benchmarking").resolve()
    )
    set = set_transform(
        Path("./statistics.json"),
        dataset=set,
        img_size=config.data.img_size,
        train=False,
        three_gray_channels=config.data.three_gray_channels,
    )
    print(set[0])
