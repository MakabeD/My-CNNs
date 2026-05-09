import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from numpy import average
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT_PATH = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_PATH))
from src.model.model import get_model
from src.pipeline.data_pipeline import (
    ImageDataset,
    get_device,
    get_transforms,
    load_statistics,
)
from src.train.training import compute_metrics
from src.utils.config import Config, load_config

DEVICE = get_device()


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


def load_benchmark_set(path: Path) -> ImageDataset:
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


def set_to_DataLoader(set: ImageDataset, batch_size: int, num_workers) -> DataLoader:
    dataloader = DataLoader(
        set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataloader


def benchmarking_loop(dataloader: DataLoader, model: nn.Module):
    all_preds = []
    all_labels = []
    all_times = []
    pbar = tqdm(dataloader)
    model.eval().to(DEVICE)
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            _if_cuda()
            start_time = time.perf_counter()
            output = model(inputs)
            _if_cuda()
            end_time = time.perf_counter()
            all_times.append(end_time - start_time)
            pbar.set_postfix({"time": f"{end_time - start_time:.4f}s"})

            _, resoult = torch.max(output.data, 1)
            all_preds.extend(resoult.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
        print(len(dataloader))
        y_true = torch.tensor(all_labels)
        y_pred = torch.tensor(all_preds)

        average_time = sum(all_times) / len(dataloader)
        print(average_time)
        print(all_times)
        accuracy, precision, recall, f1_score = compute_metrics(y_true, y_pred)


def _if_cuda():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


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
    loader = set_to_DataLoader(set, config.data.batch_size, config.data.num_workers)
    benchmarking_loop(loader, model)
