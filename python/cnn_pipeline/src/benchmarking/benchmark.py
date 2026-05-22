import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import dagshub
import mlflow
import numpy as np
import torch
import torch.nn as nn
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


def _if_cuda():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def _warmup_model(model: nn.Module, dataloader: DataLoader, num_warmup: int = 10):
    model.eval()
    warmup_iter = iter(dataloader)
    for _ in range(num_warmup):
        try:
            inputs, _ = next(warmup_iter)
        except StopIteration:
            warmup_iter = iter(dataloader)
            inputs, _ = next(warmup_iter)
        with torch.no_grad():
            _ = model(inputs.to(DEVICE))
        if DEVICE == "cuda":
            torch.cuda.synchronize()


def _remove_outliers(times: List[float], percentile: float = 95) -> np.ndarray:
    times_array = np.array(times)
    threshold = np.percentile(times_array, percentile)
    return times_array[times_array <= threshold]


def compute_inference_stats(
    all_times: List[float], batch_size: int
) -> Dict[str, float]:
    times_array = _remove_outliers(all_times)

    return {
        "mean_time_per_batch": float(times_array.mean()),
        "std_time_per_batch": float(times_array.std()),
        "min_time_per_batch": float(times_array.min()),
        "max_time_per_batch": float(times_array.max()),
        "p50_time_per_batch": float(np.median(times_array)),
        "p95_time_per_batch": float(np.percentile(times_array, 95)),
        "throughput_batches_per_sec": float(1 / times_array.mean()),
        "throughput_images_per_sec": float(batch_size / times_array.mean()),
        "total_batches": len(all_times),
        "valid_batches_after_outlier_removal": len(times_array),
        "outliers_removed": len(all_times) - len(times_array),
    }


def benchmarking_loop(
    dataloader: DataLoader,
    model: nn.Module,
    num_warmup: int = 10,
    verbose: bool = True,
) -> Dict[str, any]:
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_times: List[float] = []
    batch_size = dataloader.batch_size

    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    if verbose:
        print(f"[1/3] Warming up model ({num_warmup} iterations)...")
    _warmup_model(model, dataloader, num_warmup)

    if verbose:
        print("[2/3] Running inference benchmark...")
    model.eval()

    pbar = tqdm(dataloader, desc="Benchmarking", disable=not verbose)
    with torch.no_grad():
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            _if_cuda()
            start_time = time.perf_counter()
            output = model(inputs)
            _if_cuda()
            end_time = time.perf_counter()

            all_times.append(end_time - start_time)
            _, preds = torch.max(output, 1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            current_throughput = batch_size / (end_time - start_time)
            pbar.set_postfix(
                {
                    "time": f"{end_time - start_time:.4f}s",
                    "fps": f"{current_throughput:.1f}",
                }
            )

    if verbose:
        print("[3/3] Computing metrics...")

    inference_stats = compute_inference_stats(all_times, batch_size)

    y_true = torch.tensor(all_labels)
    y_pred = torch.tensor(all_preds)
    accuracy, precision, recall, f1_score = compute_metrics(y_true, y_pred)

    results = {
        "inference": inference_stats,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "device": DEVICE,
    }

    if DEVICE == "cuda":
        results["gpu_memory"] = {
            "allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
            "reserved_mb": torch.cuda.max_memory_reserved() / (1024**2),
        }

    return results


def print_results(results: Dict):
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    print("\n--- Inference Performance ---")
    inf = results["inference"]
    print(f"  Mean time/batch:    {inf['mean_time_per_batch'] * 1000:.2f} ms")
    print(f"  Std dev:            {inf['std_time_per_batch'] * 1000:.2f} ms")
    print(
        f"  Min/Max:            {inf['min_time_per_batch'] * 1000:.2f} / {inf['max_time_per_batch'] * 1000:.2f} ms"
    )
    print(
        f"  P50/P95:            {inf['p50_time_per_batch'] * 1000:.2f} / {inf['p95_time_per_batch'] * 1000:.2f} ms"
    )
    print(f"  Throughput:         {inf['throughput_images_per_sec']:.1f} images/sec")
    print(
        f"  Valid batches:      {inf['valid_batches_after_outlier_removal']}/{inf['total_batches']} (removed {inf['outliers_removed']} outliers)"
    )

    print("\n--- Classification Metrics ---")
    print(f"  Accuracy:    {results['accuracy']:.4f}")
    print(f"  Precision:   {results['precision']:.4f}")
    print(f"  Recall:      {results['recall']:.4f}")
    print(f"  F1 Score:    {results['f1_score']:.4f}")

    if "gpu_memory" in results:
        print("\n--- GPU Memory ---")
        print(f"  Allocated:  {results['gpu_memory']['allocated_mb']:.1f} MB")
        print(f"  Reserved:   {results['gpu_memory']['reserved_mb']:.1f} MB")

    print("\n--- Device ---")
    print(f"  {results['device']}")
    print("=" * 60 + "\n")


def mlflow_save_results(
    config: Config,
    results: dict,
    repo_owner: str = "MakabeD",
    repo_name: str = "My-CNNs",
):
    if mlflow.active_run():
        mlflow.end_run()
    dagshub.init(repo_name, repo_owner, mlflow=True)
    mlflow.set_experiment(f"benchmarking_{config.mlflow.experiment_name}")
    if config.training.run_name:
        mlflow.set_tag("mlflow.runName", config.training.run_name)
    if mlflow.active_run():
        mlflow.log_param("name", config.training.run_name)
        mlflow.log_metric("precision", results.get("precision", 999))
        mlflow.log_metric("accuracy", results.get("accuracy", 999))
        mlflow.log_metric("recall", results.get("recall", 999))
        mlflow.log_metric("f1-score", results.get("f1-score", 999))


if __name__ == "__main__":
    config = load_config("./configs/xray-config1.yaml")
    model = load_model(
        config=config,
        model_path=Path("./models/chest-xray-own-best_model_0.2604.pt").resolve(),
    )
    model.to(DEVICE)

    dataset = load_benchmark_set(
        Path("../../datasets/chest_xray_data/benchmarking").resolve()
    )
    dataset = set_transform(
        Path("./statistics.json"),
        dataset=dataset,
        img_size=config.data.img_size,
        train=False,
        three_gray_channels=config.data.three_gray_channels,
    )
    dataloader = set_to_DataLoader(
        dataset, config.data.batch_size, config.data.num_workers
    )

    results = benchmarking_loop(dataloader, model, num_warmup=10, verbose=True)
    print_results(results)
    mlflow_save_results(config, results)
