import sys
import time
from pathlib import Path
from typing import Dict, List

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
from src.utils.config import Config


def _get_device():
    if not hasattr(_get_device, "_cache"):
        _get_device._cache = get_device()
    return _get_device._cache


def load_model(config: Config, model_path: Path) -> nn.Module:
    model = get_model(
        model_name=config.model.name,
        num_classes=config.model.num_classes,
        pretrained=config.model.pretrained,
        dropout=config.model.dropout,
    )
    print(model_path)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    return model


def apply_transform(
    statistics_path: Path,
    dataset: ImageDataset,
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


def _warmup_model(model: nn.Module, dataloader: DataLoader, num_warmup: int = 10):
    device = _get_device()
    model.eval()
    warmup_iter = iter(dataloader)
    for _ in range(num_warmup):
        try:
            inputs, _ = next(warmup_iter)
        except StopIteration:
            warmup_iter = iter(dataloader)
            inputs, _ = next(warmup_iter)
        with torch.inference_mode():
            _ = model(inputs.to(device))
        if device == "cuda":
            torch.cuda.synchronize()


def _remove_outliers(times: List[float], percentile: float = 95) -> np.ndarray:
    times_array = np.array(times)
    threshold = np.percentile(times_array, percentile)
    return times_array[times_array <= threshold]


def compute_inference_stats(
    batch_times: List[float], batch_size: int
) -> Dict[str, float]:
    times_array = _remove_outliers(batch_times)

    return {
        "mean_time_per_batch": float(times_array.mean()),
        "std_time_per_batch": float(times_array.std()),
        "min_time_per_batch": float(times_array.min()),
        "max_time_per_batch": float(times_array.max()),
        "p50_time_per_batch": float(np.median(times_array)),
        "p95_time_per_batch": float(np.percentile(times_array, 95)),
        "throughput_batches_per_sec": float(1 / times_array.mean()),
        "throughput_images_per_sec": float(batch_size / times_array.mean()),
        "total_batches": len(batch_times),
        "valid_batches_after_outlier_removal": len(times_array),
        "outliers_removed": len(batch_times) - len(times_array),
    }


def benchmarking_loop(
    dataloader: DataLoader,
    model: nn.Module,
    num_warmup: int = 10,
    verbose: bool = True,
) -> Dict[str, any]:
    device = _get_device()
    all_preds: List[int] = []
    all_labels: List[int] = []
    batch_times: List[float] = []
    batch_size = dataloader.batch_size

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    if verbose:
        print(f"[1/3] Warming up model ({num_warmup} iterations)...")
    _warmup_model(model, dataloader, num_warmup)

    if verbose:
        print("[2/3] Running inference benchmark...")
    model.eval()

    pbar = tqdm(dataloader, desc="Benchmarking", disable=not verbose)
    with torch.inference_mode():
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            if device == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            output = model(inputs)
            if device == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            batch_times.append(end_time - start_time)
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

    inference_stats = compute_inference_stats(batch_times, batch_size)

    y_true = torch.tensor(all_labels)
    y_pred = torch.tensor(all_preds)
    accuracy, precision, recall, f1_score = compute_metrics(y_true, y_pred)

    results = {
        "inference": inference_stats,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "device": device,
    }

    if device == "cuda":
        results["gpu_memory"] = {
            "allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
            "reserved_mb": torch.cuda.max_memory_reserved() / (1024**2),
        }

    return results


def print_results(results: Dict, file=None):
    text = "\n" + "=" * 60
    text += "\nBENCHMARK RESULTS"
    text += "\n" + "=" * 60

    text += "\n\n--- Inference Performance ---"
    inf = results["inference"]
    text += f"\n  Mean time/batch:    {inf['mean_time_per_batch'] * 1000:.2f} ms"
    text += f"\n  Std dev:            {inf['std_time_per_batch'] * 1000:.2f} ms"
    text += (
        f"\n  Min/Max:            {inf['min_time_per_batch'] * 1000:.2f} /"
        f" {inf['max_time_per_batch'] * 1000:.2f} ms"
    )
    text += (
        f"\n  P50/P95:            {inf['p50_time_per_batch'] * 1000:.2f} /"
        f" {inf['p95_time_per_batch'] * 1000:.2f} ms"
    )
    text += f"\n  Throughput:         {inf['throughput_images_per_sec']:.1f} images/sec"
    text += (
        f"\n  Valid batches:      {inf['valid_batches_after_outlier_removal']}"
        f"/{inf['total_batches']} (removed {inf['outliers_removed']} outliers)"
    )

    text += "\n\n--- Classification Metrics ---"
    text += f"\n  Accuracy:    {results['accuracy']:.4f}"
    text += f"\n  Precision:   {results['precision']:.4f}"
    text += f"\n  Recall:      {results['recall']:.4f}"
    text += f"\n  F1 Score:    {results['f1_score']:.4f}"

    if "gpu_memory" in results:
        text += "\n\n--- GPU Memory ---"
        text += f"\n  Allocated:  {results['gpu_memory']['allocated_mb']:.1f} MB"
        text += f"\n  Reserved:   {results['gpu_memory']['reserved_mb']:.1f} MB"

    text += "\n\n--- Device ---"
    text += f"\n  {results['device']}"
    text += "\n" + "=" * 60 + "\n"

    print(text, file=file)


def mlflow_save_results(
    config: Config,
    results: dict,
    repo_owner: str = "MakabeD",
    repo_name: str = "My-CNNs",
):
    import dagshub
    import mlflow

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
        mlflow.log_metric("f1_score", results.get("f1_score", 999))
        mlflow.log_metric(
            "mean_time/batch", results["inference"]["mean_time_per_batch"]
        )



