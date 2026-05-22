import argparse
import logging
import sys
from pathlib import Path

from torch.utils.data import DataLoader
from src.benchmarking.benchmark import (
    ImageDataset,
    _get_device,
    apply_transform,
    benchmarking_loop,
    load_model,
    mlflow_save_results,
    print_results,
)
from src.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark model inference performance and classification metrics"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to benchmark dataset directory",
    )
    parser.add_argument(
        "--statistics",
        type=str,
        default="statistics.json",
        help="Path to statistics JSON (default: statistics.json)",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Skip logging results to MLflow",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Show progress bars and detailed output (default: True)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress bars and detailed output",
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    resolved = Path(path_str).resolve()
    if not resolved.exists():
        logger.error(f"Path does not exist: {resolved}")
        sys.exit(1)
    return resolved


def main() -> None:
    args = parse_args()
    if args.quiet:
        args.verbose = False

    config_path = resolve_path(args.config)
    model_path = resolve_path(args.model)
    data_path = resolve_path(args.data)
    statistics_path = Path(args.statistics).resolve()

    logger.info(f"Loading configuration from: {config_path}")
    config = load_config(str(config_path))

    logger.info(f"Loading model from: {model_path}")
    model = load_model(config, model_path)
    model.to(_get_device())

    logger.info(f"Loading dataset from: {data_path}")
    dataset = ImageDataset(root_dir=data_path)

    if statistics_path.exists():
        logger.info(f"Applying transforms with statistics from: {statistics_path}")
        apply_transform(
            statistics_path,
            dataset=dataset,
            img_size=config.data.img_size,
            train=False,
            three_gray_channels=config.data.three_gray_channels,
        )
    else:
        logger.warning(
            f"Statistics file not found at {statistics_path}, "
            "skipping transform setup"
        )

    dataloader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )

    logger.info("Starting benchmark...")
    results = benchmarking_loop(
        dataloader, model, num_warmup=args.num_warmup, verbose=args.verbose
    )

    print_results(results)

    if not args.no_mlflow and statistics_path.exists():
        logger.info("Logging results to MLflow...")
        mlflow_save_results(config, results)
    elif args.no_mlflow:
        logger.info("MLflow logging skipped (--no-mlflow)")
    else:
        logger.info(
            "MLflow logging skipped (statistics file required for experiment name)"
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
        sys.exit(0)
    except Exception as exc:
        logger.exception(f"Benchmark failed: {exc}")
        sys.exit(1)
