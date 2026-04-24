import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from src.train.training import train_loop
from src.utils.config import (
    Config,
    build_criterion,
    build_optimizer,
    load_config,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train casting defect detection model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file",
    )
    return parser.parse_args()


def load_training_config(config_path: str) -> Config:
    logger.info(f"Loading configuration from: {config_path}")
    try:
        config = load_config(config_path)
        logger.info("Configuration loaded successfully!")
        return config
    except FileNotFoundError as exc:
        logger.error(f"Configuration file not found: {exc}")
        sys.exit(1)
    except Exception as exc:
        logger.exception(f"Unexpected error loading configuration: {exc}")
        sys.exit(1)


def set_random_seed(seed: int | None) -> None:
    if seed:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info(f"Random seed set to: {seed}")


def get_device(device_name: str) -> torch.device:
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_name == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")
    return device


def build_model(config: Config, device: torch.device) -> nn.Module:
    logger.info(f"Creating model: {config.model.name}")

    try:
        from src.model.model import get_model

        model = get_model(
            model_name=config.model.name,
            num_classes=config.model.num_classes,
            pretrained=config.model.pretrained,
            dropout=config.model.dropout,
            img_size=config.data.img_size,
        )
        model = model.to(device)
        logger.info("Model created successfully!")
        return model
    except ImportError as exc:
        logger.warning(
            "Could not import get_model (%s). Please implement get_model() in "
            "model/model.py",
            exc,
        )
        model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, config.model.num_classes),
        ).to(device)
        logger.info("Created fallback model for testing")
        return model


def build_dataloaders(config: Config):
    logger.info("Creating data loaders...")

    try:
        from src.pipeline.data_pipeline import prepare_data

        train_loader, val_loader, test_loader, classes = prepare_data(
            root_data_path=config.data.root_data_path,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            img_size=config.data.img_size,
            split_test=config.data.split_test
        )
        logger.info("Data loaders created successfully!")
        return train_loader, val_loader, test_loader, classes
    except (ImportError, AttributeError) as exc:
        logger.warning(f"Could not create dataloaders: {exc}")
        logger.warning(
            "Please implement prepare_data() in pipeline/data_pipeline.py"
        )

        from torch.utils.data import DataLoader, TensorDataset

        dummy_dataset = TensorDataset(
            torch.randn(100, 3, *config.data.img_size),
            torch.randint(0, config.model.num_classes, (100,)),
        )
        train_loader = DataLoader(dummy_dataset, batch_size=config.data.batch_size)
        val_loader = DataLoader(dummy_dataset, batch_size=config.data.batch_size)
        test_loader = None
        logger.info("Created dummy data loaders for testing")
        return train_loader, val_loader, test_loader, None


def build_hyperparams(config: Config) -> dict[str, object]:
    return {
        "model_name": config.model.name,
        "num_classes": config.model.num_classes,
        "pretrained": config.model.pretrained,
        "optimizer": config.optimizer.name,
        "learning_rate": config.optimizer.lr,
        "weight_decay": config.optimizer.weight_decay,
        "criterion": config.criterion.name,
        "batch_size": config.data.batch_size,
        "img_size": str(config.data.img_size),
        "epochs": config.training.epochs,
        "seed": config.training.seed,
        "early_stopping": config.training.early_stopping,
        "patience": config.training.patience,
    }


def save_best_model(config: Config, results: dict[str, object]) -> None:
    savedir = Path(config.training.save_dir)
    file_name = f"casting-defects-best_model_{results['best_val_loss']:.4f}.pt"
    save_path = savedir.joinpath(file_name)

    torch.save(results["model_state"], save_path)
    logger.info(
        f"Best model saved to: ./models/casting-defects-best_model_{results['best_val_loss']:.4f}.pt"
    )


def run_training(config: Config) -> None:
    set_random_seed(config.training.seed)
    device = get_device(config.training.device)

    model = build_model(config, device)

    logger.info(f"Creating optimizer: {config.optimizer.name}")
    optimizer = build_optimizer(config.optimizer, model)

    logger.info(f"Creating criterion: {config.criterion.name}")
    criterion = build_criterion(config.criterion)

    train_loader, val_loader, test_loader, _classes = build_dataloaders(config)
    hyperparams = build_hyperparams(config)

    logger.info("=" * 60)
    logger.info("Starting Training")
    logger.info("=" * 60)

    results = train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=config.training.epochs,
        device=device,
        experiment_name=config.mlflow.experiment_name,
        run_name=config.training.run_name
        or f"{config.training.experiment_name}_{config.model.name}",
        log_hyperparams=hyperparams,
    )

    logger.info("=" * 60)
    logger.info("Training Completed Successfully!")
    logger.info(f"Best Validation Loss: {results['best_val_loss']:.4f}")
    logger.info(f"MLflow Run ID: {results['run_id']}")
    save_best_model(config, results)
    logger.info("=" * 60)


def main() -> None:
    args = parse_args()
    config = load_training_config(args.config)

    try:
        run_training(config)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
    except Exception as exc:
        logger.exception(f"Training failed with error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    """
    Main entry point for training when running the script directly.

    This block loads configuration from a YAML file and starts the training process.
    Usage: python src/train/training.py --config path/to/config.yaml
    """
    main()
