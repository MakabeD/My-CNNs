import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from src.train.training import train_loop

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class main:
    def __init__(self) -> None:
        from src.utils.config import (
            Config,
            build_criterion,
            build_optimizer,
            load_config,
        )

        # Import model and pipeline modules (handled with try-except for missing dependencies)
        try:
            from src.model.model import get_model
        except ImportError as e:
            logger.warning(f"Could not import get_model: {e}")
            get_model = None

        try:
            from src.pipeline.data_pipeline import prepare_data
        except ImportError as e:
            logger.warning(f"Could not import prepare_data: {e}")
            prepare_data = None

        # Parse command line arguments
        parser = argparse.ArgumentParser(
            description="Train casting defect detection model"
        )
        parser.add_argument(
            "--config",
            type=str,
            default="configs/config.yaml",
            help="Path to YAML configuration file",
        )
        args = parser.parse_args()

        # Load configuration
        logger.info(f"Loading configuration from: {args.config}")
        try:
            config: Config = load_config(args.config)
            logger.info("Configuration loaded successfully!")
        except FileNotFoundError as e:
            logger.error(f"Configuration file not found: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            sys.exit(1)

        # Set random seed for reproducibility
        if config.training.seed:
            torch.manual_seed(config.training.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.training.seed)
            logger.info(f"Random seed set to: {config.training.seed}")

        # Setup device
        if config.training.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif config.training.device == "cuda":
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        logger.info(f"Using device: {device}")

        # Create model
        logger.info(f"Creating model: {config.model.name}")
        try:
            # Assuming you have a get_model function in model.py
            # Adjust this based on your actual model creation logic
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
        except ImportError:
            logger.warning(
                "Model module not found. Please implement get_model() in model/model.py"
            )
            # Fallback: create a simple model for testing
            model = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, config.model.num_classes),
            ).to(device)
            logger.info("Created fallback model for testing")

        # Create optimizer
        logger.info(f"Creating optimizer: {config.optimizer.name}")
        optimizer = build_optimizer(config.optimizer, model)

        # Create criterion
        logger.info(f"Creating criterion: {config.criterion.name}")
        criterion = build_criterion(config.criterion)

        # Create dataloaders
        logger.info("Creating data loaders...")
        try:
            # Assuming you have a create_dataloaders function in data_pipeline.py
            # Adjust this based on your actual data loading logic
            from src.pipeline.data_pipeline import prepare_data

            train_loader, val_loader, test_loader, classes = prepare_data(
                root_data_path=config.data.root_data_path,
                batch_size=config.data.batch_size,
                num_workers=config.data.num_workers,
                img_size=config.data.img_size,
            )
            logger.info("Data loaders created successfully!")
        except (ImportError, AttributeError) as e:
            logger.warning(f"Could not create dataloaders: {e}")
            logger.warning(
                "Please implement create_dataloaders() in pipeline/data_pipeline.py"
            )
            # Create dummy dataloaders for testing
            from torch.utils.data import DataLoader, TensorDataset

            dummy_dataset = TensorDataset(
                torch.randn(100, 3, *config.data.img_size),
                torch.randint(0, config.model.num_classes, (100,)),
            )
            train_loader = DataLoader(dummy_dataset, batch_size=config.data.batch_size)
            val_loader = DataLoader(dummy_dataset, batch_size=config.data.batch_size)
            test_loader = None
            logger.info("Created dummy data loaders for testing")

        # Prepare hyperparameters for logging
        hyperparams = {
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

        # Start training
        logger.info("=" * 60)
        logger.info("Starting Training")
        logger.info("=" * 60)

        try:
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
            torch.save(
                results["model_state"],
                f"./models/casting-defects-best_model_{results['best_val_loss']:.4f}.pt",
            )
            logger.info(
                f"Best model saved to: ./models/casting-defects-best_model_{results['best_val_loss']:.4f}.pt"
            )
            logger.info("=" * 60)

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    """
    Main entry point for training when running the script directly.

    This block loads configuration from a YAML file and starts the training process.
    Usage: python src/train/training.py --config path/to/config.yaml
    """
    main()
