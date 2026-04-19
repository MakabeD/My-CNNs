"""
Training module for Casting Defect Detection CNN.

This module provides a clean architecture training loop with MLflow experiment tracking
via DagsHub, logging metrics such as loss, accuracy, precision, recall, and F1 score.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import dagshub
import mlflow
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def init_mlflow_experiment(
    experiment_name: str = "casting-defect-detection",
    repo_owner: str = "MakabeD",
    repo_name: str = "My-CNNs",
    run_name: Optional[str] = None,
) -> None:
    """
    Initialize DagsHub and MLflow experiment tracking.

    Args:
        experiment_name: Name of the MLflow experiment.
        repo_owner: GitHub repository owner.
        repo_name: GitHub repository name.
        run_name: Optional name for the specific run.
    """
    logger.info("Initializing DagsHub and MLflow...")

    # End any existing run before initializing
    if mlflow.active_run():
        logger.info("Ending existing MLflow run before initialization...")
        mlflow.end_run()

    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
    mlflow.set_experiment(experiment_name)
    if run_name:
        mlflow.set_tag("mlflow.runName", run_name)
    logger.info(f"MLflow experiment '{experiment_name}' initialized successfully.")


def compute_metrics(
    y_true: torch.Tensor, y_pred: torch.Tensor
) -> Tuple[float, float, float, float]:
    """
    Compute accuracy, precision, recall, and F1 score.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels (after argmax).

    Returns:
        Tuple of (accuracy, precision, recall, f1_score).
    """
    # Ensure tensors are on CPU for computation
    y_true = y_true.cpu()
    y_pred = y_pred.cpu()

    # Binary classification metrics (assuming 2 classes)
    tp = ((y_pred == 1) & (y_true == 1)).sum().item()
    tn = ((y_pred == 0) & (y_true == 0)).sum().item()
    fp = ((y_pred == 1) & (y_true == 0)).sum().item()
    fn = ((y_pred == 0) & (y_true == 1)).sum().item()

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return accuracy, precision, recall, f1_score


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """
    Train the model for one epoch.

    Args:
        model: The neural network model.
        dataloader: Training data loader.
        criterion: Loss function.
        optimizer: Optimization algorithm.
        device: Device to run computations on.
        epoch: Current epoch number.

    Returns:
        Dictionary containing average training loss and accuracy.
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    print(dataloader)
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        # Update progress bar
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    # Compute epoch metrics
    avg_loss = total_loss / len(dataloader)
    y_true = torch.tensor(all_labels)
    y_pred = torch.tensor(all_preds)
    accuracy, precision, recall, f1 = compute_metrics(y_true, y_pred)

    logger.info(
        f"Train Epoch {epoch} - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, "
        f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
    )

    return {
        "train_loss": avg_loss,
        "train_acc": accuracy,
        "train_precision": precision,
        "train_recall": recall,
        "train_f1": f1,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    phase: str = "Val",
) -> Dict[str, float]:
    """
    Validate the model on a dataset.

    Args:
        model: The neural network model.
        dataloader: Validation or test data loader.
        criterion: Loss function.
        device: Device to run computations on.
        epoch: Current epoch number.
        phase: Phase name ('Val' or 'Test').

    Returns:
        Dictionary containing average loss and metrics.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [{phase}]")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    # Compute epoch metrics
    avg_loss = total_loss / len(dataloader)
    y_true = torch.tensor(all_labels)
    y_pred = torch.tensor(all_preds)
    accuracy, precision, recall, f1 = compute_metrics(y_true, y_pred)

    logger.info(
        f"{phase} Epoch {epoch} - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, "
        f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
    )

    return {
        f"{phase.lower()}_loss": avg_loss,
        f"{phase.lower()}_acc": accuracy,
        f"{phase.lower()}_precision": precision,
        f"{phase.lower()}_recall": recall,
        f"{phase.lower()}_f1": f1,
    }


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader] = None,
    criterion: Optional[nn.Module] = None,
    optimizer: Optional[Optimizer] = None,
    num_epochs: int = 10,
    device: Optional[torch.device] = None,
    experiment_name: str = "casting-defect-detection",
    run_name: Optional[str] = None,
    log_hyperparams: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Main training loop with MLflow experiment tracking via DagsHub.

    This function handles the complete training process including:
    - Initializing MLflow tracking with DagsHub
    - Training and validation for each epoch
    - Logging all metrics (loss, accuracy, precision, recall, F1) with step
    - Final evaluation on test set (if provided)
    - Saving the best model based on validation loss

    Args:
        model: The neural network model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        test_loader: Optional DataLoader for test data.
        criterion: Loss function. Defaults to CrossEntropyLoss.
        optimizer: Optimizer. Defaults to Adam with lr=0.001.
        num_epochs: Number of training epochs.
        device: Device to use. Defaults to auto-detection.
        experiment_name: Name of the MLflow experiment.
        run_name: Optional name for this specific run.
        log_hyperparams: Optional dictionary of hyperparameters to log.

    Returns:
        Dictionary containing training history and best model path.
    """
    # Setup device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Setup loss and optimizer
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Initialize MLflow (this also starts a run via dagshub.init)
    init_mlflow_experiment(
        experiment_name=experiment_name, run_name=run_name
    )

    # Get the current active run (started by dagshub.init)
    run = mlflow.active_run()
    if run is None:
        # If no run is active, start a new one
        logger.info("No active MLflow run found, starting a new run...")
        with mlflow.start_run() as new_run:
            return _train_with_mlflow(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                criterion=criterion,
                optimizer=optimizer,
                num_epochs=num_epochs,
                device=device,
                log_hyperparams=log_hyperparams,
                run=new_run,
            )
    else:
        logger.info(f"Using existing MLflow run with ID: {run.info.run_id}")
        return _train_with_mlflow(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=num_epochs,
            device=device,
            log_hyperparams=log_hyperparams,
            run=run,
        )


def _train_with_mlflow(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader],
    criterion: nn.Module,
    optimizer: Optimizer,
    num_epochs: int,
    device: torch.device,
    log_hyperparams: Optional[Dict[str, Any]],
    run: Any,
) -> Dict[str, Any]:
    """
    Internal training function that runs within an MLflow context.

    Args:
        model: The neural network model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        test_loader: Optional DataLoader for test data.
        criterion: Loss function.
        optimizer: Optimizer.
        num_epochs: Number of training epochs.
        device: Device to use.
        log_hyperparams: Optional dictionary of hyperparameters to log.
        run: The active MLflow run.

    Returns:
        Dictionary containing training history and best model path.
    """
    logger.info(f"MLflow run active with ID: {run.info.run_id}")

    # Log hyperparameters
    if log_hyperparams:
        mlflow.log_params(log_hyperparams)
        logger.info(f"Logged hyperparameters: {log_hyperparams}")

    # Log model architecture
    mlflow.pytorch.log_model(model, artifact_path="model_initial")
    logger.info("Logged initial model architecture")

    best_val_loss = float("inf")
    best_model_state = None
    history = {"train": [], "val": [], "test": None}

    for epoch in range(1, num_epochs + 1):
        logger.info(f"\n{'='*50}\nEpoch {epoch}/{num_epochs}\n{'='*50}")

        # Training phase
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )

        # Validation phase
        val_metrics = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            phase="Val",
        )

        # Combine all metrics
        epoch_metrics = {**train_metrics, **val_metrics}

        # Log all metrics to MLflow with step
        for metric_name, metric_value in epoch_metrics.items():
            mlflow.log_metric(metric_name, metric_value, step=epoch)
        logger.info(f"Logged metrics for epoch {epoch} to MLflow")

        # Store history
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        # Save best model
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            best_model_state = model.state_dict().copy()
            logger.info(f"New best model saved with val_loss: {best_val_loss:.4f}")

            # Log best model to MLflow
            mlflow.pytorch.log_model(model, artifact_path="model_best")

    # Final evaluation on test set
    if test_loader is not None:
        logger.info("\n" + "=" * 50)
        logger.info("Final Evaluation on Test Set")
        logger.info("=" * 50)

        # Load best model for final evaluation
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        test_metrics = validate(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
            epoch=num_epochs,
            phase="Test",
        )
        history["test"] = test_metrics

        # Log test metrics
        for metric_name, metric_value in test_metrics.items():
            mlflow.log_metric(metric_name, metric_value, step=num_epochs)

    # Log final summary
    mlflow.log_param("best_val_loss", best_val_loss)
    mlflow.log_param("num_epochs", num_epochs)

    logger.info("\n" + "=" * 50)
    logger.info("Training Complete!")
    logger.info(f"Best Validation Loss: {best_val_loss:.4f}")
    logger.info("=" * 50)

    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "run_id": run.info.run_id,
        "model_state": best_model_state,
    }


if __name__ == "__main__":
    """
    Main entry point for training when running the script directly.

    This block loads configuration from a YAML file and starts the training process.
    Usage: python src/train/training.py --config path/to/config.yaml
    """
    import argparse
    import sys

    # Add src to path for imports
    src_path = Path(__file__).parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from utils.config import (
        load_config,
        build_optimizer,
        build_criterion,
        Config,
    )

    # Import model and pipeline modules (handled with try-except for missing dependencies)
    try:
        from model.model import get_model
    except ImportError as e:
        logger.warning(f"Could not import get_model: {e}")
        get_model = None

    try:
        from pipeline.data_pipeline import prepare_data
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
        from model.model import get_model
        model = get_model(
            model_name=config.model.name,
            num_classes=config.model.num_classes,
            pretrained=config.model.pretrained,
            dropout=config.model.dropout,
        )
        model = model.to(device)
        logger.info("Model created successfully!")
    except ImportError:
        logger.warning("Model module not found. Please implement get_model() in model/model.py")
        # Fallback: create a simple model for testing
        model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, config.model.num_classes)
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
        from pipeline.data_pipeline import prepare_data

        train_loader, val_loader, test_loader,classes  = prepare_data(
            root_data_path=config.data.root_data_path,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
        )
        logger.info("Data loaders created successfully!")
    except (ImportError, AttributeError) as e:
        logger.warning(f"Could not create dataloaders: {e}")
        logger.warning("Please implement create_dataloaders() in pipeline/data_pipeline.py")
        # Create dummy dataloaders for testing
        from torch.utils.data import DataLoader, TensorDataset
        dummy_dataset = TensorDataset(
            torch.randn(100, 3, *config.data.img_size),
            torch.randint(0, config.model.num_classes, (100,))
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
            run_name=config.training.run_name or f"{config.training.experiment_name}_{config.model.name}",
            log_hyperparams=hyperparams,
        )

        logger.info("=" * 60)
        logger.info("Training Completed Successfully!")
        logger.info(f"Best Validation Loss: {results['best_val_loss']:.4f}")
        logger.info(f"MLflow Run ID: {results['run_id']}")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)