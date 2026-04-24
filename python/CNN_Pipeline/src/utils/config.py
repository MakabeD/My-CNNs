"""
Configuration loader for training parameters from YAML files.

This module provides functionality to load, validate, and parse
training configuration from YAML files, including model hyperparameters,
optimizer settings, criterion, data pipeline parameters, and training options.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class OptimizerConfig:
    """Configuration for the optimizer."""

    name: str = "adam"
    lr: float = 0.001
    weight_decay: float = 0.0
    momentum: float = 0.9
    betas: tuple = (0.9, 0.999)

    def __post_init__(self):
        if isinstance(self.betas, list):
            self.betas = tuple(self.betas)


@dataclass
class CriterionConfig:
    """Configuration for the loss function."""

    name: str = "cross_entropy"
    label_smoothing: float = 0.0
    reduction: str = "mean"


@dataclass
class ModelConfig:
    """Configuration for the model architecture."""

    name: str = "resnet18"
    num_classes: int = 2
    pretrained: bool = True
    dropout: float = 0.5
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    """Configuration for data loading and augmentation."""

    root_data_path: str = ""
    batch_size: int = 32
    num_workers: int = 4
    img_size: tuple = (224, 224)
    augment: bool = True
    config_path: Optional[str] = None  # Path to statistics config
    split_test: bool = True  # Whether to split the test set for validation (if false, requires having a 'val' folder)

    def __post_init__(self):
        if isinstance(self.img_size, list):
            self.img_size = tuple(self.img_size)


@dataclass
class TrainingConfig:
    """Configuration for training loop."""

    epochs: int = 100
    device: str = "auto"  # auto, cuda, cpu
    seed: int = 42
    early_stopping: bool = True
    patience: int = 10
    save_dir: str = "./checkpoints"
    experiment_name: str = "default_experiment"
    run_name: Optional[str] = None


@dataclass
class MLflowConfig:
    """Configuration for MLflow tracking."""

    tracking_uri: str = "https://dagshub.com"
    experiment_name: str = "My-CNNs"
    log_artifacts: bool = True
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class Config:
    """Main configuration container."""

    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    criterion: CriterionConfig = field(default_factory=CriterionConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)


def load_config(config_path: str) -> Config:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Config: A Config dataclass instance with all loaded parameters.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the YAML file is malformed.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path, "r") as f:
        config_dict = yaml.safe_load(f)

    return parse_config(config_dict)


def parse_config(config_dict: Dict[str, Any]) -> Config:
    """
    Parse a dictionary into a Config object.

    Args:
        config_dict: Dictionary containing configuration parameters.

    Returns:
        Config: Parsed configuration object.
    """
    # Extract sections with defaults
    model_dict = config_dict.get("model", {})
    optimizer_dict = config_dict.get("optimizer", {})
    criterion_dict = config_dict.get("criterion", {})
    data_dict = config_dict.get("data", {})
    training_dict = config_dict.get("training", {})
    mlflow_dict = config_dict.get("mlflow", {})

    # Create config objects
    model_config = ModelConfig(**model_dict)
    optimizer_config = OptimizerConfig(**optimizer_dict)
    criterion_config = CriterionConfig(**criterion_dict)
    data_config = DataConfig(**data_dict)
    training_config = TrainingConfig(**training_dict)
    mlflow_config = MLflowConfig(**mlflow_dict)

    return Config(
        model=model_config,
        optimizer=optimizer_config,
        criterion=criterion_config,
        data=data_config,
        training=training_config,
        mlflow=mlflow_config,
    )


def get_optimizer_class(name: str):
    """
    Get optimizer class by name.

    Args:
        name: Optimizer name (e.g., 'adam', 'sgd', 'rmsprop').

    Returns:
        Optimizer class from torch.optim.

    Raises:
        ValueError: If optimizer name is not supported.
    """
    import torch.optim as optim

    optimizers = {
        "adam": optim.Adam,
        "sgd": optim.SGD,
        "rmsprop": optim.RMSprop,
        "adamw": optim.AdamW,
        "adagrad": optim.Adagrad,
    }

    name_lower = name.lower()
    if name_lower not in optimizers:
        raise ValueError(
            f"Unsupported optimizer: {name}. Supported: {list(optimizers.keys())}"
        )

    return optimizers[name_lower]


def get_criterion_class(name: str):
    """
    Get criterion (loss function) class by name.

    Args:
        name: Criterion name (e.g., 'cross_entropy', 'bce', 'mse').

    Returns:
        Criterion class from torch.nn.

    Raises:
        ValueError: If criterion name is not supported.
    """
    import torch.nn as nn

    criteria = {
        "cross_entropy": nn.CrossEntropyLoss,
        "bce": nn.BCELoss,
        "bce_with_logits": nn.BCEWithLogitsLoss,
        "mse": nn.MSELoss,
        "l1": nn.L1Loss,
        "nll": nn.NLLLoss,
    }

    name_lower = name.lower()
    if name_lower not in criteria:
        raise ValueError(
            f"Unsupported criterion: {name}. Supported: {list(criteria.keys())}"
        )

    return criteria[name_lower]


def build_optimizer(optimizer_config: OptimizerConfig, model):
    """
    Build optimizer from configuration.

    Args:
        optimizer_config: OptimizerConfig instance.
        model: PyTorch model to optimize.

    Returns:
        torch.optim.Optimizer instance.
    """
    optimizer_class = get_optimizer_class(optimizer_config.name)

    kwargs = {"lr": optimizer_config.lr}

    if optimizer_config.name.lower() == "sgd":
        kwargs["momentum"] = optimizer_config.momentum
        kwargs["weight_decay"] = optimizer_config.weight_decay
    elif optimizer_config.name.lower() in ["adam", "adamw"]:
        kwargs["betas"] = optimizer_config.betas
        kwargs["weight_decay"] = optimizer_config.weight_decay
    else:
        kwargs["weight_decay"] = optimizer_config.weight_decay

    return optimizer_class(model.parameters(), **kwargs)


def build_criterion(criterion_config: CriterionConfig):
    """
    Build criterion from configuration.

    Args:
        criterion_config: CriterionConfig instance.

    Returns:
        torch.nn.Module (loss function) instance.
    """
    criterion_class = get_criterion_class(criterion_config.name)

    kwargs = {"reduction": criterion_config.reduction}

    if criterion_config.name.lower() == "cross_entropy":
        kwargs["label_smoothing"] = criterion_config.label_smoothing

    return criterion_class(**kwargs)


def save_config(config: Config, save_path: str) -> None:
    """
    Save configuration to a YAML file.

    Args:
        config: Config instance to save.
        save_path: Path where to save the YAML file.
    """
    config_dict = {
        "model": {
            "name": config.model.name,
            "num_classes": config.model.num_classes,
            "pretrained": config.model.pretrained,
            "dropout": config.model.dropout,
            "custom_params": config.model.custom_params,
        },
        "optimizer": {
            "name": config.optimizer.name,
            "lr": config.optimizer.lr,
            "weight_decay": config.optimizer.weight_decay,
            "momentum": config.optimizer.momentum,
            "betas": list(config.optimizer.betas),
        },
        "criterion": {
            "name": config.criterion.name,
            "label_smoothing": config.criterion.label_smoothing,
            "reduction": config.criterion.reduction,
        },
        "data": {
            "root_data_path": config.data.root_data_path,
            "batch_size": config.data.batch_size,
            "num_workers": config.data.num_workers,
            "img_size": list(config.data.img_size),
            "augment": config.data.augment,
            "config_path": config.data.config_path,
        },
        "training": {
            "epochs": config.training.epochs,
            "device": config.training.device,
            "seed": config.training.seed,
            "early_stopping": config.training.early_stopping,
            "patience": config.training.patience,
            "save_dir": config.training.save_dir,
            "experiment_name": config.training.experiment_name,
            "run_name": config.training.run_name,
        },
        "mlflow": {
            "tracking_uri": config.mlflow.tracking_uri,
            "experiment_name": config.mlflow.experiment_name,
            "log_artifacts": config.mlflow.log_artifacts,
            "tags": config.mlflow.tags,
        },
    }

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
