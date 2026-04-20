import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import ImageFolder

CONFIG_PATH = Path(__file__).parent.parent.parent.joinpath("statistics.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class CastingDataset(Dataset):
    """
    Custom dataset to handle specific folder structures and ensure grayscale conversion.
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Use ImageFolder to automatically handle subfolders (def_front, ok_front)
        self.dataset = ImageFolder(root=root_dir, transform=None)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get image and label from underlying ImageFolder
        img_path, label = self.dataset.samples[idx]

        # Open image and force Grayscale (1 channel)
        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)

        return image, label

    @property
    def classes(self):
        return self.dataset.classes

    @property
    def class_to_idx(self):
        return self.dataset.class_to_idx


def save_statistics(config_path, mean, std, classes=None, additional_info=None):
    """
    Saves dataset statistics (mean, std) and optional metadata to a JSON config file.

    Args:
        config_path (str or Path): Path to the output JSON config file.
        mean (float): Mean value of the dataset.
        std (float): Standard deviation of the dataset.
        classes (list, optional): List of class names. Defaults to None.
        additional_info (dict, optional): Additional metadata to save. Defaults to None.
    """
    config_data = {
        "mean": mean,
        "std": std,
    }

    if classes is not None:
        config_data["classes"] = classes

    if additional_info is not None:
        config_data.update(additional_info)

    # Ensure parent directory exists
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=4)

    logger.info(f"Statistics saved to {config_path}")


def load_statistics(config_path):
    """
    Loads dataset statistics (mean, std) and optional metadata from a JSON config file.

    Args:
        config_path (str or Path): Path to the JSON config file.

    Returns:
        dict: Dictionary containing mean, std, classes (if available), and any additional info.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If required fields (mean, std) are missing from the config.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config_data = json.load(f)

    # Validate required fields
    if "mean" not in config_data or "std" not in config_data:
        raise ValueError("Config file must contain 'mean' and 'std' fields.")

    logger.info(f"Statistics loaded from {config_path}")
    return config_data


def calculate_mean_std_from_subset(dataset_subset, sample_size=2000):
    """
    Calculates mean and std ONLY from the provided subset (e.g., training subset).
    This prevents data leakage from validation/test sets.
    """
    logger.info(
        "Calculating dataset statistics (mean/std) from TRAINING subset only..."
    )

    # Basic transform just to get tensors (No augmentation, just resize/to_tensor)
    temp_transform = transforms.Compose(
        [transforms.Grayscale(), transforms.Resize((300, 300)), transforms.ToTensor()]
    )

    # We need to access the underlying dataset and map the subset indices
    base_dataset = dataset_subset.dataset
    indices = dataset_subset.indices

    total_len = len(indices)
    if total_len > sample_size:
        # Sample randomly from the SUBSET indices only
        sampled_indices = np.random.choice(indices, sample_size, replace=False)
        effective_size = sample_size
    else:
        sampled_indices = indices
        effective_size = total_len

    mean = 0.0
    std = 0.0
    total_samples = 0

    # Manual iteration to avoid creating a full Subset DataLoader which might be tricky with transforms
    # We temporarily apply the temp_transform to read pixels
    for idx in sampled_indices:
        img_path, _ = base_dataset.dataset.samples[idx]
        image = Image.open(img_path).convert("L")
        tensor = temp_transform(image)

        # Flatten spatial dims (C, H, W) -> (C, H*W)
        tensor = tensor.view(tensor.size(0), -1)

        mean += tensor.mean(1).sum().item()
        std += tensor.std(1).sum().item()
        total_samples += 1

        # Progress indicator for large datasets
        if total_samples % 500 == 0:
            logger.info(f"  Processed {total_samples}/{effective_size} images...")

    mean /= effective_size * 1.0  # 1 channel
    std /= effective_size * 1.0
    save_statistics(CONFIG_PATH, mean, std)
    return mean, std


def get_transforms(mean, std, train=True):
    """Returns transforms with optional augmentation for training"""
    base_transforms = [
        transforms.Resize((300, 300)),  # Enforce 300x300
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std]),
    ]

    if train:
        # Insert augmentations before ToTensor/Normalize
        aug_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Small shifts
        ]
        return transforms.Compose(aug_transforms + base_transforms)

    return transforms.Compose(base_transforms)


def prepare_data(root_data_path, batch_size=32, val_split=0.2, num_workers=0):
    """
    Main function to prepare DataLoaders.
    CRITICAL: Splits data FIRST, then calculates stats from TRAIN split ONLY.

    Args:
        root_data_path (str): Path to the folder containing 'train' and 'test' folders.
        batch_size (int): Batch size for loaders.
        val_split (float): Fraction of training data to use for validation.
        num_workers (int): Number of subprocesses. Default 0 for Windows/OneDrive safety.

    Returns:
        train_loader, val_loader, test_loader, classes
    """

    train_dir = Path(root_data_path) / "train"
    test_dir = Path(root_data_path) / "test"

    logger.info(f"Looking for data in: {Path(root_data_path).absolute().resolve()}")
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        raise FileNotFoundError(
            f"Expected 'train' and 'test' folders inside {root_data_path}"
        )

    # 1. Load Raw Training Dataset (no transforms yet)
    full_train_dataset = CastingDataset(root_dir=str(train_dir), transform=None)
    classes = full_train_dataset.classes
    logger.info(f"Classes found: {classes}")

    # 2. Split Indices FIRST (Stratified to keep class balance)
    labels = [label for _, label in full_train_dataset.dataset.samples]

    train_indices, val_indices = train_test_split(
        list(range(len(full_train_dataset))),
        test_size=val_split,
        stratify=labels,
        random_state=42,  # Reproducible splits
    )

    logger.info(
        f"Split complete: {len(train_indices)} Train, {len(val_indices)} Val samples."
    )

    # 3. Create Subsets based on indices
    # We create temporary subsets just to pass to the stats calculator
    train_subset_for_stats = Subset(full_train_dataset, train_indices)

    # 4. Calculate Stats FROM THE TRAIN SUBSET ONLY (No Leakage!)
    mean, std = calculate_mean_std_from_subset(train_subset_for_stats)
    logger.info(f"Calculated Stats (Train Only) -> Mean: {mean:.4f}, Std: {std:.4f}")

    # 5. Define Transforms using the calculated stats
    train_transform = get_transforms(mean, std, train=True)
    val_test_transform = get_transforms(mean, std, train=False)

    # 6. Create Final Datasets with Transforms
    # We re-instantiate datasets to ensure clean transform application
    train_base = CastingDataset(root_dir=str(train_dir), transform=train_transform)
    val_base = CastingDataset(root_dir=str(train_dir), transform=val_test_transform)
    test_base = CastingDataset(root_dir=str(test_dir), transform=val_test_transform)

    # Apply the split indices to the new transformed datasets
    train_set = Subset(train_base, train_indices)
    val_set = Subset(val_base, val_indices)
    test_set = test_base  # Test set is separate folder, no splitting needed

    # 7. Create DataLoaders
    # Note: num_workers=0 is safer for Windows/OneDrive. Increase to 4 if on local SSD.
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info("Data Ready:")
    logger.info(f"  - Train batches: {len(train_loader)} ({len(train_set)} samples)")
    logger.info(f"  - Val batches:   {len(val_loader)} ({len(val_set)} samples)")
    logger.info(f"  - Test batches:  {len(test_loader)} ({len(test_set)} samples)")

    return train_loader, val_loader, test_loader, classes


def get_device():
    """Selects the best available device (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        logger.warning("Using CPU")
    return device
