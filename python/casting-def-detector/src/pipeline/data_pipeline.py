import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import ImageFolder


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


def calculate_mean_std(data_dir, sample_size=2000):
    """
    Calculates mean and std of the dataset for normalization.
    Only samples a subset to speed up calculation if dataset is huge.
    """
    print("Calculating dataset statistics (mean/std)...")

    # Basic transform just to get tensors
    temp_transform = transforms.Compose(
        [transforms.Grayscale(), transforms.Resize((300, 300)), transforms.ToTensor()]
    )

    temp_dataset = CastingDataset(root_dir=data_dir, transform=temp_transform)

    # Sample if too large
    total_len = len(temp_dataset)
    if total_len > sample_size:
        indices = np.random.choice(total_len, sample_size, replace=False)
        sampler = torch.utils.data.SubsetRandomSampler(indices)
        loader = DataLoader(temp_dataset, batch_size=64, sampler=sampler, num_workers=4)
        effective_size = sample_size
    else:
        loader = DataLoader(temp_dataset, batch_size=64, shuffle=False, num_workers=4)
        effective_size = total_len

    mean = 0.0
    std = 0.0
    total_samples = 0

    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)  # Flatten spatial dims

        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_samples += batch_samples
        print(mean)
    mean /= effective_size
    std /= effective_size

    return mean.item(), std.item()


def get_transforms(mean, std, train=True):
    """Returns transforms with optional augmentation for training"""
    base_transforms = [
        transforms.Resize((300, 300)),  # Enforce 300x300
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std]),
    ]

    if train:
        # Insert augmentations before ToTensor/Normalize
        # Note: We apply these on the PIL image before ToTensor converts to tensor
        # So we reconstruct the list slightly differently or use separate logic.
        # Let's rebuild specifically for clarity:

        aug_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Small shifts
        ]

        return transforms.Compose(aug_transforms + base_transforms)

    return transforms.Compose(base_transforms)


def prepare_data(root_data_path, batch_size=32, val_split=0.2, num_workers=4):
    """
    Main function to prepare DataLoaders.

    Args:
        root_data_path (str): Path to the folder containing 'train' and 'test' folders.
        batch_size (int): Batch size for loaders.
        val_split (float): Fraction of training data to use for validation.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        train_loader, val_loader, test_loader, classes
    """

    train_dir = root_data_path / "train"
    test_dir = root_data_path / "test"

    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        raise FileNotFoundError(
            f"Expected 'train' and 'test' folders inside {root_data_path}"
        )

    # 1. Calculate Stats from Training Data only (to prevent data leakage)
    mean, std = calculate_mean_std(train_dir)
    print(f"Calculated Stats -> Mean: {mean:.4f}, Std: {std:.4f}")

    # 2. Define Transforms
    train_transform = get_transforms(mean, std, train=True)
    # Validation and Test should NOT have augmentation, only normalization
    val_test_transform = get_transforms(mean, std, train=False)

    # 3. Load Full Training Dataset (initially without transform to split indices)
    # We load raw to split indices, then apply transforms to subsets
    full_train_dataset = CastingDataset(root_dir=train_dir, transform=None)
    test_dataset = CastingDataset(root_dir=test_dir, transform=val_test_transform)

    classes = full_train_dataset.classes
    print(f"Classes found: {classes}")

    # 4. Split Training into Train & Validation
    # We split indices based on labels to ensure stratification (balanced classes)
    labels = [label for _, label in full_train_dataset.dataset.samples]

    train_indices, val_indices = train_test_split(
        list(range(len(full_train_dataset))),
        test_size=val_split,
        stratify=labels,
        random_state=42,
    )

    # Create Subsets
    train_subset = Subset(full_train_dataset, train_indices)
    val_subset = Subset(full_train_dataset, val_indices)

    # Assign transforms to the underlying dataset of the subsets
    # Note: Subset shares the underlying dataset, so we must be careful.
    # Best practice here: Create new datasets with transforms for each split
    # OR modify the transform of the base dataset carefully.
    # Since Subset references the same base object, changing transform affects all.
    # Solution: Re-instantiate datasets for splits with specific transforms.

    # Re-create specific datasets for splits to ensure independent transforms
    train_final_dataset = CastingDataset(root_dir=train_dir, transform=train_transform)
    val_final_dataset = CastingDataset(root_dir=train_dir, transform=val_test_transform)

    # Now re-apply the split indices to these new dataset instances
    # We need to map the indices from the original un-transformed dataset to the new ones
    # Since the file order in ImageFolder is deterministic (sorted), indices match.
    train_set = Subset(train_final_dataset, train_indices)
    val_set = Subset(val_final_dataset, val_indices)

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop last batch if it's smaller than batch_size (good for BatchNorm)
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print("Data Ready:")
    print(f"  - Train batches: {len(train_loader)} ({len(train_set)} samples)")
    print(f"  - Val batches:   {len(val_loader)} ({len(val_set)} samples)")
    print(f"  - Test batches:  {len(test_loader)} ({len(test_dataset)} samples)")

    return train_loader, val_loader, test_loader, classes


def get_device():
    """Selects the best available device (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("⚠️  Using CPU")
    return device
