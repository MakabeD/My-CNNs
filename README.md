# My-CNN

PyTorch project for binary classification of industrial casting images into defective and non-defective classes.

The repository already includes a full training pipeline instead of just a notebook or a single script:

- Config-driven training with YAML
- A custom CNN and support for popular torchvision backbones
- Data preparation with grayscale conversion, normalization, augmentation, and train/validation splitting
- Experiment tracking with MLflow and DagsHub
- DVC-managed datasets and trained model artifacts

## Project Structure

- `datasets/`: dataset files tracked with DVC
- `python/casting-def-detector/configs/config.yaml`: training configuration
- `python/casting-def-detector/src/pipeline/data_pipeline.py`: dataloaders, transforms, and dataset statistics
- `python/casting-def-detector/src/model/model.py`: model definitions and model factory
- `python/casting-def-detector/src/train/training.py`: training, validation, metrics, and MLflow logging
- `python/casting-def-detector/src/utils/config.py`: config parsing plus optimizer and criterion builders
- `python/casting-def-detector/models/`: saved model artifacts tracked with DVC

## Dataset

Source:

[Real Life Industrial Dataset of Casting Product](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)

Classes:

1. `def_front`
2. `ok_front`

Repository dataset summary:

- Train: `def_front = 3758`, `ok_front = 2875`
- Test: `def_front = 453`, `ok_front = 262`

The training pipeline creates a validation split from the training set and computes normalization statistics from the training subset only to avoid data leakage.

## Current Features

- Grayscale image handling tailored to the dataset
- Stratified train/validation split
- Mean and standard deviation calculation from training data only
- Data augmentation for training
- Configurable optimizers and loss functions
- Accuracy, precision, recall, and F1 tracking
- Best-model saving
- Experiment tracking with MLflow and DagsHub

## How To Train

Main config:

`python/casting-def-detector/configs/config.yaml`

Run from the project root:

```powershell
cd python/casting-def-detector
python src/train/training.py --config configs/config.yaml
```

## Current Training Setup

- Model: `casting_cnn`
- Classes: `2`
- Image size: `300x300`
- Optimizer: `Adam`
- Loss: `CrossEntropyLoss` with label smoothing
- Tracking: `MLflow` with `DagsHub`

## Reproducibility

The repository already includes:

- DVC tracking for datasets and model outputs
- Saved normalization statistics in `python/casting-def-detector/statistics.json`
- Centralized configuration in YAML
