# My-CNN

A modular PyTorch pipeline for training and deploying CNN-based image classification models. Currently supports **casting defect detection** and **chest X-ray classification**, with a roadmap toward a unified service API.

## Overview

My-CNN is designed as a reusable training and inference framework. Each domain (casting defects, chest X-rays) lives as its own module under `python/`, sharing a common core pipeline (`pipelines`) that handles:

- Config-driven training with YAML
- Custom CNNs and torchvision backbones (ResNet, VGG, EfficientNet)
- Data preparation with grayscale handling, normalization, augmentation, and stratified splits
- Experiment tracking with MLflow and DagsHub
- Dataset and model versioning with DVC

## Project Structure

```
My-CNN/
├── datasets/                     # DVC-tracked datasets
├── python/
│   ├── pipelines/             # Core training pipeline (shared)
│   │   ├── train.py              # Training entry point
│   │   ├── src/
│   │   │   ├── pipeline/         # Data loaders, transforms, statistics
│   │   │   ├── model/            # Model factory (custom + torchvision)
│   │   │   ├── train/            # Training loop, MLflow logging
│   │   │   └── utils/            # Config parsing, optimizer/criterion builders
│   │   └── requirements.txt
│   │
│   ├── casting-def-detector/     # Casting defect detection module
│   │   ├── configs/casting.yaml
│   │   ├── models/               # DVC-trained model artifacts
│   │   └── statistics.json
│   │
│   └── chest_xray/               # Chest X-ray classification module
│       ├── src/
│       │   ├── processing/       # Inference preprocessing
│       │   └── inference/        # Predictor class & CLI
│       ├── configs/
│       ├── models/               # DVC-trained model artifacts
│       └── statistics.json
```

## Supported Models

| Model Name       | Description                           | Architecture        |
|------------------|---------------------------------------|---------------------|
| `casting_cnn`    | Custom CNN for casting defect images  | 3 conv blocks + FC  |
| `chest_xray`     | Custom CNN for chest X-ray images     | 3 conv blocks + FC  |
| `resnet18`       | Torchvision ResNet-18                 | Pretrained optional |
| `resnet34`       | Torchvision ResNet-34                 | Pretrained optional |
| `resnet50`       | Torchvision ResNet-50                 | Pretrained optional |
| `vgg16`          | Torchvision VGG-16                    | Pretrained optional |
| `efficientnet_b0`| Torchvision EfficientNet-B0           | Pretrained optional |

## Installation

1. Create a virtual environment and install dependencies:

```bash
cd python/pipelines
pip install -r requirements.txt
```

## Training

Each domain module has its own YAML configuration. To train a model:

```bash
cd python/<module>
python ../pipelines/train.py --config configs/<config>.yaml
```

### Examples

**Casting defect detector:**

```bash
cd python/casting-def-detector
python ../pipelines/train.py --config configs/casting.yaml
```

**Chest X-ray with custom CNN:**

```bash
cd python/chest_xray
python ../pipelines/train.py --config configs/xray-config1.yaml
```

**Chest X-ray with ResNet-18:**

```bash
cd python/chest_xray
python ../pipelines/train.py --config configs/xray-resnet-config1.yaml
```

## Inference

The chest X-ray module includes a CLI predictor:

```bash
cd python/chest_xray
python src/inference/predictor.py images /path/to/image1.jpg /path/to/image2.jpg \
  --model-path models/<checkpoint>.pt \
  --config configs/xray-config1.yaml \
  --class-names anomaly normal
```

## Configuration

All training runs are driven by YAML configs with the following sections:

```yaml
model:
  name: "casting_cnn"       # Model architecture
  num_classes: 2            # Output classes
  pretrained: false         # Use ImageNet weights (torchvision only)
  dropout: 0.5              # Dropout rate

optimizer:
  name: "adam"              # adam, sgd, rmsprop, adamw, adagrad
  lr: 0.001
  weight_decay: 0.0001

criterion:
  name: "cross_entropy"     # cross_entropy, bce, mse, l1, nll
  label_smoothing: 0.1

data:
  root_data_path: "../../datasets/casting_data"
  batch_size: 32
  img_size: [300, 300]
  augment: true
  three_gray_channels: false

training:
  epochs: 100
  device: "auto"
  seed: 42
  early_stopping: true
  patience: 10
  save_dir: "./models"

mlflow:
  tracking_uri: "https://dagshub.com"
  experiment_name: "My-CNNs_casting-defect-detection"
  log_artifacts: true
```

## Key Features

- **No data leakage**: Normalization statistics computed from training split only
- **Stratified splits**: Maintains class balance across train/val sets
- **Grayscale support**: Native 1-channel or 3-channel replicated mode
- **Data augmentation**: Random horizontal flip, rotation, and affine transforms
- **MLflow + DagsHub**: Full experiment tracking (loss, accuracy, precision, recall, F1)
- **DVC versioning**: Datasets and model checkpoints tracked with DVC
- **Flexible model factory**: Swap architectures via config without code changes

## Reproducibility

- Random seeds are set for PyTorch and CUDA
- Stratified train/val splits use `random_state=42`
- Normalization statistics are saved to `statistics.json` per module
- All hyperparameters captured in YAML and logged to MLflow

## Roadmap

- [ ] Unified REST API to expose each trained model as a service
- [ ] Docker containerization for deployment
- [ ] Additional domain modules (e.g., skin lesion, retinal scan)
- [ ] Model comparison dashboard
- [ ] Automated hyperparameter tuning
- [ ] ONNX export for edge deployment
