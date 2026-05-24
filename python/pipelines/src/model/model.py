import torch
import torch.nn as nn
import torch.nn.functional as F


class CastingDefectCNN(nn.Module):
    def __init__(self, num_classes=2, dropout=0.5):
        """
        Binary classification model for casting defect detection.
        Classes: 0 = Ok, 1 = Defective

        Args:
            num_classes (int): Number of output classes (default=2 for binary classification).
        """
        super(CastingDefectCNN, self).__init__()

        # --- Convolutional Block 1 ---
        # Input: (Batch, 1, H, W) [Grayscale images]
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Convolutional Block 2 ---
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Convolutional Block 3 ---
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Adaptive pooling removes the dependency on a fixed input image size.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(in_features=128, out_features=64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        # Block 1: Conv -> BN -> ReLU -> MaxPool
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Block 2: Conv -> BN -> ReLU -> MaxPool
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Block 3: Conv -> BN -> ReLU -> MaxPool
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)

        # Global pooling keeps the classifier small and input-size agnostic.
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Dense Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class ChestXray(nn.Module):
    def __init__(self, num_classes=2, dropout=0.5):
        """
        Binary classification model for casting defect detection.
        Classes: 0 = Ok, 1 = Defective

        Args:
            num_classes (int): Number of output classes (default=2 for binary classification).
        """
        super(ChestXray, self).__init__()

        # --- Convolutional Block 1 ---
        # Input: (Batch, 1, H, W) [Grayscale images]
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Convolutional Block 2 ---
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Convolutional Block 3 ---
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Adaptive pooling removes the dependency on a fixed input image size.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(in_features=128, out_features=64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        # Block 1: Conv -> BN -> ReLU -> MaxPool
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Block 2: Conv -> BN -> ReLU -> MaxPool
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Block 3: Conv -> BN -> ReLU -> MaxPool
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)

        # Global pooling keeps the classifier small and input-size agnostic.
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Dense Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def _change_head(model: nn.Module, num_classes: int) -> nn.Module:
    """
    Helper function to change the classification head of a pretrained model.

    Args:
        model: The base model with a pretrained head.
        num_classes: The number of output classes for the new head.

    Returns:
        The model with the modified classification head.
    """
    if hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        # For models like VGG and EfficientNet
        if isinstance(model.classifier[-1], nn.Linear):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
        else:
            raise ValueError("Unsupported classifier structure for modifying head.")
    else:
        raise ValueError("Model does not have a recognizable classification head.")

    return model


def get_model(
    model_name: str = "casting_cnn",
    num_classes: int = 2,
    pretrained: bool = False,
    dropout: float = 0.5,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create models by name.

    Args:
        model_name: Name of the model architecture.
        num_classes: Number of output classes.
        pretrained: Whether to use pretrained weights (for torchvision models).
        dropout: Dropout rate.
        **kwargs: Additional model-specific arguments.

    Returns:
        nn.Module: The instantiated model.

    Raises:
        ValueError: If model_name is not supported.
    """
    model_name_lower = model_name.lower()

    if model_name_lower == "casting_cnn":
        model = CastingDefectCNN(num_classes=num_classes, dropout=dropout)
    elif model_name_lower == "chest_xray":
        model = ChestXray(num_classes=num_classes, dropout=dropout)
    elif model_name_lower in (
        "resnet18",
        "resnet34",
        "resnet50",
        "vgg16",
        "efficientnet",
    ):
        try:
            import torchvision.models as models

            resnet_configs = {
                "resnet18": (models.resnet18, models.ResNet18_Weights),
                "resnet34": (models.resnet34, models.ResNet34_Weights),
                "resnet50": (models.resnet50, models.ResNet50_Weights),
            }

            if model_name_lower in resnet_configs:
                builder, weights_cls = resnet_configs[model_name_lower]
                base_model = builder(
                    weights=weights_cls.DEFAULT if pretrained else None
                )
            elif model_name_lower == "vgg16":
                base_model = models.vgg16(
                    weights=models.VGG16_Weights.DEFAULT if pretrained else None
                )
            elif model_name_lower == "efficientnet":
                base_model = models.efficientnet_b0(
                    weights=models.EfficientNet_B0_Weights.DEFAULT
                    if pretrained
                    else None
                )
            else:
                raise ValueError(f"Unsupported torchvision model: {model_name}")

            if pretrained:
                for param in base_model.parameters():
                    param.requires_grad = False

            base_model = _change_head(base_model, num_classes)
            model = base_model
        except ImportError:
            raise ImportError(
                "torchvision is required for pretrained models. Install with: pip install torchvision"
            )
    else:
        raise ValueError(
            f"Unsupported model: {model_name}. Supported: casting_cnn, resnet18, resnet34, resnet50, vgg16, efficientnet"
        )

    return model


# --- Example Usage ---
if __name__ == "__main__":
    # Initialize the model for binary classification (Ok vs Defective)
    model = CastingDefectCNN(num_classes=2)

    # Create a dummy input tensor (Batch size=4, 1 Channel Grayscale, 256x256 Image)
    # The adaptive pooling head allows different input sizes.
    dummy_input = torch.randn(4, 1, 256, 256)
    # Forward pass
    output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")  # Should be [4, 2] for binary classification
    print(f"\nModel Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
