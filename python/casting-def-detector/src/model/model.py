import torch
import torch.nn as nn
import torch.nn.functional as F

class CastingDefectCNN(nn.Module):
    def __init__(self, num_classes=2):
        """
        Binary classification model for casting defect detection.
        Classes: 0 = Ok, 1 = Defective

        Args:
            num_classes (int): Number of output classes (default=2 for binary classification).
        """
        super(CastingDefectCNN, self).__init__()

        # --- Convolutional Block 1 ---
        # Input: (Batch, 1, 300, 300) [Grayscale images, 300x300]
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # Output: (Batch, 32, 300, 300)
        self.bn1 = nn.BatchNorm2d(32)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output: (Batch, 32, 150, 150) -> Halved dimensions

        # --- Convolutional Block 2 ---
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # Output: (Batch, 64, 150, 150)
        self.bn2 = nn.BatchNorm2d(64)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output: (Batch, 64, 75, 75) -> Halved dimensions again

        # --- Dense (Fully Connected) Layer ---
        # We flatten the output from pool2: 64 channels * 75 height * 75 width
        self.fc1 = nn.Linear(in_features=64 * 75 * 75, out_features=128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

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

        # Flatten the tensor for the dense layer
        # Shape changes from (Batch, 64, 75, 75) to (Batch, 360000)
        x = x.view(x.size(0), -1)

        # Dense Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

# --- Example Usage ---
if __name__ == "__main__":
    # Initialize the model for binary classification (Ok vs Defective)
    model = CastingDefectCNN(num_classes=2)

    # Create a dummy input tensor (Batch size=4, 1 Channel Grayscale, 300x300 Image)
    # This matches the casting defect dataset specifications
    dummy_input = torch.randn(4, 1, 300, 300)
    # Forward pass
    output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}") # Should be [4, 2] for binary classification
    print(f"\nModel Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
