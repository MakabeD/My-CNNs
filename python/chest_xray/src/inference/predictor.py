import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image

ROOT_PATH = Path(__file__).resolve().parents[4]
PYTHON_ROOT = ROOT_PATH / "python"

if str(PYTHON_ROOT) not in sys.path:
    sys.path.append(str(PYTHON_ROOT))

from chest_xray.src.processing.processing import InferenceProcessing  # noqa: E402
from cnn_pipeline.src.model.model import get_model  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ChestXrayPredictor:
    """Loads a trained model and runs single-image or batch-style inference."""

    def __init__(
        self,
        model_path: str | Path,
        config_path: str | Path,
        statistics_path: str | Path | None = None,
        class_names: list[str] | None = None,
    ) -> None:
        self.processing = InferenceProcessing(
            config_path=config_path,
            statistics_path=statistics_path,
        )
        self.model_path = self._resolve_path(model_path)
        self.device = self.processing.device
        self.transform = self.processing.get_inference_transform()
        self.class_names = class_names or self.processing.class_names
        self.model = self._load_model()

    def _resolve_path(self, path: str | Path) -> Path:
        path_obj = Path(path)
        if path_obj.is_absolute():
            return path_obj
        return (self.processing.project_root / path_obj).resolve()

    def _load_model(self) -> torch.nn.Module:
        config = self.processing.config
        model = get_model(
            model_name=config.model.name,
            num_classes=config.model.num_classes,
            pretrained=config.model.pretrained,
            dropout=config.model.dropout,
            img_size=config.data.img_size,
        )

        checkpoint = torch.load(self.model_path, map_location=self.device)
        state_dict = checkpoint.get("model_state", checkpoint)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        logger.info("Loaded model from %s", self.model_path)
        return model

    def preprocess_image(self, image_path: str | Path) -> torch.Tensor:
        image = Image.open(image_path).convert("L")
        tensor = self.transform(image).unsqueeze(0)
        return tensor.to(self.device)

    @torch.inference_mode()
    def predict_tensor(self, tensor: torch.Tensor) -> dict[str, Any]:
        logits = self.model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        predicted_index = int(torch.argmax(probabilities).item())

        scores = {
            self.class_names[index]
            if index < len(self.class_names)
            else str(index): float(probabilities[index].item())
            for index in range(probabilities.shape[0])
        }

        predicted_label = (
            self.class_names[predicted_index]
            if predicted_index < len(self.class_names)
            else str(predicted_index)
        )

        return {
            "predicted_index": predicted_index,
            "predicted_label": predicted_label,
            "confidence": float(probabilities[predicted_index].item()),
            "scores": scores,
        }

    def predict_image(self, image_path: str | Path) -> dict[str, Any]:
        image_path = Path(image_path).resolve()
        tensor = self.preprocess_image(image_path)
        prediction = self.predict_tensor(tensor)
        prediction["image_path"] = str(image_path)
        return prediction

    def predict_images(self, image_paths: list[str | Path]) -> list[dict[str, Any]]:
        return [self.predict_image(image_path) for image_path in image_paths]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run chest X-ray model inference.")
    parser.add_argument(
        "image_paths",
        nargs="*",
        help="One or more image paths to classify.",
    )
    parser.add_argument(
        "--model-path",
        default="models/chest-xray-own-best_model_0.2604.pt",
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--config",
        default="configs/xray-config1.yaml",
        help="Path to the inference/training config file.",
    )
    parser.add_argument(
        "--statistics",
        default="./statistics.json",
        help="Optional path to the statistics JSON file.",
    )
    parser.add_argument(
        "--class-names",
        nargs="*",
        default=["anomaly", "normal"],
        help="Optional class names to override config/statistics classes.",
    )
    parser.add_argument(
        "--images",
        nargs="+",
        dest="images_flag",
        default=None,
        help="Explicit image paths to classify.",
    )
    return parser


def resolve_cli_images(args: argparse.Namespace) -> list[str]:
    if args.images_flag:
        return args.images_flag

    image_paths = list(args.image_paths)
    if image_paths and image_paths[0].lower() == "images":
        image_paths = image_paths[1:]

    if not image_paths:
        raise SystemExit(
            "No image paths were provided. Pass them positionally or with --images."
        )

    return image_paths


def main() -> None:
    args = build_parser().parse_args()
    image_paths = resolve_cli_images(args)
    predictor = ChestXrayPredictor(
        model_path=args.model_path,
        config_path=args.config,
        statistics_path=args.statistics,
        class_names=args.class_names,
    )

    for prediction in predictor.predict_images(image_paths):
        logger.info("Image: %s", prediction["image_path"])
        logger.info(
            "Prediction: %s (confidence: %.4f)",
            prediction["predicted_label"],
            prediction["confidence"],
        )
        logger.info("Scores: %s", prediction["scores"])


if __name__ == "__main__":
    main()
