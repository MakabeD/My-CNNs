import logging
import sys
from pathlib import Path
from typing import Any

ROOT_PATH = Path(__file__).resolve().parents[4]
PYTHON_ROOT = ROOT_PATH / "python"

if str(PYTHON_ROOT) not in sys.path:
    sys.path.append(str(PYTHON_ROOT))

from cnn_pipeline.src.pipeline.data_pipeline import (  # noqa: E402
    get_device,
    get_transforms,
    load_statistics,
)
from cnn_pipeline.src.utils.config import Config, load_config  # noqa: E402

logger = logging.getLogger(__name__)


class InferenceProcessing:
    """Utility class for chest X-ray inference preprocessing and runtime setup."""

    def __init__(
        self,
        config_path: str | Path,
        statistics_path: str | Path | None = None,
    ) -> None:
        self.project_root = ROOT_PATH / "python" / "chest_xray"
        self.config_path = self._resolve_path(config_path)
        self.config = self._load_config()
        self.statistics_path = self._resolve_statistics_path(statistics_path)
        self.statistics = self._load_statistics()
        self.device = get_device()

    def _resolve_path(self, path: str | Path) -> Path:
        path_obj = Path(path)
        if path_obj.is_absolute():
            return path_obj
        return (self.project_root / path_obj).resolve()

    def _load_config(self) -> Config:
        return load_config(str(self.config_path))

    def _resolve_statistics_path(self, statistics_path: str | Path | None) -> Path:
        if statistics_path is not None:
            return self._resolve_path(statistics_path)

        config_stats_path = self.config.data.config_path
        if config_stats_path:
            return self._resolve_path(config_stats_path)

        return (self.project_root / "statistics.json").resolve()

    def _load_statistics(self) -> dict[str, Any]:
        return load_statistics(self.statistics_path)

    @property
    def mean(self) -> float:
        return float(self.statistics["mean"])

    @property
    def std(self) -> float:
        return float(self.statistics["std"])

    @property
    def class_names(self) -> list[str]:
        classes = self.statistics.get("classes")
        if classes:
            return list(classes)
        return [str(index) for index in range(self.config.model.num_classes)]

    def get_inference_transform(self):
        return get_transforms(
            mean=self.mean,
            std=self.std,
            img_size=self.config.data.img_size,
            train=False,
            three_gray_channels=self.config.data.three_gray_channels,
        )
