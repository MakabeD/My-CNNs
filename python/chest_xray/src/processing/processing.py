import os
import sys
from pathlib import Path

ROOT_PATH = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(ROOT_PATH))
from cnn_pipeline.src.pipeline.data_pipeline import (
    get_device,
    get_transforms,
    load_statistics,
)
# TODO: a class that imports pipeline functions (load statistics, get transforms, get device)
