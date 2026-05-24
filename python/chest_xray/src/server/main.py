import io
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from PIL import Image

sub_root = Path(__file__).parent.parent.parent
sys.path.append(str(sub_root))
from src.inference.predictor import ChestXrayPredictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = ChestXrayPredictor(
        "models/chest-xray-own-best_model_0.2604.pt",
        "configs/xray-config1.yaml",
        "statistics.json",
    )
    yield

    app.state.predictor = None


app = FastAPI(lifespan=lifespan)


@app.post("/predict")
async def siu(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    predictor = app.state.predictor

    result = predictor.predict_image(image)
    print(result)
    return result
