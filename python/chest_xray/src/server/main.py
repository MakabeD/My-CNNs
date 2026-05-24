import asyncio
import io
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

sub_root = Path(__file__).parent.parent.parent
sys.path.append(str(sub_root))
from src.inference.predictor import ChestXrayPredictor

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    predictor = await loop.run_in_executor(
        None,
        lambda: ChestXrayPredictor(
            "models/chest-xray-own-best_model_0.2604.pt",
            "configs/xray-config1.yaml",
            "statistics.json",
        ),
    )
    app.state.predictor = predictor
    yield
    app.state.predictor = None


app = FastAPI(lifespan=lifespan)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("L")

        predictor = app.state.predictor
        result = predictor.predict_image(image)
        return result
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))
