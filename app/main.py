import joblib
from pathlib import Path
import json
import logging

import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]


MODEL_NAME = "MP150_PowerLoom_Model"
MODEL_URI = f"models:/{MODEL_NAME}@champion"

FEATURES = [
    "loom_age_yr",
    "thread_count",
    "power_cuts_min",
    "weaver_exp_yr",
    "humidity_pct",
    "shift_hours",
    "fabric",
    "shed",
]


# --------------------------------------------------
# 2. JSON logging configuration
# --------------------------------------------------

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
        }

        event_data = getattr(record, "event_data", None)

        if isinstance(event_data, dict):
            log_entry.update(event_data)

        return json.dumps(log_entry)


logger = logging.getLogger("mp150_api")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)

logger.propagate = False


# --------------------------------------------------
# 3. Request schema
# --------------------------------------------------

class PredictionRequest(BaseModel):

    model_config = ConfigDict(extra="forbid")

    loom_age_yr: float = Field(
        le=55,
        description="Age of the loom in years",
    )

    thread_count: int = Field(
        ge=20,
        description="Threads per inch",
    )

    power_cuts_min: float = Field(
        ge=0,
        description="Power cuts during the shift in minutes",
    )

    weaver_exp_yr: float = Field(
        ge=0,
        description="Weaver experience in years",
    )

    humidity_pct: float = Field(
        ge=0,
        le=100,
        description="Humidity percentage",
    )

    shift_hours: float = Field(
        gt=0,
        description="Duration of the shift in hours",
    )

    fabric: str
    shed: str


# --------------------------------------------------
# 4. Initialize FastAPI
# --------------------------------------------------

app = FastAPI(
    title="MP150 Power Loom Output Prediction API",
    description="Predicts metres produced by a power loom per shift.",
    version="1.0",
)


# --------------------------------------------------
# 5. Load the current champion model
# --------------------------------------------------

model = None


@app.on_event("startup")
def load_model():
    global model

    model_path = PROJECT_ROOT / "models" / "model.joblib"
    model = joblib.load(model_path)

    logger.info("Champion model loaded successfully")

# --------------------------------------------------
# 6. Health endpoint
# --------------------------------------------------

@app.get("/health")
def health():

    return {"status": "ok"}


# --------------------------------------------------
# 7. Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(request: PredictionRequest):

    input_data = pd.DataFrame(
        [request.model_dump()],
        columns=FEATURES,
    )

    prediction = float(model.predict(input_data)[0])

    logger.info(
        "prediction_generated",
        extra={
            "event_data": {
                "event": "prediction",
                "prediction": prediction,
                "shed": request.shed,
                "fabric": request.fabric,
            }
        }
    )

    return {
        "prediction": prediction,
        "unit": "metres_per_shift",
    }
