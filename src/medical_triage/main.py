from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import uvicorn
from fastapi import FastAPI

from medical_triage.api import router
from medical_triage.metrics import metrics_middleware

MODEL_PATH = Path("models/random_forest_pipeline.joblib")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = joblib.load(MODEL_PATH)
    yield  # aplicação roda aqui
    # aqui viria cleanup no shutdown, se necessário


app = FastAPI(title="Medical Triage API", lifespan=lifespan)

app.include_router(router)
app.middleware("http")(metrics_middleware)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
