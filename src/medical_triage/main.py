from contextlib import asynccontextmanager

import joblib
import uvicorn
from fastapi import FastAPI

from medical_triage.infra.api.routes import router
from medical_triage.infra.ml.sklearn_predictor import MODEL_PATH


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = joblib.load(MODEL_PATH)
    yield  # aplicação roda aqui
    # aqui viria cleanup no shutdown, se necessário

app = FastAPI(title="Medical Triage API", lifespan=lifespan)

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
