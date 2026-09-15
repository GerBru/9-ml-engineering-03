from typing import Annotated

from fastapi import APIRouter, Depends
from sklearn.pipeline import Pipeline

from medical_triage.infra.api.dependencies import get_model
from medical_triage.infra.api.schemas import PredictRequest, PredictResponse

router = APIRouter()

ModelDep = Annotated[Pipeline, Depends(get_model)]

@router.get("/health")
def health_check() -> dict:
    return {"status": "ok"}

@router.post("/predict")
def predict(body: PredictRequest, model: ModelDep) -> PredictResponse:
    label = model.predict([body.text])[0]
    return PredictResponse(label=label)
