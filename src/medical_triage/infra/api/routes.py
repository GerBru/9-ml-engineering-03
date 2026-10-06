from typing import Annotated

from fastapi import APIRouter, Depends, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
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

@router.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
