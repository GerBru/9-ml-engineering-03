from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sklearn.pipeline import Pipeline

from medical_triage.schemas import PredictRequest, PredictResponse

router = APIRouter()


def get_model(request: Request) -> Pipeline:
    """Modelo carregado no startup. Responde 503 se não estiver disponível."""
    model = request.app.state.model
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo não disponível",
        )
    return model


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
