# dependencies.py
from fastapi import HTTPException, Request, status
from sklearn.pipeline import Pipeline


def get_model(request: Request) -> Pipeline:
    model = request.app.state.model
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo não disponível",
        )
    return model