import logging
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from medical_triage.application.exceptions import ModelNotLoadedError

logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/random_forest_pipeline.joblib")
_model: Pipeline | None = None


def load_model() -> None:
    """Carrega o modelo do disco. Deve ser chamado na inicialização da aplicação."""
    global _model
    _model = joblib.load(MODEL_PATH)
    logger.info(f"Modelo carregado com sucesso: {MODEL_PATH}")


def predict(text: str) -> str:
    """Recebe um texto e retorna a classe predita."""
    if _model is None:
        raise ModelNotLoadedError("Chame load_model() antes de predizer.")
    return _model.predict([text])[0]
