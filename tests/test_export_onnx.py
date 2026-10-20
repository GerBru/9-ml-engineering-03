# tests/test_export_onnx.py

import onnxruntime as ort
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from training.export_onnx import check_parity, export_onnx, predict_onnx
from training.train import train_model

LABEL_MAP = {1: "positivo", 2: "negativo"}
# Mesmos textos com os rótulos trocados: gera um modelo que prevê o oposto.
LABEL_MAP_TROCADO = {1: "negativo", 2: "positivo"}

TREINO = pd.DataFrame(
    {
        "condition_label": [1, 1, 1, 2, 2, 2],
        "medical_abstract": [
            "tumor cancer malignant cells",
            "cancer treatment chemotherapy tumor",
            "malignant tumor biopsy cancer",
            "healthy normal routine checkup",
            "routine exam normal healthy patient",
            "normal checkup healthy routine",
        ],
    }
)

TEXTOS = [
    "tumor cancer cells",
    "healthy normal patient",
    "malignant biopsy",
    "routine checkup",
]

CONFIG = {
    "tfidf": {
        "max_features": 50,
        "ngram_range": [1, 1],
        "stop_words": "english",
        "min_df": 1,
    },
    "model": {
        "n_estimators": 10,
        "max_depth": 5,
        "min_samples_leaf": 1,
        "class_weight": "balanced",
        "random_state": 42,
    },
}


def _pipeline(label_map: dict[int, str]) -> Pipeline:
    vetorizador, modelo = train_model(TREINO, label_map, CONFIG)
    return Pipeline([("tfidf", vetorizador), ("clf", modelo)])


# --- Testes de export_onnx / predict_onnx ---


def test_onnx_preve_igual_ao_sklearn():
    pipeline = _pipeline(LABEL_MAP)
    sessao = ort.InferenceSession(export_onnx(pipeline))

    assert list(predict_onnx(sessao, TEXTOS)) == list(pipeline.predict(TEXTOS))


def test_onnx_devolve_rotulos_em_texto():
    """A API espera a classe como string, igual ao predict do scikit-learn."""
    sessao = ort.InferenceSession(export_onnx(_pipeline(LABEL_MAP)))

    assert all(isinstance(rotulo, str) for rotulo in predict_onnx(sessao, TEXTOS))


# --- Testes de check_parity ---


def test_check_parity_aprova_modelo_identico():
    pipeline = _pipeline(LABEL_MAP)

    assert check_parity(pipeline, export_onnx(pipeline), TEXTOS, min_parity=0.99) == 1.0


def test_check_parity_reprova_quando_onnx_diverge():
    onnx_bytes = export_onnx(_pipeline(LABEL_MAP))
    pipeline_oposto = _pipeline(LABEL_MAP_TROCADO)

    with pytest.raises(ValueError, match="ONNX reprovado"):
        check_parity(pipeline_oposto, onnx_bytes, TEXTOS, min_parity=0.99)
