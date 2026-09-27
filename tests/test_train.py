# tests/test_train.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from training.evaluate import evaluate_model
from training.train import train_model

LABEL_MAP = {1: "positivo", 2: "negativo"}

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

TESTE = pd.DataFrame(
    {
        "condition_label": [1, 2],
        "medical_abstract": ["tumor cancer cells", "healthy normal patient"],
    }
)

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


# --- Testes de train ---


def test_train_model_retorna_vetorizador_e_classificador_ajustados():
    vetorizador, modelo = train_model(TREINO, LABEL_MAP, CONFIG)

    assert isinstance(vetorizador, TfidfVectorizer)
    assert isinstance(modelo, RandomForestClassifier)
    assert len(vetorizador.vocabulary_) > 0
    assert set(modelo.classes_) == {"positivo", "negativo"}


def test_train_model_respeita_hiperparametros_do_config():
    _, modelo = train_model(TREINO, LABEL_MAP, CONFIG)

    assert modelo.n_estimators == 10
    assert modelo.max_depth == 5


# --- Testes de evaluate ---


def test_evaluate_model_retorna_metricas_esperadas():
    vetorizador, modelo = train_model(TREINO, LABEL_MAP, CONFIG)

    metricas = evaluate_model(vetorizador, modelo, TESTE, LABEL_MAP)

    assert "accuracy" in metricas
    assert "f1_macro" in metricas
    assert 0.0 <= metricas["accuracy"] <= 1.0
    assert metricas["n_teste"] == len(TESTE)
