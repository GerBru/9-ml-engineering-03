"""Avaliação do modelo no conjunto de teste."""

import logging

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score

logger = logging.getLogger(__name__)


def evaluate_model(
    vetorizador: TfidfVectorizer,
    modelo: RandomForestClassifier,
    test_df: pd.DataFrame,
    label_map: dict[int, str],
) -> dict:
    """Avalia o modelo no conjunto de teste e retorna métricas."""
    logger.info("Avaliando modelo...")

    X_teste = vetorizador.transform(test_df["medical_abstract"])
    y_teste = test_df["condition_label"].map(label_map)
    y_pred = modelo.predict(X_teste)

    metricas = {
        "accuracy": accuracy_score(y_teste, y_pred),
        "f1_macro": f1_score(y_teste, y_pred, average="macro"),
        "classification_report": classification_report(
            y_teste, y_pred, output_dict=True
        ),
        "n_treino": None,  # preenchido pelo pipeline
        "n_teste": len(test_df),
    }
    logger.info(
        f"✅ Accuracy: {metricas['accuracy']:.4f} | "
        f"F1-macro: {metricas['f1_macro']:.4f}"
    )
    return metricas


def check_quality_gate(metricas: dict, min_f1_macro: float) -> None:
    """Barra a publicação do modelo se o F1-macro ficar abaixo do mínimo.

    Raises:
        ValueError: se metricas["f1_macro"] < min_f1_macro.
    """
    f1_macro = metricas["f1_macro"]
    if f1_macro < min_f1_macro:
        raise ValueError(
            f"Quality gate reprovado: F1-macro {f1_macro:.4f} < mínimo {min_f1_macro}"
        )
    logger.info(f"✅ Quality gate aprovado: F1-macro {f1_macro:.4f} >= {min_f1_macro}")
