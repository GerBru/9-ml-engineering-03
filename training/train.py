"""Treina TF-IDF + RandomForestClassifier (sem Pipeline) no Medical Abstracts."""

import logging

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


def train_model(
    train_df: pd.DataFrame, label_map: dict[int, str], config: dict
) -> tuple[TfidfVectorizer, RandomForestClassifier]:
    """Ajusta o vetorizador e o classificador nos dados de treino.

    max_depth/min_samples_leaf limitados de propósito: sem isso as árvores
    ficam gigantes em cima de um vetor esparso de milhares de features, e o
    .joblib passa de 200MB (acima do limite de arquivo do GitHub).
    """
    tfidf_cfg = config["tfidf"]
    model_cfg = config["model"]

    logger.info("Treinando TfidfVectorizer...")
    vetorizador = TfidfVectorizer(
        max_features=tfidf_cfg["max_features"],
        ngram_range=tuple(tfidf_cfg["ngram_range"]),
        stop_words=tfidf_cfg["stop_words"],
        min_df=tfidf_cfg["min_df"],
    )
    X_treino = vetorizador.fit_transform(train_df["medical_abstract"])
    logger.info(f"Vocabulário: {len(vetorizador.vocabulary_)} termos")

    logger.info("Treinando RandomForestClassifier...")
    y_treino = train_df["condition_label"].map(label_map)
    modelo = RandomForestClassifier(
        n_estimators=model_cfg["n_estimators"],
        max_depth=model_cfg["max_depth"],
        min_samples_leaf=model_cfg["min_samples_leaf"],
        class_weight=model_cfg["class_weight"],
        random_state=model_cfg["random_state"],
        n_jobs=-1,
    )
    modelo.fit(X_treino, y_treino)
    logger.info(f"✅ Modelo treinado em {len(train_df)} amostras")

    return vetorizador, modelo
