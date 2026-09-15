"""
Treina TF-IDF + Random Forest no dataset real (Medical Abstracts TC Corpus).

Etapa exploratória: vetorizador e classificador são treinados e salvos
SEPARADAMENTE (sem sklearn.pipeline.Pipeline). O empacotamento em Pipeline
fica para uma etapa posterior, depois de validar os dois componentes de
forma isolada.

Dataset esperado em (fora deste repo, pasta irmã de tech-challenge-03):
  ../../Medical_Abstracts_TC_Corpus/medical_tc_train.csv
  ../../Medical_Abstracts_TC_Corpus/medical_tc_test.csv
  ../../Medical_Abstracts_TC_Corpus/medical_tc_labels.csv
"""
import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score

DATA_DIR = Path(__file__).resolve().parents[2] / "Medical_Abstracts_TC_Corpus"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
CLASSIFIER_PATH = MODELS_DIR / "random_forest_classifier.joblib"
METRICS_PATH = MODELS_DIR / "random_forest_metrics.json"


def carregar_dados() -> tuple[pd.DataFrame, pd.DataFrame, dict[int, str]]:
    """Lê train/test/labels do Medical Abstracts TC Corpus."""
    train_df = pd.read_csv(DATA_DIR / "medical_tc_train.csv")
    test_df = pd.read_csv(DATA_DIR / "medical_tc_test.csv")
    labels_df = pd.read_csv(DATA_DIR / "medical_tc_labels.csv")
    label_map = dict(
        zip(labels_df["condition_label"], labels_df["condition_name"], strict=True)
    )
    return train_df, test_df, label_map


def treinar_vetorizador(textos_treino: pd.Series):
    """Ajusta o TfidfVectorizer nos textos de treino."""
    vetorizador = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 1),
        stop_words="english",
        min_df=2,
    )
    X_treino = vetorizador.fit_transform(textos_treino)
    return vetorizador, X_treino


def treinar_classificador(X_treino, y_treino) -> RandomForestClassifier:
    """Ajusta o RandomForestClassifier já com os textos vetorizados.

    max_depth/min_samples_leaf limitados de propósito: sem isso as árvores
    ficam gigantes em cima de um vetor esparso de milhares de features, e o
    .joblib passa de 200MB (acima do limite de arquivo do GitHub).
    """
    modelo = RandomForestClassifier(
        n_estimators=200,
        max_depth=25,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    modelo.fit(X_treino, y_treino)
    return modelo


def avaliar(
    modelo, vetorizador, test_df: pd.DataFrame, label_map: dict[int, str]
) -> dict:
    """Avalia no conjunto de teste e retorna métricas em dict."""
    X_teste = vetorizador.transform(test_df["medical_abstract"])
    y_teste = test_df["condition_label"]
    y_pred = modelo.predict(X_teste)

    acc = accuracy_score(y_teste, y_pred)
    f1_macro = f1_score(y_teste, y_pred, average="macro")
    report = classification_report(
        y_teste,
        y_pred,
        target_names=[label_map[k] for k in sorted(label_map)],
        output_dict=True,
    )
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "classification_report": report,
        "n_treino": None,  # preenchido no main
        "n_teste": len(test_df),
    }


def salvar_artefatos(
    vetorizador: TfidfVectorizer, modelo: RandomForestClassifier, metricas: dict
) -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(vetorizador, VECTORIZER_PATH)
    joblib.dump(modelo, CLASSIFIER_PATH)
    METRICS_PATH.write_text(json.dumps(metricas, indent=2, ensure_ascii=False))
    print(f"Vetorizador salvo em: {VECTORIZER_PATH}")
    print(f"Classificador salvo em: {CLASSIFIER_PATH}")
    print(f"Métricas salvas em: {METRICS_PATH}")


if __name__ == "__main__":
    inicio = time.time()

    print("Carregando dados...")
    train_df, test_df, label_map = carregar_dados()
    print(f"Treino: {len(train_df)} amostras | Teste: {len(test_df)} amostras")
    print(f"Classes: {label_map}")

    print("Treinando TfidfVectorizer...")
    vetorizador, X_treino = treinar_vetorizador(train_df["medical_abstract"])
    print(f"Vocabulário: {len(vetorizador.vocabulary_)} termos")

    print("Treinando RandomForestClassifier...")
    modelo = treinar_classificador(X_treino, train_df["condition_label"])

    print("Avaliando no conjunto de teste...")
    metricas = avaliar(modelo, vetorizador, test_df, label_map)
    metricas["n_treino"] = len(train_df)

    print(f"Accuracy: {metricas['accuracy']:.4f}")
    print(f"F1-macro: {metricas['f1_macro']:.4f}")

    salvar_artefatos(vetorizador, modelo, metricas)

    print(f"Tempo total: {time.time() - inicio:.1f}s")
