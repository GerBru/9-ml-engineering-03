"""
Empacota o vetorizador + classificador (já treinados por train_tfidf_rf.py)
num sklearn.pipeline.Pipeline único, pronto pro contrato que a API espera
(joblib.load(path).predict([texto])[0] -> str).

Não retreina nada: só carrega os dois artefatos já ajustados e os encadeia.
"""
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
CLASSIFIER_PATH = MODELS_DIR / "random_forest_classifier.joblib"
PIPELINE_PATH = MODELS_DIR / "random_forest_pipeline.joblib"


def montar_pipeline() -> Pipeline:
    vetorizador = joblib.load(VECTORIZER_PATH)
    modelo = joblib.load(CLASSIFIER_PATH)
    return Pipeline([("tfidf", vetorizador), ("clf", modelo)])


if __name__ == "__main__":
    pipeline = montar_pipeline()
    joblib.dump(pipeline, PIPELINE_PATH)
    print(f"Pipeline salvo em: {PIPELINE_PATH}")

    texto_teste = "patient with cardiac chest pain and heart failure"
    predicao = pipeline.predict([texto_teste])[0]
    print(f"Teste — entrada: '{texto_teste}'")
    print(f"Teste — classe predita: '{predicao}' (tipo: {type(predicao).__name__})")
