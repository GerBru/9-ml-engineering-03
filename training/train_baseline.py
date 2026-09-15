"""
Treina um modelo baseline de classificação de laudos médicos.
Dataset: sintético, baseado nas 5 classes do Medical Abstracts TC Corpus.
Usado apenas para desenvolvimento da API enquanto o modelo real não está pronto.
"""
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

TEXTOS = [
    # neoplasms
    "tumor malignant cells cancer treatment chemotherapy biopsy",
    "breast cancer lymph node metastasis radiation therapy",
    "carcinoma tumor growth malignant neoplasm surgery",
    # digestive
    "gastric ulcer stomach pain digestive disorder inflammation",
    "liver cirrhosis hepatitis bile duct obstruction",
    "intestinal obstruction bowel disease colon inflammation",
    # nervous
    "brain stroke neurological deficit paralysis motor function",
    "epilepsy seizure neural activity cortex abnormal",
    "parkinson disease tremor dopamine neurodegenerative",
    # cardiovascular
    "heart failure myocardial infarction coronary artery disease",
    "hypertension blood pressure cardiac arrhythmia",
    "atherosclerosis vascular disease arterial stenosis",
    # general
    "infection bacterial fever inflammation immune response",
    "chronic pain fatigue general pathological condition",
    "metabolic disorder systemic disease multi-organ involvement",
]

LABELS = [
    "neoplasms", "neoplasms", "neoplasms",
    "digestive", "digestive", "digestive",
    "nervous", "nervous", "nervous",
    "cardiovascular", "cardiovascular", "cardiovascular",
    "general", "general", "general",
]

def treinar_modelo() -> Pipeline:
    """Treina e retorna o pipeline TF-IDF + Logistic Regression."""
    modelo = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=500)),
        ("clf", LogisticRegression(max_iter=200, random_state=42)),
    ])
    modelo.fit(TEXTOS, LABELS)
    return modelo


def salvar_modelo(modelo: Pipeline, caminho: str = "models/classifier.joblib") -> None:
    """Salva o modelo treinado em disco."""
    Path("models").mkdir(exist_ok=True)
    joblib.dump(modelo, caminho)
    print(f"Modelo salvo em: {caminho}")


if __name__ == "__main__":
    modelo = treinar_modelo()
    salvar_modelo(modelo)

    # Teste rápido para confirmar que funciona
    texto_teste = "patient with cardiac chest pain heart failure"
    predicao = modelo.predict([texto_teste])[0]
    print(f"Teste — entrada: '{texto_teste}'")
    print(f"Teste — classe predita: '{predicao}'")