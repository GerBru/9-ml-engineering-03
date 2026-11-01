"""Busca de hiperparâmetros do RandomForestClassifier via RandomizedSearchCV.

Lê o dataset diretamente da pasta externa Medical_Abstracts_TC_Corpus/ (não
usa data/raw/ nem o remote do DVC): o acesso ao Google Drive está bloqueado
nesta máquina pela política de rede da empresa, então este script evita
depender do dvc pull.

Execução:
    uv run python -m training.tuning
"""

import json
import logging
from pathlib import Path

import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("tuning")

DATA_DIR = Path(__file__).resolve().parents[2] / "Medical_Abstracts_TC_Corpus"
RESULT_PATH = Path(__file__).resolve().parent / "tuning_result.json"


def carregar_dados_treino() -> tuple[pd.Series, pd.Series]:
    """Lê só o conjunto de treino (a busca já faz sua própria validação cruzada)."""
    train_df = pd.read_csv(DATA_DIR / "medical_tc_train.csv")
    labels_df = pd.read_csv(DATA_DIR / "medical_tc_labels.csv")
    label_map = dict(
        zip(labels_df["condition_label"], labels_df["condition_name"], strict=True)
    )
    y = train_df["condition_label"].map(label_map)
    return train_df["medical_abstract"], y


def buscar_hiperparametros(
    n_iter: int = 20, cv: int = 5, random_state: int = 42
) -> tuple[dict, float]:
    """Roda RandomizedSearchCV sobre o RandomForestClassifier.

    O espaço de busca é limitado de propósito (max_depth <= 30,
    min_samples_leaf >= 1) pro mesmo motivo do treino original: árvores sem
    limite de profundidade em cima de um vetor TF-IDF esparso geram um
    .joblib gigante (>200MB, acima do limite do GitHub).
    """
    logger.info("Carregando dados de treino de %s", DATA_DIR)
    textos, y = carregar_dados_treino()
    logger.info("Treino: %d amostras", len(textos))

    logger.info("Vetorizando com TfidfVectorizer...")
    vetorizador = TfidfVectorizer(
        max_features=10000, ngram_range=(1, 1), stop_words="english", min_df=2
    )
    X = vetorizador.fit_transform(textos)

    param_dist = {
        "n_estimators": randint(100, 300),
        "max_depth": randint(15, 30),
        "min_samples_leaf": randint(1, 4),
    }

    search = RandomizedSearchCV(
        estimator=RandomForestClassifier(
            class_weight="balanced", random_state=random_state, n_jobs=-1
        ),
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="f1_macro",
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )
    logger.info("Rodando RandomizedSearchCV (n_iter=%d, cv=%d)...", n_iter, cv)
    search.fit(X, y)

    logger.info("Melhores hiperparâmetros: %s", search.best_params_)
    logger.info(
        "Melhor F1-macro (média da validação cruzada): %.4f", search.best_score_
    )

    resultado = {
        "best_params": search.best_params_,
        "best_cv_f1_macro": search.best_score_,
        "n_iter": n_iter,
        "cv": cv,
    }
    RESULT_PATH.write_text(json.dumps(resultado, indent=2, ensure_ascii=False))
    logger.info("Resultado salvo em %s", RESULT_PATH)

    return search.best_params_, search.best_score_


if __name__ == "__main__":
    buscar_hiperparametros()
