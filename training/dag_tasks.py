"""Tasks do pipeline de treino, no formato que a DAG do Airflow orquestra.

Cada função vira uma task da DAG (airflow/dags/training_dag.py). A lógica fica
aqui, e não no arquivo da DAG, pra poder ser testada sem o Airflow instalado.

Entre uma task e outra só passam caminhos e números: o retorno de cada função
vira XCom, que fica gravado no banco de metadados do Airflow. DataFrames e
modelos são gravados numa pasta de staging (data/processed/<run_id>/) e as
tasks seguintes leem de lá.
"""

import re
from pathlib import Path

import joblib

from training import pipeline
from training.evaluate import check_quality_gate, evaluate_model
from training.ingest import load_data
from training.train import train_model
from training.validate import validate_data

STAGING_ROOT = Path(__file__).resolve().parents[1] / "data" / "processed"

TRAIN_FILE = "train_df.joblib"
TEST_FILE = "test_df.joblib"
LABEL_MAP_FILE = "label_map.joblib"
VECTORIZER_FILE = "vectorizer.joblib"
CLASSIFIER_FILE = "classifier.joblib"
METRICS_FILE = "metrics.joblib"


def _staging_dir(run_id: str) -> Path:
    """Pasta de staging de uma execução.

    O run_id do Airflow tem ':' e '+' (ex.: manual__2026-10-05T12:00:00+00:00),
    que não são válidos em nome de pasta no Windows, então troco por '_'.
    """
    return STAGING_ROOT / re.sub(r"[^A-Za-z0-9_.-]", "_", run_id)


def ingest_data(run_id: str) -> str:
    """Lê os CSVs e grava train/test/label_map na pasta de staging."""
    train_df, test_df, label_map = load_data()

    staging = _staging_dir(run_id)
    staging.mkdir(parents=True, exist_ok=True)
    joblib.dump(train_df, staging / TRAIN_FILE)
    joblib.dump(test_df, staging / TEST_FILE)
    # label_map não vai pelo XCom: o JSON do XCom transformaria as chaves int em str.
    joblib.dump(label_map, staging / LABEL_MAP_FILE)
    return str(staging)


def validate_staged_data(staging_dir: str) -> str:
    """Valida train e test já gravados. Falha a task se algo estiver errado."""
    staging = Path(staging_dir)
    label_map = joblib.load(staging / LABEL_MAP_FILE)
    validate_data(joblib.load(staging / TRAIN_FILE), label_map)
    validate_data(joblib.load(staging / TEST_FILE), label_map)
    return staging_dir


def train_staged_model(staging_dir: str) -> str:
    """Treina TF-IDF + Random Forest e grava os dois na pasta de staging."""
    staging = Path(staging_dir)
    vetorizador, modelo = train_model(
        joblib.load(staging / TRAIN_FILE),
        joblib.load(staging / LABEL_MAP_FILE),
        pipeline.load_config(),
    )
    joblib.dump(vetorizador, staging / VECTORIZER_FILE)
    joblib.dump(modelo, staging / CLASSIFIER_FILE)
    return staging_dir


def evaluate_staged_model(staging_dir: str) -> dict:
    """Avalia no conjunto de teste e aplica o quality gate do config.yaml.

    Returns:
        staging_dir e as métricas principais (só números, cabem no XCom).

    Raises:
        ValueError: se o F1-macro ficar abaixo de quality_gate.min_f1_macro.
    """
    staging = Path(staging_dir)
    train_df = joblib.load(staging / TRAIN_FILE)
    metricas = evaluate_model(
        joblib.load(staging / VECTORIZER_FILE),
        joblib.load(staging / CLASSIFIER_FILE),
        joblib.load(staging / TEST_FILE),
        joblib.load(staging / LABEL_MAP_FILE),
    )
    metricas["n_treino"] = len(train_df)
    joblib.dump(metricas, staging / METRICS_FILE)

    config = pipeline.load_config()
    check_quality_gate(metricas, config["quality_gate"]["min_f1_macro"])

    return {
        "staging_dir": staging_dir,
        "accuracy": float(metricas["accuracy"]),
        "f1_macro": float(metricas["f1_macro"]),
    }


def deploy_model(resultado: dict) -> str:
    """Publica o modelo aprovado em models/ (sobrescreve o anterior)."""
    staging = Path(resultado["staging_dir"])
    pipeline_path = pipeline.save_artifacts(
        joblib.load(staging / VECTORIZER_FILE),
        joblib.load(staging / CLASSIFIER_FILE),
        joblib.load(staging / METRICS_FILE),
        pipeline.load_config(),
    )
    return str(pipeline_path)
