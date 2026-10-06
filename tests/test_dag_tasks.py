# tests/test_dag_tasks.py

import json

import pandas as pd
import pytest

from training import dag_tasks, pipeline
from training.evaluate import check_quality_gate

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


def _config(min_f1_macro: float) -> dict:
    return {
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
        "quality_gate": {"min_f1_macro": min_f1_macro},
        "output": {
            "vectorizer_filename": "tfidf_vectorizer.joblib",
            "classifier_filename": "random_forest_classifier.joblib",
            "pipeline_filename": "random_forest_pipeline.joblib",
            "metrics_filename": "random_forest_metrics.json",
        },
    }


# --- Fixture: isola dados, staging e models/ numa pasta temporária ---
@pytest.fixture
def ambiente(tmp_path, monkeypatch):
    monkeypatch.setattr(
        dag_tasks, "load_data", lambda: (TREINO.copy(), TESTE.copy(), LABEL_MAP)
    )
    monkeypatch.setattr(dag_tasks, "STAGING_ROOT", tmp_path / "processed")
    monkeypatch.setattr(pipeline, "MODELS_DIR", tmp_path / "models")
    return tmp_path


def test_staging_dir_remove_caracteres_invalidos_no_windows():
    staging = dag_tasks._staging_dir("manual__2026-10-05T12:00:00.123+00:00")

    assert ":" not in staging.name
    assert "+" not in staging.name


def test_tasks_encadeadas_publicam_o_modelo(ambiente, monkeypatch):
    monkeypatch.setattr(pipeline, "load_config", lambda: _config(min_f1_macro=0.0))

    staging_dir = dag_tasks.ingest_data(run_id="manual__teste")
    staging_dir = dag_tasks.validate_staged_data(staging_dir)
    staging_dir = dag_tasks.train_staged_model(staging_dir)
    resultado = dag_tasks.evaluate_staged_model(staging_dir)
    pipeline_path = dag_tasks.deploy_model(resultado)

    assert pipeline_path.endswith("random_forest_pipeline.joblib")
    assert (ambiente / "models" / "random_forest_pipeline.joblib").exists()
    assert (ambiente / "models" / "random_forest_metrics.json").exists()


def test_retornos_das_tasks_cabem_no_xcom(ambiente, monkeypatch):
    """O XCom serializa em JSON: os retornos não podem ter DataFrame nem modelo."""
    monkeypatch.setattr(pipeline, "load_config", lambda: _config(min_f1_macro=0.0))

    staging_dir = dag_tasks.ingest_data(run_id="manual__teste")
    dag_tasks.train_staged_model(staging_dir)
    resultado = dag_tasks.evaluate_staged_model(staging_dir)

    json.dumps(staging_dir)
    json.dumps(resultado)


def test_quality_gate_reprovado_nao_publica_modelo(ambiente, monkeypatch):
    monkeypatch.setattr(pipeline, "load_config", lambda: _config(min_f1_macro=1.01))

    staging_dir = dag_tasks.ingest_data(run_id="manual__teste")
    dag_tasks.train_staged_model(staging_dir)

    with pytest.raises(ValueError, match="Quality gate reprovado"):
        dag_tasks.evaluate_staged_model(staging_dir)
    assert not (ambiente / "models").exists()


def test_check_quality_gate_aprova_no_limite():
    check_quality_gate({"f1_macro": 0.5}, min_f1_macro=0.5)  # não deve levantar
