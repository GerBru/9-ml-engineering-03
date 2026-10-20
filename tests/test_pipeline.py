# tests/test_pipeline.py

import pytest

import training.pipeline as pipeline_module
from training.ingest import DATA_DIR

# Os CSVs ficam no DVC, fora do Git: no CI (sem dvc pull) eles não existem.
ARQUIVOS_DATASET = (
    "medical_tc_train.csv",
    "medical_tc_test.csv",
    "medical_tc_labels.csv",
)

requires_dataset = pytest.mark.skipif(
    not all((DATA_DIR / nome).exists() for nome in ARQUIVOS_DATASET),
    reason="dataset não está em data/raw/ (versionado no DVC: rode `uv run dvc pull`)",
)


@requires_dataset
def test_pipeline_end_to_end_roda_e_retorna_metricas(tmp_path, monkeypatch):
    """Integração: roda o pipeline completo com o dataset real, salvando
    os artefatos numa pasta temporária (não polui models/ de verdade)."""
    monkeypatch.setattr(pipeline_module, "MODELS_DIR", tmp_path)

    metricas = pipeline_module.run_pipeline()

    config = pipeline_module.load_config()
    assert (tmp_path / config["output"]["pipeline_filename"]).exists()
    assert (tmp_path / config["output"]["vectorizer_filename"]).exists()
    assert (tmp_path / config["output"]["classifier_filename"]).exists()
    assert (tmp_path / config["output"]["onnx_filename"]).exists()
    assert metricas["onnx_parity"] >= config["onnx"]["min_parity"]

    assert 0.0 <= metricas["accuracy"] <= 1.0
    assert 0.0 <= metricas["f1_macro"] <= 1.0
    assert metricas["n_treino"] > 0
    assert metricas["n_teste"] > 0
