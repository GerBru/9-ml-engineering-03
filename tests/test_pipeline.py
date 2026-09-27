# tests/test_pipeline.py

import training.pipeline as pipeline_module


def test_pipeline_end_to_end_roda_e_retorna_metricas(tmp_path, monkeypatch):
    """Integração: roda o pipeline completo com o dataset real, salvando
    os artefatos numa pasta temporária (não polui models/ de verdade)."""
    monkeypatch.setattr(pipeline_module, "MODELS_DIR", tmp_path)

    metricas = pipeline_module.run_pipeline()

    config = pipeline_module.load_config()
    assert (tmp_path / config["output"]["pipeline_filename"]).exists()
    assert (tmp_path / config["output"]["vectorizer_filename"]).exists()
    assert (tmp_path / config["output"]["classifier_filename"]).exists()

    assert 0.0 <= metricas["accuracy"] <= 1.0
    assert 0.0 <= metricas["f1_macro"] <= 1.0
    assert metricas["n_treino"] > 0
    assert metricas["n_teste"] > 0
