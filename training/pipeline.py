"""Pipeline de treino end-to-end: ingest -> validate -> train -> evaluate -> save.

Execução:
    uv run python -m training.pipeline
"""

import json
import logging
import time
from pathlib import Path

import joblib
import yaml
from sklearn.pipeline import Pipeline

from training.evaluate import evaluate_model
from training.export_onnx import check_parity, export_onnx
from training.ingest import load_data
from training.train import train_model
from training.validate import validate_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("pipeline")

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def load_config() -> dict:
    """Carrega os hiperparâmetros e caminhos de saída do config.yaml."""
    return yaml.safe_load(CONFIG_PATH.read_text())


def save_artifacts(
    vetorizador, modelo, metricas: dict, config: dict, textos_teste: list[str]
) -> Path:
    """Salva vetorizador, classificador, Pipeline (.joblib e .onnx) e métricas.

    Usado tanto pelo run_pipeline() quanto pela task deploy_model da DAG do
    Airflow. Sempre sobrescreve os arquivos, então rodar de novo é seguro.

    O .onnx é gerado e conferido contra o scikit-learn (textos_teste) ANTES de
    gravar qualquer arquivo: se a paridade reprovar, nada em models/ muda.

    Returns:
        Caminho do Pipeline empacotado (o arquivo que a API carrega).
    """
    output = config["output"]

    # Empacota os dois componentes já treinados num Pipeline único, pronto
    # pro contrato que a API espera (joblib.load(path).predict([texto])).
    pipeline_sklearn = Pipeline([("tfidf", vetorizador), ("clf", modelo)])

    onnx_bytes = export_onnx(pipeline_sklearn)
    metricas["onnx_parity"] = check_parity(
        pipeline_sklearn, onnx_bytes, textos_teste, config["onnx"]["min_parity"]
    )

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(vetorizador, MODELS_DIR / output["vectorizer_filename"])
    joblib.dump(modelo, MODELS_DIR / output["classifier_filename"])
    pipeline_path = MODELS_DIR / output["pipeline_filename"]
    joblib.dump(pipeline_sklearn, pipeline_path)
    (MODELS_DIR / output["onnx_filename"]).write_bytes(onnx_bytes)

    (MODELS_DIR / output["metrics_filename"]).write_text(
        json.dumps(metricas, indent=2, ensure_ascii=False)
    )
    logger.info(f"💾 Artefatos salvos em {MODELS_DIR}")
    return pipeline_path


def run_pipeline() -> dict:
    """Executa o pipeline completo e retorna as métricas de avaliação."""
    inicio = time.time()
    config = load_config()
    logger.info("🚀 Iniciando pipeline de treino...")

    # 1. Ingest
    train_df, test_df, label_map = load_data()

    # 2. Validate
    validate_data(train_df, label_map)
    validate_data(test_df, label_map)

    # 3. Train
    vetorizador, modelo = train_model(train_df, label_map, config)

    # 4. Evaluate
    metricas = evaluate_model(vetorizador, modelo, test_df, label_map)
    metricas["n_treino"] = len(train_df)

    # 5. Deploy (salvar artefatos localmente)
    save_artifacts(
        vetorizador, modelo, metricas, config, test_df["medical_abstract"].tolist()
    )

    logger.info(
        f"🎉 Pipeline concluído em {time.time() - inicio:.1f}s | "
        f"Accuracy: {metricas['accuracy']:.4f} | F1-macro: {metricas['f1_macro']:.4f}"
    )
    return metricas


if __name__ == "__main__":
    run_pipeline()
