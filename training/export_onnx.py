"""Exporta o Pipeline treinado (TF-IDF + Random Forest) para ONNX.

O ONNX Runtime executa o mesmo modelo bem mais rápido que o scikit-learn (é a
técnica de otimização de latência da Etapa 4 do Tech Challenge). Antes de
publicar, confiro que as predições do ONNX batem com as do scikit-learn.

Exportar o modelo que já está em models/, sem retreinar:
    uv run python -m training.export_onnx
"""

import logging

import numpy as np
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# Nome da entrada do modelo ONNX: um tensor de textos com formato [n, 1].
INPUT_NAME = "text"


def export_onnx(pipeline: Pipeline) -> bytes:
    """Converte o Pipeline do scikit-learn para ONNX e devolve os bytes do modelo.

    O Pipeline é convertido inteiro: o TF-IDF (tokenização, stop words e pesos)
    vira parte do grafo ONNX, então o modelo recebe o texto cru, igual à API.
    """
    classificador = pipeline.steps[-1][1]
    modelo_onnx = convert_sklearn(
        pipeline,
        initial_types=[(INPUT_NAME, StringTensorType([None, 1]))],
        # Sem zipmap a saída de probabilidades é um array, não uma lista de dicts.
        options={id(classificador): {"zipmap": False}},
    )
    return modelo_onnx.SerializeToString()


def predict_onnx(sessao: ort.InferenceSession, textos: list[str]) -> np.ndarray:
    """Classifica os textos com o modelo ONNX e devolve os rótulos (str)."""
    entrada = np.array(textos, dtype=object).reshape(-1, 1)
    return sessao.run(["label"], {INPUT_NAME: entrada})[0]


def check_parity(
    pipeline: Pipeline, onnx_bytes: bytes, textos: list[str], min_parity: float
) -> float:
    """Fração de textos em que ONNX e scikit-learn preveem a mesma classe.

    Raises:
        ValueError: se a paridade ficar abaixo de min_parity.
    """
    sessao = ort.InferenceSession(onnx_bytes)
    paridade = float(np.mean(pipeline.predict(textos) == predict_onnx(sessao, textos)))
    if paridade < min_parity:
        raise ValueError(
            f"ONNX reprovado: paridade {paridade:.4f} com o scikit-learn "
            f"< mínimo {min_parity}"
        )
    logger.info(
        f"✅ Paridade ONNX x scikit-learn: {paridade:.2%} ({len(textos)} textos)"
    )
    return paridade


def main() -> None:
    """Exporta o models/<pipeline_filename> atual, conferindo a paridade."""
    import joblib

    from training import pipeline as training_pipeline
    from training.ingest import load_data

    config = training_pipeline.load_config()
    output = config["output"]
    models_dir = training_pipeline.MODELS_DIR

    pipeline_sklearn = joblib.load(models_dir / output["pipeline_filename"])
    _, test_df, _ = load_data()

    onnx_bytes = export_onnx(pipeline_sklearn)
    check_parity(
        pipeline_sklearn,
        onnx_bytes,
        test_df["medical_abstract"].tolist(),
        config["onnx"]["min_parity"],
    )
    onnx_path = models_dir / output["onnx_filename"]
    onnx_path.write_bytes(onnx_bytes)
    logger.info(f"💾 Modelo ONNX salvo em {onnx_path}")


if __name__ == "__main__":
    main()
