"""Compara a latência de inferência do modelo original (scikit-learn) com o ONNX.

Mede o tempo de classificar UM laudo por vez, como numa requisição da API, e
grava o resultado em docs/latency_benchmark.json. Só a inferência: não inclui
HTTP, serialização nem rede (isso aparece no /metrics da API).

Execução (precisa de models/*.joblib, models/*.onnx e dos CSVs em data/raw/):
    uv run python -m training.benchmark_onnx
"""

import json
import logging
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import sklearn

from training import pipeline as training_pipeline
from training.export_onnx import predict_onnx
from training.ingest import load_data

logger = logging.getLogger("benchmark_onnx")

RESULT_PATH = Path(__file__).resolve().parents[1] / "docs" / "latency_benchmark.json"
N_REQUISICOES = 500
N_AQUECIMENTO = 20


def medir_latencia(prever, textos: list[str]) -> dict:
    """Latência (ms) de prever um texto por vez: média e percentis."""
    for texto in textos[:N_AQUECIMENTO]:
        prever(texto)
    tempos = []
    for texto in textos[:N_REQUISICOES]:
        inicio = time.perf_counter()
        prever(texto)
        tempos.append((time.perf_counter() - inicio) * 1000)
    return {
        "mean_ms": float(np.mean(tempos)),
        "p50_ms": float(np.percentile(tempos, 50)),
        "p95_ms": float(np.percentile(tempos, 95)),
        "p99_ms": float(np.percentile(tempos, 99)),
        "n": len(tempos),
    }


def main() -> None:
    config = training_pipeline.load_config()
    output = config["output"]
    models_dir = training_pipeline.MODELS_DIR

    pipeline_sklearn = joblib.load(models_dir / output["pipeline_filename"])
    sessao = ort.InferenceSession(str(models_dir / output["onnx_filename"]))
    _, test_df, _ = load_data()
    textos = test_df["medical_abstract"].tolist()

    classificador = pipeline_sklearn.steps[-1][1]
    n_jobs_publicado = classificador.n_jobs

    resultados = {}
    logger.info(f"Medindo scikit-learn como publicado (n_jobs={n_jobs_publicado})...")
    resultados["sklearn_publicado"] = medir_latencia(
        lambda t: pipeline_sklearn.predict([t]), textos
    )
    # Sem o pool de threads a cada predição: separa o custo do n_jobs=-1 do
    # custo do próprio scikit-learn, pra comparação com o ONNX ser justa.
    classificador.n_jobs = 1
    logger.info("Medindo scikit-learn com n_jobs=1...")
    resultados["sklearn_n_jobs_1"] = medir_latencia(
        lambda t: pipeline_sklearn.predict([t]), textos
    )
    logger.info("Medindo ONNX Runtime...")
    resultados["onnx_runtime"] = medir_latencia(
        lambda t: predict_onnx(sessao, [t]), textos
    )

    onnx_p50 = resultados["onnx_runtime"]["p50_ms"]
    relatorio = {
        "data": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ambiente": {
            "maquina": f"{platform.system()} {platform.machine()}",
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "onnxruntime": ort.__version__,
        },
        "n_jobs_publicado": n_jobs_publicado,
        "resultados": resultados,
        "speedup_p50_vs_publicado": resultados["sklearn_publicado"]["p50_ms"]
        / onnx_p50,
        "speedup_p50_vs_n_jobs_1": resultados["sklearn_n_jobs_1"]["p50_ms"] / onnx_p50,
    }
    RESULT_PATH.write_text(json.dumps(relatorio, indent=2, ensure_ascii=False))

    for nome, r in resultados.items():
        logger.info(
            f"{nome:18} média {r['mean_ms']:8.3f} ms | p50 {r['p50_ms']:8.3f} | "
            f"p95 {r['p95_ms']:8.3f} | p99 {r['p99_ms']:8.3f}"
        )
    logger.info(
        f"Speedup p50 do ONNX: {relatorio['speedup_p50_vs_publicado']:.0f}x vs "
        f"publicado | {relatorio['speedup_p50_vs_n_jobs_1']:.0f}x vs n_jobs=1"
    )
    logger.info(f"💾 Resultado salvo em {RESULT_PATH}")


if __name__ == "__main__":
    main()
