# Otimização de latência: modelo exportado para ONNX

Etapa 4 do Tech Challenge: aplicar uma técnica de otimização de latência e comparar o modelo original com o otimizado.

## Técnica escolhida

O Pipeline treinado (TF-IDF + Random Forest) é exportado para **ONNX** e executado com o **ONNX Runtime**, em vez do scikit-learn.

Por que ONNX e não quantização: a quantização vista no curso (Etapa 6, Aula 04 — Pruning e Quantização) reduz a precisão de **pesos e ativações de redes neurais**. Um Random Forest não tem pesos: é um conjunto de árvores de decisão com limiares. Quantizar exigiria trocar o modelo. O ONNX se aplica ao modelo atual sem retreinar (referência do curso: `Material_complementar/06-latencia-performance/aula01-quantizacao-onnx/`).

## Como o modelo ONNX é gerado

- O Pipeline inteiro é convertido com `skl2onnx` (`training/export_onnx.py`): a tokenização, as stop words e o TF-IDF viram parte do grafo ONNX. O modelo recebe o **texto cru** e devolve a **classe em texto**, o mesmo contrato do `.joblib` que a API usa.
- Entrada: `text` (tensor de strings `[n, 1]`). Saídas: `label` (classe) e `probabilities` (5 classes).
- Arquivo: `models/random_forest_pipeline.onnx` (13,9 MB; o `.joblib` tem 22,5 MB).
- Gerado automaticamente a cada treino, pelo `pipeline.py` e pela task `deploy_model` da DAG do Airflow. Para exportar o modelo já publicado sem retreinar: `uv run python -m training.export_onnx`.

**Paridade com o original.** Antes de publicar, as predições do ONNX são comparadas com as do scikit-learn no conjunto de teste. Se baterem em menos de 99% (`onnx.min_parity` no `config.yaml`), nada em `models/` é gravado. Resultado atual: **100% (2.888 de 2.888 laudos)**, com accuracy (0.5637) e F1-macro (0.5575) idênticos.

## Resultados

Latência de classificar **um laudo por vez** (como numa requisição da API), 500 laudos do conjunto de teste, após 20 de aquecimento. Mede só a inferência (sem HTTP). Script: `training/benchmark_onnx.py`; dados brutos: `docs/latency_benchmark.json`.

| Variante | Média | p50 | p95 | p99 |
| --- | --- | --- | --- | --- |
| scikit-learn, como publicado (`n_jobs=-1`) | 13,209 ms | 13,504 ms | 13,903 ms | 14,686 ms |
| scikit-learn, `n_jobs=1` | 5,511 ms | 5,485 ms | 5,715 ms | 5,821 ms |
| **ONNX Runtime** | **0,051 ms** | **0,050 ms** | **0,069 ms** | **0,079 ms** |

**Ganho no p50: 268 vezes** em relação ao modelo publicado e **109 vezes** em relação ao scikit-learn com `n_jobs=1`.

Ambiente da medição: macOS (Apple Silicon, arm64), Python 3.12.2, scikit-learn 1.9.0, onnxruntime 1.30.0. Os valores absolutos dependem da máquina; a proporção entre as variantes é o que importa.

## Leitura dos resultados

- **Por que a linha `n_jobs=1`:** o Random Forest publicado usa `n_jobs=-1`, que abre um pool de threads a cada predição. Para um laudo só, esse custo domina: sozinho, ele faz o scikit-learn passar de ~5,5 ms para ~13,5 ms. Mostrar as duas linhas separa o ganho do ONNX do ganho de só trocar o `n_jobs`.
- **De onde vem o ganho do ONNX:** o ONNX Runtime executa a tokenização, o TF-IDF e as 289 árvores em C++ otimizado, sem a sobrecarga do Python e do scikit-learn a cada chamada.
- **A cauda também caiu:** o p99 do ONNX (0,079 ms) é menor que o melhor p50 do scikit-learn.

## Reproduzir

```bash
uv sync --group training
uv run python -m training.export_onnx      # gera models/random_forest_pipeline.onnx
uv run python -m training.benchmark_onnx   # mede e grava docs/latency_benchmark.json
```

Precisa dos CSVs em `data/raw/` (`uv run dvc pull`).

## Próximo passo

A API ainda serve o `.joblib`. Para o ganho chegar ao `/predict` (e aparecer no Grafana), a API precisa carregar o `.onnx` com o `onnxruntime`. É uma decisão com o Fellipe (ver `binnacle.md`).
