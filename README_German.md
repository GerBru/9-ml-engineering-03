# Notas do German

Anotações do que eu for fazendo na minha parte (modelo de classificação). Fellipe, usa isso pra acompanhar sem precisar vasculhar o Git.

Organizei por **dia**, igual você fez no seu. Cada data é o que eu fechei naquele dia; o estado atual é sempre a última entrada.

---

## 15/09/2026

### Dataset real: Medical Abstracts TC Corpus

Comecei a treinar em cima do dataset de verdade, não mais nas 15 frases sintéticas do `train_baseline.py`. Os CSVs ficam **fora** deste repo, numa pasta irmã (`FASE 3 - Cloud and MLOps/Medical_Abstracts_TC_Corpus/`), então o script busca o caminho relativo ao próprio arquivo (`Path(__file__).resolve().parents[2]`) em vez de depender do diretório de onde ele é chamado.

- `medical_tc_train.csv` — 11.550 amostras
- `medical_tc_test.csv` — 2.888 amostras
- `medical_tc_labels.csv` — mapa `condition_label` → `condition_name`

5 classes, com desbalanceamento real (não é o dataset sintético equilibrado do baseline):

| Classe | Suporte no teste |
| --- | --- |
| neoplasms | 633 |
| digestive system diseases | 299 |
| nervous system diseases | 385 |
| cardiovascular diseases | 610 |
| general pathological conditions | 961 |

### TF-IDF + Random Forest — sem Pipeline ainda

Por pedido explícito nesta etapa: vetorizador e classificador ficam **separados**, cada um serializado no seu próprio `.joblib`. Ainda não empacotei em `sklearn.pipeline.Pipeline` — isso fica pra depois, quando eu (ou a gente) decidir a interface final de inferência. Por isso este modelo **não substitui** o `models/classifier.joblib` que a API usa hoje; são artefatos novos, lado a lado.

- `training/train_tfidf_rf.py`
- `models/tfidf_vectorizer.joblib` (~373 KB)
- `models/random_forest_classifier.joblib` (~16.9 MB)
- `models/random_forest_metrics.json` (métricas do teste, pra não perder o número depois)

```bash
uv run python training/train_tfidf_rf.py
```

**Duas voltas que dei até chegar num resultado razoável:**

1. **Tamanho do arquivo estourou.** Primeira tentativa (`max_features=20000`, bigramas, `RandomForestClassifier(n_estimators=300, max_depth=None)`) gerou um `.joblib` de **217 MB** — acima do limite de 100MB do GitHub, o push ia falhar. Cortei pra `max_features=10000`, só unigramas, `n_estimators=200`, `max_depth=25`, `min_samples_leaf=2`. Ficou em ~17MB.
2. **Desbalanceamento mascarando o resultado.** Com hiperparâmetros mais curtos, o accuracy subiu (0.49) mas o F1-macro caiu pra 0.38 — o modelo estava só chutando a classe majoritária (`general pathological conditions`, 961 de 2888). Adicionei `class_weight="balanced"` e o F1-macro subiu pra **0.54** sem crescer o arquivo.

**Métricas finais (conjunto de teste, 2888 amostras):**

| Métrica | Valor |
| --- | --- |
| Accuracy | 0.549 |
| F1-macro | 0.541 |

| Classe | Precision | Recall | F1 |
| --- | --- | --- | --- |
| neoplasms | 0.66 | 0.76 | 0.71 |
| digestive system diseases | 0.44 | 0.70 | 0.54 |
| nervous system diseases | 0.42 | 0.73 | 0.54 |
| cardiovascular diseases | 0.62 | 0.78 | 0.69 |
| general pathological conditions | 0.53 | **0.15** | 0.23 |

**Limitação que já vi e ainda não resolvi:** o `class_weight="balanced"` super-corrigiu pra classe majoritária — o recall dela despencou pra 0.15 (o modelo praticamente parou de prever essa classe pra acertar mais as minoritárias). Compensa no F1-macro agregado, mas não é um resultado equilibrado de verdade. Próxima iteração: testar `class_weight="balanced_subsample"`, threshold por classe, ou balancear via oversampling/undersampling antes do TF-IDF.

### Ajuste no ambiente (bloqueava qualquer `uv sync`)

Pra instalar o `pandas` (grupo de dependência novo, `training`, só pra scripts de treino — não vai pra API), esbarrei em dois pins impossíveis que já estavam no `pyproject.toml`: `scikit-learn>=1.9.1` e `ruff>=0.16.7`. O índice interno da MELI (`pypi.artifacts.furycloud.io`) só tem até `scikit-learn==1.9.0` e `ruff==0.16.5` — ou seja, **ninguém** conseguia rodar `uv sync`/`uv add` nesse projeto antes desse fix, não é coisa que eu quebrei agora. Baixei os dois pins pro que existe de fato:

```diff
- "scikit-learn>=1.9.1",
+ "scikit-learn>=1.9.0",
...
- dev = ["ruff>=0.16.7"]
+ dev = ["ruff>=0.16.5"]
+ training = ["pandas>=3.0.5"]
```

### Como reproduzir

```bash
uv sync --group training
uv run python training/train_tfidf_rf.py
```

Precisa ter a pasta `Medical_Abstracts_TC_Corpus/` como irmã de `tech-challenge-03/` (mesmo nível), com os 3 CSVs dentro.

## Aberto

1. Empacotar `tfidf_vectorizer.joblib` + `random_forest_classifier.joblib` num `Pipeline` único
2. Resolver o recall baixo de `general pathological conditions` (classe majoritária sendo penalizada demais pelo balanceamento)
3. Decidir se este modelo troca o `models/classifier.joblib` da API ou se convive com ele até comparar os dois
4. Rodar `ruff` nos scripts novos de treino
