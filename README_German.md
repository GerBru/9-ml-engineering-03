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

Por pedido explícito nesta etapa: vetorizador e classificador ficam **separados**, cada um serializado no seu próprio `.joblib`. Empacotamento em `sklearn.pipeline.Pipeline` ficou pra uma segunda etapa (ver abaixo). Este modelo **não substitui** o `models/classifier.joblib` que a API usa hoje; são artefatos novos, lado a lado — ainda não decidimos trocar (ver "Aberto").

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

### Empacotando num Pipeline (pro Fellipe não precisar mudar nada)

Perguntei pro Fellipe se o jeito atual (dois `.joblib` separados) ia dar pra ele continuar a parte dele do mesmo jeito que fazia com o `classifier.joblib`. Resposta: **não direto**. O `sklearn_predictor.py` dele faz `joblib.load(path).predict([texto])[0]` esperando um `Pipeline` único — se carregasse só o classificador, ia quebrar (`RandomForestClassifier.predict` espera a matriz TF-IDF já vetorizada, não texto cru); se carregasse só o vetorizador, nem tem `.predict()`.

Enquanto validava isso, achei um bug melhor resolver agora do que depois: eu tinha treinado o classificador com `condition_label` (o **código numérico**, 1 a 5), não com o nome da classe. Isso não ia quebrar (`RandomForestClassifier` treina normal com int como target), mas o `PredictResponse.label` da API é tipado como `str` — a predição ia sair como `"3"` em vez de `"nervous system diseases"`, tecnicamente "funcionando" mas com o significado errado. Corrigi o `train_tfidf_rf.py` pra treinar com `condition_label.map(label_map)` (a string do `medical_tc_labels.csv`), retreinei (métricas idênticas — é só relabeling) e conferi:

```python
pipeline.predict(["patient with cardiac chest pain and heart failure"])[0]
# 'cardiovascular diseases' (str)
```

Criei `training/build_pipeline.py` — não retreina nada, só carrega os dois `.joblib` já ajustados e monta:

```python
Pipeline([("tfidf", vetorizador), ("clf", modelo)])
```

- `training/build_pipeline.py`
- `models/random_forest_pipeline.joblib` (~18.1 MB)

```bash
uv run python training/build_pipeline.py
```

Continua **sem substituir** `models/classifier.joblib` — deixei os três arquivos lado a lado (`tfidf_vectorizer.joblib`, `random_forest_classifier.joblib` e agora `random_forest_pipeline.joblib`) até decidirmos junto quando fazer a troca de fato.

### Removendo os artefatos intermediários — pipeline pronto pro Fé usar

Decisão tomada: o `random_forest_pipeline.joblib` já contém o vetorizador e o classificador encadeados, então `tfidf_vectorizer.joblib` e `random_forest_classifier.joblib` não precisam mais ficar no repo — eram só o material bruto que o `build_pipeline.py` consumiu pra montar o pipeline final. Removi os dois (17MB + 373KB a menos no repo). Se precisar retreinar do zero por qualquer motivo, o dataset `Medical_Abstracts_TC_Corpus/` continua guardado fora do repo, então nada se perde.

**Fica assim, em `models/`:**
- `random_forest_pipeline.joblib` (~18.1 MB) — o novo, pronto pra API
- `classifier.joblib` — o do Fé, inalterado
- `random_forest_metrics.json` — métricas do treino, inalterado

**Pro Fellipe:** dá pra trocar pro `random_forest_pipeline.joblib` com a mesma simplicidade de hoje. Em `sklearn_predictor.py`, é só mudar o `MODEL_PATH`:

```python
MODEL_PATH = Path("models/classifier.joblib")
# vira
MODEL_PATH = Path("models/random_forest_pipeline.joblib")
```

Nada mais muda no arquivo — o objeto carregado já é um `Pipeline` completo (`.predict([texto])[0]` retorna string, igual ao contrato atual). Não mexi no `classifier.joblib` nem no `sklearn_predictor.py`: a troca fica a critério do Fé, no tempo dele.

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

1. Resolver o recall baixo de `general pathological conditions` (classe majoritária sendo penalizada demais pelo balanceamento)
2. ~~Decidir, com o Fellipe, se `random_forest_pipeline.joblib` troca o `models/classifier.joblib` da API~~ — decidido: o pipeline fica disponível, a troca do `MODEL_PATH` em `sklearn_predictor.py` fica com o Fé
3. ~~Rodar `ruff` nos scripts novos de treino~~ — feito, `ruff check training/` passa limpo
