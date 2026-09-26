# Diário de bordo

Anotações do que cada um for fazendo (modelo de classificação). Usamos isso pra acompanhar sem precisar revisar o Git inteiro.

Organizado por **dia**, cada data é o que foi fechado por cada um naquele dia; o estado atual é sempre a última entrada.

Esse arquivo substitui ambos os arquivos **README_Fellipe.md** e **README_German.md** que foram apagados.

---

## 13/09/2026 - Fellipe

### Estratégia de deploy: tempo real

Para um modelo de NLP clássico e simples, sem GPU, dá para ir de serverless ou de serviço em tempo real. No serverless o cold start é curto e não prejudica a resposta. No tempo real a latência fica previsível e escala bem com volume.

Batch faz sentido para indexação inicial ou processamento em massa, quando não precisa de resposta na hora. No Tech Challenge a API tem que classificar o laudo na requisição, então eu vou de **deploy em tempo real**. Se o volume crescer muito no futuro, aí batch entra como complemento.

A escolha de nuvem (AWS, Azure ou GCP) eu deixo escrita no README oficial quando fechar a Etapa 1.

### API FastAPI — esqueleto

Subi o esqueleto da API no pacote `medical_triage`. O `/predict` ainda era stub: devolvia um valor fixo só para a rota existir.

Arquivos naquele dia:

- `src/medical_triage/main.py` — FastAPI e registro das rotas
- `src/medical_triage/api/routes.py` — endpoints


| Método | Rota       | O que fazia                                     |
| ------ | ---------- | ----------------------------------------------- |
| `GET`  | `/health`  | `{"status": "ok"}`                              |
| `POST` | `/predict` | `{"prediction": "low"}` — sem texto, sem modelo |


Para rodar:

```bash
uv sync
uv run python -m medical_triage.main
```

Docs interativas em `http://localhost:8000/docs`.

### Setup do projeto

Decidi usar `uv` **+** `pyproject.toml` **+** `uv.lock`, e não `requirements.txt` como fonte da verdade. O `pyproject.toml` declara o que o projeto é e o que ele precisa; o `uv.lock` trava as versões para a gente, o CI e o Docker instalarem a mesma coisa.

Python mínimo: **3.12** (arquivo `.python-version`). Dependências de produção: FastAPI e uvicorn. No grupo `dev`: Ruff.

Deixei um `.gitignore` decente (venv, cache, `.env`, artefatos de teste).

A pasta que eu tinha em mente naquele dia ainda era plana (`api/`, `schemas/`, `services/`). No dia seguinte isso muda.

Docker e o texto de decisão de nuvem no README oficial ficaram para depois.



---

## 14/09/2026 - Fellipe

Dois movimentos: um modelo baseline para a API ter o que inferir, e a reorganização em Clean Architecture com o `/predict` de verdade.

### Modelo baseline

Treinei um classificador **simples**, só para eu desenvolver a API enquanto o modelo real não está pronto. Não é o modelo da entrega.

- `training/train_baseline.py`
- `models/classifier.joblib`
- `joblib` e `scikit-learn` no `pyproject.toml`

Pipeline: **TF-IDF** (`max_features=500`) + **Regressão Logística**. Dataset sintético, 15 frases (3 por classe), nas 5 classes do Medical Abstracts TC Corpus: `neoplasms`, `digestive`, `nervous`, `cardiovascular`, `general`.

```bash
uv run python training/train_baseline.py
```

Quando o seu modelo estiver pronto, a gente troca o arquivo em `models/`.

### Clean Architecture

Adotei Clean Architecture com três camadas explícitas.

- **Domínio** — entidades Python puras, sem dependência de framework.
- **Aplicação** — ports (interfaces abstratas), use cases e exceptions.
- **Infraestrutura** (`infra/`) — implementações concretas: o adaptador HTTP (FastAPI) e o adaptador de ML (sklearn/joblib). Os dois ficam em `infra/` porque são detalhes externos.

A regra de dependência é unidirecional: infraestrutura depende de aplicação, aplicação depende de domínio. Nunca o contrário.

```
src/medical_triage/
├── domain/
├── application/
│   ├── ports.py
│   ├── use_cases.py
│   └── exceptions.py
├── infra/
│   ├── ml/
│   │   └── sklearn_predictor.py
│   └── api/
│       ├── dependencies.py
│       ├── routes.py
│       └── schemas.py
└── main.py
```



### Decisões técnicas

O modelo sklearn é carregado **uma única vez no startup**, via `lifespan` do FastAPI (`@asynccontextmanager`), e fica em `app.state.model`. Evita variável global e facilita teste. Nas rotas o acesso é por injeção de dependência com `Annotated[Pipeline, Depends(get_model)]`.

Os schemas Pydantic (`PredictRequest`, `PredictResponse`) ficam em `infra/api`, não no domínio, porque dependem do Pydantic. Cada um usa `ConfigDict` com `json_schema_extra` para o exemplo aparecer no Swagger, e docstring na classe.

Se o modelo não estiver no ar, quem responde é a dependência `get_model` com **HTTP 503**. A rota não carrega lógica de infraestrutura.

O `/predict` passou a classificar de verdade:


| Método | Rota       | O que faz agora                                                  |
| ------ | ---------- | ---------------------------------------------------------------- |
| `GET`  | `/health`  | `{"status": "ok"}`                                               |
| `POST` | `/predict` | Body `{"text": "..."}` → `{"label": "cardiovascular"}` (exemplo) |


Também deixei o Ruff configurado no `pyproject.toml` (`E`, `F`, `I`, `B`).

```bash
uv sync
uv run python -m medical_triage.main
```

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "patient with cardiac chest pain and heart failure"}'
```



---

## 15/09/2026 - Fellipe

Três entregas: Docker, medição de latência baseline e testes da API. Os testes vieram **antes** do CI/CD de propósito — o GitHub Actions vai rodá-los a cada push, então o pipeline precisa ter o que validar.

### Docker

`Dockerfile` na raiz, imagem `python:3.12-slim` (não Alpine: numpy/sklearn precisam de glibc). Instala com `uv sync --frozen --no-dev`: o lockfile trava as versões e o `--no-dev` deixa pytest/ruff/httpx2 fora da imagem de produção.

Camadas na ordem do que muda menos: `pyproject.toml` + `uv.lock` primeiro, depois `src/` e `models/`. Se só o código mudar, o Docker reusa o cache da instalação.

O predictor passou a apontar para o teu pipeline combinado (`models/random_forest_pipeline.joblib`). O `scikit-learn` ficou pinado em `==1.9.0` — joblib quebra se a versão do treino e da API divergirem. O `uv.lock` foi regenerado no PyPI público (saiu do índice interno) para o `docker build` funcionar em qualquer máquina.

Build e subida:

```bash
docker build -t medical-triage .
docker run --rm -p 8000:8000 medical-triage
```

A API fica em `http://localhost:8000`. Docs: `http://localhost:8000/docs`.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "patient with cardiac chest pain and heart failure"}'
```



### Latência baseline

`scripts/benchmark.sh` dispara 10 POSTs em `/predict` via `curl` e imprime média, mínimo e máximo. O container precisa estar no ar.

```bash
./scripts/benchmark.sh
```

Esse número é o baseline sklearn.

### Testes pytest

Stack: **pytest** + **httpx2** + `fastapi.testclient.TestClient`, no grupo `dev` (`uv add pytest httpx2 --dev`). Produção não leva isso (`uv sync --no-dev` no Docker).

O `TestClient` não sobe servidor HTTP. Ele chama a app ASGI[^1] em memória (`async(scope, receive, send)`). O `lifespan` (startup/shutdown) roda dentro do `with TestClient(app) as c:`.

Em vez de depender do `.joblib` real, uso `app.dependency_overrides`: o FastAPI troca `get_model()` por um `MagicMock` que devolve `["cardiovascular"]` sem invocar o sklearn.

A fixture `@pytest.fixture` com `yield` centraliza setup/teardown — configura o override, entrega o client, e limpa `dependency_overrides` no final (mesmo se o teste falhar).

Arquivo: `tests/test_api.py`. Cenários:


| Caso                                     | Esperado                        |
| ---------------------------------------- | ------------------------------- |
| `GET /health`                            | 200 e `{"status": "ok"}`        |
| `POST /predict` com payload válido       | 200 e campo `label`             |
| `POST /predict` sem o campo `text`       | 422 (validação Pydantic)        |
| `POST /predict` com modelo não carregado | 503 (proteção do `get_model()`) |


São 6 funções de teste cobrindo esses casos. Warnings do `httpx` antigo resolvidos migrando para `httpx2`.

Para rodar:

```bash
uv sync --group dev
uv run pytest
```

Só os testes da API:

```bash
uv run pytest tests/test_api.py -v
```



## Aberto

1. Texto da decisão de nuvem no README oficial
2. CI/CD GitHub Actions (lint + pytest a cada push)
3. Monitoramento Prometheus + Grafana (docker-compose)

[^1]: **Aclaración:** Una aplicación ASGI (por sus siglas en inglés, Asynchronous Server Gateway Interface) es un estándar y objeto ejecutable en Python que permite la comunicación asíncrona entre un servidor web y una aplicación o framework.



---

## 15/09/2026 - German

### Dataset real: Medical Abstracts TC Corpus

Comecei a treinar em cima do dataset completo. Os CSVs ficam **fora** deste repo.

5 classes, com desbalanceamento real (não é o dataset sintético equilibrado do baseline):

| Classe | Suporte no teste |
| --- | --- |
| neoplasms | 633 |
| digestive system diseases | 299 |
| nervous system diseases | 385 |
| cardiovascular diseases | 610 |
| general pathological conditions | 961 |

### TF-IDF + Random Forest: voltas que dei até chegar num resultado razoável:

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

### Empacotando num Pipeline (pro Fé não precisar mudar nada)

O `random_forest_pipeline.joblib` já contém o vetorizador (TF-IDF) e o classificador (RF) encadeados. Se precisar retreinar do zero por qualquer motivo, o dataset `Medical_Abstracts_TC_Corpus/` continua guardado fora do repo, então nada se perde.

**Fica assim, em `models/`:**
- `random_forest_pipeline.joblib` (~18.1 MB) — pronto pra API
- `classifier.joblib` — o do Fé, inalterado
- `random_forest_metrics.json` — métricas do treino

**Pro Fellipe:** dá pra trocar pro `random_forest_pipeline.joblib` com a mesma simplicidade de hoje. Em `sklearn_predictor.py`, é só mudar o `MODEL_PATH`:

```python
MODEL_PATH = Path("models/classifier.joblib")
# vira
MODEL_PATH = Path("models/random_forest_pipeline.joblib")
```

Nada mais muda no arquivo — o objeto carregado já é um `Pipeline` completo (`.predict([texto])[0]` retorna string, igual ao contrato atual). Não mexi no `classifier.joblib` nem no `sklearn_predictor.py`: a troca fica a critério teu.

### Ajuste no ambiente (bloqueava qualquer `uv sync`)

Pra instalar o `pandas` (grupo de dependência novo, `training`, só pra scripts de treino — não vai pra API), esbarrei em dois pins impossíveis que já estavam no `pyproject.toml`: `scikit-learn>=1.9.1` e `ruff>=0.16.7`. O índice interno do MELI só tem até `scikit-learn==1.9.0` e `ruff==0.16.5` — ou seja, não conseguia rodar `uv sync`/`uv add` nesse projeto antes desse fix. Baixei os dois pins pro que existe de fato no MELI, se isso atrapalhar pode mudar novamente eu me viro mais pra frente:

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

Precisa ter a pasta `Medical_Abstracts_TC_Corpus/` como irmã da tua pasta local (mesmo nível), com os 3 CSVs dentro.

### Nota de segurança: PAT exposto, push bloqueado

A URL do `origin` estava com um token do GitHub embutido em texto puro (salvo sem criptografia em `.git/config`); achei também outro token solto num `github_pat.txt` fora de qualquer repositório. Tirei o token da URL do remote, mas isso quer dizer que **o push vai falhar** até eu gerar um PAT novo e autenticar de novo. Pendência: revogar os dois tokens antigos no GitHub e gerar um novo antes do próximo push.


