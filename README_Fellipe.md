# Notas do Fellipe

Anotações do que eu for fazendo na minha parte (deploy, API, CI/CD e monitoramento). German, usa isso pra acompanhar sem precisar vasculhar o Git.

Organizei por **dia**. Cada data é o que eu fechei naquele dia; o estado atual é sempre a última entrada.

---

## 13/09/2026

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



## 14/09/2026

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



## 15/09/2026

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

O `TestClient` não sobe servidor HTTP. Ele chama a app ASGI em memória (`async(scope, receive, send)`). O `lifespan` (startup/shutdown) roda dentro do `with TestClient(app) as c:`.

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

