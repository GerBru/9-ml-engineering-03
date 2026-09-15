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

## Aberto

1. Docker + medição de latência (baseline)
2. Texto da decisão de nuvem no README oficial
3. Testes pytest da API
4. Trocar o `.joblib` sintético pelo modelo que você treinar

