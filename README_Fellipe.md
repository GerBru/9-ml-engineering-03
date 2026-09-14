# Notas do Fellipe

Anotações do que eu for fazendo na minha parte (deploy, API, CI/CD e monitoramento). German, usa isso pra acompanhar sem precisar vasculhar o Git.

## Etapa 1

### 1. Estratégia de deploy: tempo real

Para um modelo de NLP clássico e simples, sem GPU, dá para ir de serverless ou de serviço em tempo real. No serverless o cold start é curto e não prejudica a resposta. No tempo real a latência fica previsível e escala bem com volume.

Batch faz sentido para indexação inicial ou processamento em massa, quando não precisa de resposta na hora. No Tech Challenge a API tem que classificar o laudo na requisição, então eu vou de **deploy em tempo real**. Se o volume crescer muito no futuro, aí batch entra como complemento.

A escolha de nuvem (AWS, Azure ou GCP) eu deixo escrita no README oficial quando fechar a Etapa 1. Ainda não é o foco agora.

### 2. API FastAPI — esqueleto no ar

Subi o esqueleto da API no pacote `medical_triage`. Ainda **não classifica laudo de verdade**: o `/predict` devolve um valor fixo só para a rota existir e a gente conseguir testar o fluxo.

Arquivos:

- `src/medical_triage/main.py` — sobe o FastAPI e registra as rotas
- `src/medical_triage/api/routes.py` — endpoints


| Método | Rota       | O que faz hoje                                                                  |
| ------ | ---------- | ------------------------------------------------------------------------------- |
| `GET`  | `/health`  | Retorna `{"status": "ok"}` — serve para Docker/CI saberem que a API está viva   |
| `POST` | `/predict` | Retorna `{"prediction": "low"}` — stub, ainda não recebe texto nem chama modelo |


Para rodar:

```bash
uv sync
uv run python -m medical_triage.main
```

Docs interativas em `http://localhost:8000/docs`.

Quando o modelo existir, eu plugo em `services/`. Até lá o `/predict` continua mock.

### 3. Setup do projeto

Decidi usar `uv` **+** `pyproject.toml` **+** `uv.lock`, e não `requirements.txt` como fonte da verdade. O `pyproject.toml` declara o que o projeto é e o que ele precisa; o `uv.lock` trava as versões para a gente, o CI e o Docker instalarem a mesma coisa.

Python mínimo: **3.12** (arquivo `.python-version`).

Dependências de produção por enquanto: FastAPI e uvicorn. No grupo `dev`: Ruff.

Também deixei um `.gitignore` decente (venv, cache, `.env`, artefatos de teste) para a gente não commitar lixo.

### 4. Como eu estou pensando a pasta

```
src/medical_triage/
├── api/         ← rotas FastAPI (já existe)
├── schemas/     ← Pydantic request/response (ainda não)
├── services/    ← inferência do modelo (ainda não)
└── utils/

data/            ← CSV, dados brutos
models/          ← .joblib, .pkl, .onnx — não vão para o Git
monitoring/      ← Prometheus / Grafana (minha parte depois)
training/        ← seus scripts / Airflow
tests/
pyproject.toml
uv.lock
```



### 5. Docker e entregável — ainda não fiz

Falta empacotar a API em Docker e medir o tempo de resposta (baseline de latência local). O entregável da etapa é API funcional no Docker + o texto da decisão de nuvem no README.

## Próximo da minha parte

1. Contrato real do `/predict`: receber o texto do laudo e devolver classe + confiança
2. Docker + medição de latência
3. Texto da decisão de nuvem no README
4. Testes pytest da API

