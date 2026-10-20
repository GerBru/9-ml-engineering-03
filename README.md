# Arquitetura

mlet-fase3/
├── src/
│   └── mlet_fase3/        ← o pacote Python
│       ├── __init__.py
│       ├── api/            ← rotas FastAPI
│       ├── schemas/        ← Pydantic (request/response)
│       ├── services/       ← lógica de inferência
│       └── utils/          ← código compartilhado
│
├── data/                  ← CSV, dados brutos 
├── models/                ← .joblib, .pkl
├── monitoring/            ← prometheus.yml, docker-compose
├── training/              ← scripts do German
│
├── tests/
├── pyproject.toml
└── uv.lock

# Checklist — Tech Challenge Fase 3

Baseado nos requisitos de `docs/Tech Challenge Fase 03.pdf`.

## Repositório GitHub (Requisitos Obrigatórios)

- [x] Modelo de classificação de texto (NLP): TF-IDF + Random Forest (`training/`, `models/random_forest_pipeline.joblib`)
- [x] API REST com FastAPI (`src/medical_triage/`)
- [x] Dockerfile funcional para o serviço de inferência
- [x] Pipeline CI/CD básico com GitHub Actions (lint → test → build) (`.github/workflows/ci.yaml`); CD fica fora do escopo
- [x] Script ou DAG Airflow simples para pipeline de treino/retreino (`airflow/dags/training_dag.py`)
- [ ] Stack de monitoramento local: API + Prometheus + Grafana via Docker Compose
- [x] Histórico de commits semântico e organizado

## Boas Práticas Obrigatórias

- [x] CI/CD com pelo menos 2 automações (lint + testes) — `ruff check`, `ruff format --check` e `pytest` a cada push/PR; os 2 testes que usam o dataset real são pulados no CI (dados no DVC, fora do Git)
- [x] DAG Airflow funcional (ingestão → treino → salvamento do modelo) — `ingest → validate → train → evaluate → deploy`, com quality gate; executada com sucesso no Airflow 2.10.4 via Docker Compose ([`docs/airflow_runs.jpeg`](docs/airflow_runs.jpeg))
- [ ] Dashboard Grafana com pelo menos 3 painéis
- [x] Otimização de performance (ONNX, quantização ou pruning) — Pipeline exportado para ONNX e executado com ONNX Runtime ([`docs/otimizacao_onnx.md`](docs/otimizacao_onnx.md))

## Etapa 1 — Decisão Arquitetural e API Inicial

- [ ] Decisão de estratégia de deploy em nuvem documentada no README (AWS/Azure/GCP, batch vs. real-time)
- [x] API FastAPI que recebe texto do laudo e retorna a classificação
- [x] API empacotada em container Docker
- [x] Medição de tempo de resposta / baseline de latência local (`scripts/benchmark.sh`)

## Etapa 2 — CI/CD e Pipeline Automatizado

- [x] Workflow GitHub Actions rodando lint + pytest a cada push (push em `develop`/`main` e em todo PR)
- [x] DAG Airflow simulando o treino (task de leitura de CSV + task de treino/salvamento) — `ingest_data` lê os CSVs; `train_model` e `deploy_model` treinam e salvam em `models/`
- [x] Pipeline de treino já modular, pronto para virar tasks da DAG (`training/ingest.py`, `validate.py`, `train.py`, `evaluate.py`, `pipeline.py`)

## Etapa 3 — Monitoramento e Observabilidade

- [x] API instrumentada com `prometheus_client` (tempo de requisição, contagem de chamadas) — `src/medical_triage/infra/api/metrics.py` (`http_requests_total`, `http_request_duration_seconds`), exposto em `/metrics`
- [ ] `docker-compose.yml` com API + Prometheus + Grafana
- [ ] Dashboard Grafana configurado e exibindo as métricas

## Etapa 4 — Otimização de Latência e Entrega

- [x] Classificador de texto treinado (TF-IDF + Random Forest)
- [x] Técnica de otimização de latência aplicada (ex.: ONNX, quantização ou pruning) — ONNX Runtime (`models/random_forest_pipeline.onnx`, 100% de paridade com o original no conjunto de teste)
- [x] Comparação de latência: modelo original vs. otimizado — p50 de 13,5 ms (scikit-learn) para 0,050 ms (ONNX) por laudo, só a inferência ([`docs/otimizacao_onnx.md`](docs/otimizacao_onnx.md))
- [ ] Vídeo STAR gravado (≤ 5 minutos)

## Dataset

- [x] Dataset com ≥ 2.000 amostras — Medical Abstracts TC Corpus (11.550 treino + 2.888 teste)
