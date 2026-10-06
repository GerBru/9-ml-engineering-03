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
- [x] Pipeline CI/CD básico com GitHub Actions (lint → test → build) (`.github/workflows/ci.yaml`) — o job de testes ainda falha nos 2 testes que dependem do dataset (DVC); CD fica fora do escopo
- [x] Script ou DAG Airflow simples para pipeline de treino/retreino (`airflow/dags/training_dag.py`)
- [ ] Stack de monitoramento local: API + Prometheus + Grafana via Docker Compose
- [x] Histórico de commits semântico e organizado

## Boas Práticas Obrigatórias

- [x] CI/CD com pelo menos 2 automações (lint + testes) — `ruff check`, `ruff format --check` e `pytest` a cada push/PR; 2 dos 22 testes falham no CI por dependerem do dataset (DVC)
- [ ] DAG Airflow funcional (ingestão → treino → salvamento do modelo) — DAG implementada (`ingest → validate → train → evaluate → deploy`, com quality gate); falta registrar uma execução completa no Airflow
- [ ] Dashboard Grafana com pelo menos 3 painéis
- [ ] Otimização de performance (ONNX, quantização ou pruning)

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

- [ ] API instrumentada com `prometheus_client` (tempo de requisição, contagem de chamadas)
- [ ] `docker-compose.yml` com API + Prometheus + Grafana
- [ ] Dashboard Grafana configurado e exibindo as métricas

## Etapa 4 — Otimização de Latência e Entrega

- [x] Classificador de texto treinado (TF-IDF + Random Forest)
- [ ] Técnica de otimização de latência aplicada (ex.: ONNX, quantização ou pruning)
- [ ] Comparação de latência: modelo original vs. otimizado
- [ ] Vídeo STAR gravado (≤ 5 minutos)

## Dataset

- [x] Dataset com ≥ 2.000 amostras — Medical Abstracts TC Corpus (11.550 treino + 2.888 teste)
