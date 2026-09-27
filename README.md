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
- [ ] Pipeline CI/CD básico com GitHub Actions (lint → test → build)
- [ ] Script ou DAG Airflow simples para pipeline de treino/retreino
- [ ] Stack de monitoramento local: API + Prometheus + Grafana via Docker Compose
- [x] Histórico de commits semântico e organizado

## Boas Práticas Obrigatórias

- [ ] CI/CD com pelo menos 2 automações (lint + testes) — testes já existem e passam localmente (`uv run pytest`, 17/17), falta o workflow do GitHub Actions rodando isso a cada push
- [ ] DAG Airflow funcional (ingestão → treino → salvamento do modelo)
- [ ] Dashboard Grafana com pelo menos 3 painéis
- [ ] Otimização de performance (ONNX, quantização ou pruning)

## Etapa 1 — Decisão Arquitetural e API Inicial

- [ ] Decisão de estratégia de deploy em nuvem documentada no README (AWS/Azure/GCP, batch vs. real-time)
- [x] API FastAPI que recebe texto do laudo e retorna a classificação
- [x] API empacotada em container Docker
- [x] Medição de tempo de resposta / baseline de latência local (`scripts/benchmark.sh`)

## Etapa 2 — CI/CD e Pipeline Automatizado

- [ ] Workflow GitHub Actions rodando lint + pytest a cada push
- [ ] DAG Airflow simulando o treino (task de leitura de CSV + task de treino/salvamento)
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
