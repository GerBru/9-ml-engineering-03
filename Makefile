# Atalhos para os comandos do README. Cada alvo só chama o comando completo,
# que continua documentado no README (caminho principal no Windows, onde o
# `make` não vem instalado). Rode `make` ou `make help` para ver a lista.

.PHONY: help install data train export-onnx benchmark test cov lint format \
	up down smoke load airflow-up airflow-down demo

.DEFAULT_GOAL := help

DATASET_URL := https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/main
DATASET := data/raw/medical_tc_train.csv data/raw/medical_tc_test.csv data/raw/medical_tc_labels.csv

help: ## Lista os alvos disponíveis
	@awk 'BEGIN {FS = ":.*## "} /^[a-z-]+:.*## / {printf "  %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ── Ambiente e dados ──────────────────────────────────────────────────────────

install: ## Instala as dependências (API, testes e treino)
	uv sync --group dev --group training

data: $(DATASET) ## Baixa os CSVs do repositório público para data/raw/

# Regra de arquivo: só baixa o CSV que ainda não existe (quem fez `dvc pull` não
# baixa de novo).
data/raw/%.csv:
	curl -fL -o $@ "$(DATASET_URL)/$*.csv"

# ── Treino e ONNX ─────────────────────────────────────────────────────────────

train: $(DATASET) ## Treina e publica o modelo (.joblib + .onnx) em models/
	uv run python -m training.pipeline

export-onnx: ## Exporta o .joblib publicado para ONNX, sem retreinar
	uv run python -m training.export_onnx

benchmark: $(DATASET) ## Compara a latência scikit-learn vs. ONNX
	uv run python -m training.benchmark_onnx

# ── Testes e qualidade ────────────────────────────────────────────────────────

test: ## Roda os testes
	uv run pytest

cov: ## Roda os testes com cobertura
	uv run pytest --cov=training --cov=src/medical_triage --cov-report=term-missing

lint: ## Lint e checagem de formatação (o mesmo que o CI roda)
	uv run ruff check .
	uv run ruff format --check .

format: ## Corrige lint e formatação automaticamente
	uv run ruff check --fix .
	uv run ruff format .

# ── API + monitoramento (Docker) ──────────────────────────────────────────────

up: ## Sobe API, Prometheus e Grafana
	docker compose up -d --build

down: ## Derruba API, Prometheus e Grafana
	docker compose down

smoke: ## Smoke test da API (/health e /predict)
	./scripts/smoke_test.sh

load: ## Teste de carga da API (precisa da ferramenta `hey`)
	./scripts/load_test.sh

# ── Airflow ───────────────────────────────────────────────────────────────────

# No Mac/Linux o Airflow precisa do UID do usuário em airflow/.env para os
# arquivos criados no volume não ficarem com dono root. No Windows não precisa.
airflow-up: $(DATASET) ## Sobe o Airflow em http://localhost:8080
ifneq ($(OS),Windows_NT)
	@test -f airflow/.env || echo "AIRFLOW_UID=$$(id -u)" > airflow/.env
endif
	cd airflow && docker compose up airflow-init
	cd airflow && docker compose up -d

airflow-down: ## Derruba o Airflow
	cd airflow && docker compose down

# ── Tudo de uma vez ───────────────────────────────────────────────────────────

demo: train up smoke ## Baixa os dados, treina, sobe a API e roda o smoke test
