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



---

## 26/09/2026 - German

### Refatoração pra pipeline modular

O `training/` só tinha um script monolítico (`train_tfidf_rf.py`) fazendo tudo — carregar CSV, vetorizar, treinar, avaliar e salvar numa função só. O material de apoio da Aula 01 da Etapa 3 ("Introdução ao Pipeline de ML") ensina o padrão `ingest → validate → train → evaluate`, orquestrado por um `pipeline.py`, com config externo e testes por módulo. Não é só teoria: é pré-requisito direto pra Etapa 2 do desafio (Airflow), que pede uma task pra ler dados e outra pra treinar/salvar — precisa de funções separadas por estágio pra isso fazer sentido numa DAG.

**Removidos** (substituídos pelo `pipeline.py`): `training/train_tfidf_rf.py`, `training/build_pipeline.py`.

**Novos arquivos em `training/`:**

- `ingest.py` — `load_data()`: lê train/test/labels do Medical Abstracts TC Corpus.
- `validate.py` — `validate_data()`: checa dataset não vazio, colunas obrigatórias, sem nulos, labels conhecidos, pelo menos 2 classes. Levanta `AssertionError` com mensagem específica se algo falhar.
- `train.py` — `train_model()`: ajusta TF-IDF + RandomForest (mesmos hiperparâmetros de antes).
- `evaluate.py` — `evaluate_model()`: accuracy, F1-macro, classification report.
- `pipeline.py` — orquestra os 4 acima + salva vetorizador, classificador e o `Pipeline` empacotado, com logging em vez de `print`.
- `config.yaml` — hiperparâmetros e nomes de arquivo de saída, fora do código (facilita reaproveitar na DAG do Airflow depois).
- `__init__.py` — vira pacote Python de verdade, pra import funcionar tanto direto (`python -m training.pipeline`) quanto nos testes.

```bash
uv sync --group training
uv run python -m training.pipeline
```

Rodei de novo pra conferir que o refactor não mudou o resultado: **accuracy 0.5492, F1-macro 0.5413** — idêntico ao treino anterior.

**Testes novos** (`tests/test_data.py`, `tests/test_train.py`, `tests/test_pipeline.py`): cobrem ingest, todas as validações de `validate_data`, treino/avaliação com dataset sintético pequeno (rápido), e um teste de integração end-to-end que roda o pipeline completo com o dataset real, salvando os artefatos numa pasta temporária (via `monkeypatch.setattr` no `MODELS_DIR`) — não polui o `models/` de verdade. **17/17 testes passam** (os 6 do Fé continuam intactos).

**Ajustes no `pyproject.toml`:**

```diff
- "httpx2>=2.13.0",
+ "httpx2>=2.12.0",
```

```toml
[tool.uv]
environments = [
    "sys_platform == 'darwin'",
    "sys_platform == 'linux'",
]
```

Os dois eram pins/resoluções impossíveis pré-existentes que travavam `uv sync` pra qualquer pessoa: `httpx2==2.13.0` não existe no índice, e o `uv` por padrão tenta resolver dependências pra **todas** as plataformas (inclusive Windows + Python 3.14, que ninguém usa aqui), travando porque `scikit-learn==1.9.0` não existe pra essa combinação. Restringi a resolução a `darwin`/`linux`.

Também adicionei `[tool.pytest.ini_options] pythonpath = ["."]` — necessário pra `from training.ingest import ...` funcionar nos testes.



---

## 27/09/2026 - German

### Checklist do Tech Challenge no README

Adicionei uma seção de checklist no `README.md` oficial, baseada linha por linha no `docs/Tech Challenge Fase 03.pdf` (Requisitos Obrigatórios, Boas Práticas, Etapas 1–4, Dataset). Marquei o que já está pronto e o que falta, cruzando com o estado real do repo (não com suposição): hoje não existe `.github/workflows`, `docker-compose.yml`, stack de Prometheus/Grafana, DAG do Airflow nem nada de ONNX/quantização neste repo ainda — só a parte de modelagem (treino + API + Docker + latência baseline) está encaminhada.

### Cobertura de testes (`pytest-cov`)

Configurei medição de cobertura, **sob demanda** (decidimos não deixar automático em todo `pytest`, pra não poluir/lentificar o dia a dia). Único arquivo alterado: `pyproject.toml`.

- Adicionei `pytest-cov` no grupo `dev`.
- Adicionei `[tool.coverage.run]` (`source = ["training", "src/medical_triage"]`, ignora `__init__.py`) e `[tool.coverage.report]` (`show_missing = true`, mostra as linhas exatas não cobertas).

**Importante:** cobertura **não roda sozinha** com `uv run pytest`. Precisa passar as flags na mão:

```bash
uv sync --group dev --group training
uv run pytest --cov=training --cov=src/medical_triage --cov-report=term-missing
```

Resultado da primeira medição: **83% de cobertura total**, 17/17 testes passando. `training/ingest.py`, `validate.py`, `train.py`, `evaluate.py` em 100%; `pipeline.py` em 97%; `train_baseline.py` (script do Fellipe, não testado por ninguém) em 0% — esperado, ninguém importa esse arquivo em teste nenhum.



---

## 27/09/2026 - German

### DVC pra versionar o dataset

Branch separada: `feature/dvc-dataset-gdrive` (não fizemos direto na `develop` dessa vez).

O `training/ingest.py` lia o Medical Abstracts TC Corpus de uma pasta **fora do repo** — funcionava na minha máquina, mas quebrava pra qualquer outra pessoa que clonasse do zero. Resolvi com DVC + Google Drive como remote, mesmo padrão que a gente já usava na Fase 2.

**O que fiz:**

1. `uv add --group training "dvc[gdrive]"` — adiciona o DVC com suporte a Google Drive.
2. `dvc init` — cria `.dvc/config` e `.dvc/.gitignore`.
3. `dvc remote add -d gdrive_storage gdrive://17OqAGPyzgOxeQql4clhwiZanTweB9glT` — aponta pro Drive que compartilhei com o Fé (edição).
4. Copiei os 3 CSVs pra dentro do repo, em `data/raw/`, e rodei `dvc add` neles — isso cria um `.dvc` (ponteiro leve, com hash) pra cada um e um `.gitignore` que impede o CSV pesado de ir pro Git.
5. Atualizei `training/ingest.py`: `DATA_DIR` agora aponta pra `data/raw/` dentro do repo, não mais pra pasta externa.
6. Rodei o pipeline e os 17 testes de novo — mesmas métricas de sempre (accuracy 0.5492, F1-macro 0.5413), confirma que a troca de caminho não quebrou nada.

**Bloqueio no meu lado:** o acesso ao DVC/Google Drive está bloqueado pela política de rede da empresa nesta máquina. Já abri chamado de liberação, mas vai demorar.

### Pro Fellipe: como subir os 3 arquivos pro Drive

Como meu acesso está bloqueado, combinamos que **você** faz o primeiro `dvc push`, direto da sua máquina (já te dei acesso de edição na pasta do Drive). **Importante: não dá pra simplesmente arrastar os CSVs pro Drive pelo navegador** — o DVC guarda o conteúdo lá organizado por hash, não como arquivo com nome normal. Precisa ser pelo comando mesmo.

Passo a passo:

1. Pega os 3 arquivos por fora do Git — eles nunca foram commitados, estão no `.gitignore` do DVC:
   - `medical_tc_labels.csv`
   - `medical_tc_test.csv`
   - `medical_tc_train.csv`
2. Depois de puxar a `develop`, coloca os 3 arquivos exatamente em `data/raw/`, respeitando esses nomes.
3. Roda:
   ```bash
   uv sync --group training
   uv run dvc push
   ```
   Como o conteúdo é idêntico byte a byte ao que já está nos `.dvc` commitados, o hash bate automaticamente — o DVC reconhece sem conflito e sobe pro remote `gdrive_storage`. Na primeira vez deve abrir uma autenticação OAuth pelo navegador, com a tua conta Google (a mesma que tem acesso de edição na pasta).

Depois disso, qualquer um (eu incluso, quando o TI liberar) consegue rodar `dvc pull` e recuperar os dados normalmente.
