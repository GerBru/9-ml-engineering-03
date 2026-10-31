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

| Classe                          | Suporte no teste |
| ------------------------------- | ---------------- |
| neoplasms                       | 633              |
| digestive system diseases       | 299              |
| nervous system diseases         | 385              |
| cardiovascular diseases         | 610              |
| general pathological conditions | 961              |

### TF-IDF + Random Forest: voltas que dei até chegar num resultado razoável:

1. **Tamanho do arquivo estourou.** Primeira tentativa (`max_features=20000`, bigramas, `RandomForestClassifier(n_estimators=300, max_depth=None)`) gerou um `.joblib` de **217 MB** — acima do limite de 100MB do GitHub, o push ia falhar. Cortei pra `max_features=10000`, só unigramas, `n_estimators=200`, `max_depth=25`, `min_samples_leaf=2`. Ficou em ~17MB.
2. **Desbalanceamento mascarando o resultado.** Com hiperparâmetros mais curtos, o accuracy subiu (0.49) mas o F1-macro caiu pra 0.38 — o modelo estava só chutando a classe majoritária (`general pathological conditions`, 961 de 2888). Adicionei `class_weight="balanced"` e o F1-macro subiu pra **0.54** sem crescer o arquivo.

**Métricas finais (conjunto de teste, 2888 amostras):**

| Métrica  | Valor |
| -------- | ----- |
| Accuracy | 0.549 |
| F1-macro | 0.541 |

| Classe                          | Precision | Recall   | F1   |
| ------------------------------- | --------- | -------- | ---- |
| neoplasms                       | 0.66      | 0.76     | 0.71 |
| digestive system diseases       | 0.44      | 0.70     | 0.54 |
| nervous system diseases         | 0.42      | 0.73     | 0.54 |
| cardiovascular diseases         | 0.62      | 0.78     | 0.69 |
| general pathological conditions | 0.53      | **0.15** | 0.23 |

**Limitação que já vi e ainda não resolvi:** o `class_weight="balanced"` super-corrigiu pra classe majoritária — o recall dela despencou pra 0.15 (o modelo praticamente parou de prever essa classe pra acertar mais as minoritárias). Compensa no F1-macro agregado, mas não é um resultado equilibrado de verdade. Próxima iteração: testar `class_weight="balanced_subsample"`, threshold por classe, ou balancear via oversampling/undersampling antes do TF-IDF.

### Empacotando num Pipeline (pro Fé não precisar mudar nada)

O `random_forest_pipeline.joblib` já contém o vetorizador (TF-IDF) e o classificador (RF) encadeados. Se precisar retreinar do zero por qualquer motivo, o dataset `Medical_Abstracts_TC_Corpus/` continua guardado fora do repo, então nada se perde.

**Fica assim, em** `models/`**:**

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

**Novos arquivos em** `training/`**:**

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

**Ajustes no** `pyproject.toml`**:**

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

---

## 28/09/2026 - German

### Tuning de hiperparâmetros (RandomizedSearchCV)

Branch separada: `feature/tuning-random-forest`. Vinha da Aula 03 da Etapa 3 (Treinamento de Modelos e Validação) — não é requisito do Tech Challenge, mas os hiperparâmetros do Random Forest até agora tinham sido escolhidos no olho (documentei isso lá atrás, quando resolvi o problema do tamanho do arquivo), nunca por busca sistemática.

`training/tuning.py` (novo): `RandomizedSearchCV` com 20 combinações × 5 folds, otimizando `f1_macro`. Espaço de busca limitado de propósito (`max_depth` 15–30, `min_samples_leaf` 1–4, `n_estimators` 100–300) pelo mesmo motivo de sempre — árvore sem limite de profundidade em vetor TF-IDF esparso gera `.joblib` gigante.

**Importante:** lê os CSVs direto de `Medical_Abstracts_TC_Corpus/` (pasta externa), **não** de `data/raw/` (DVC) — meu acesso ao remote do Google Drive segue bloqueado pela política de rede da empresa, então evitei qualquer dependência do `dvc pull` pra rodar essa tarefa.

**Resultado:** `max_depth=28, min_samples_leaf=3, n_estimators=289` — F1-macro de 0.5656 na validação cruzada. Apliquei no `config.yaml` e retreinei o pipeline completo:

| Métrica               | Antes  | Depois                                         |
| --------------------- | ------ | ---------------------------------------------- |
| Accuracy (teste real) | 0.549  | **0.564**                                      |
| F1-macro (teste real) | 0.541  | **0.558**                                      |
| Tamanho do `.joblib`  | ~18 MB | ~22.5 MB (ainda bem abaixo do limite de 100MB) |

O `random_forest_pipeline.joblib` foi sobrescrito com o modelo novo — mesma estrutura de sempre (`Pipeline` com `tfidf` + `clf` juntos), então o Fellipe não precisa mudar nada no `sklearn_predictor.py`: mesmo nome de arquivo, mesmo contrato (`predict([texto])[0]` retorna string). 17/17 testes continuam passando.

Descoberta de lado, sem ação por enquanto: `models/tfidf_vectorizer.joblib` e `models/random_forest_classifier.joblib` (os artefatos "redundantes" que a gente tinha decidido não versionar) acabaram indo pra `develop` mesmo assim, provavelmente num `git add .` durante o PR do DVC. Não atrapalha nada, só ficou registrado.

---

## 03/10/2026 - German

Destravei o `uv sync` do Fé e fiz o primeiro `dvc push`. As duas coisas eram bloqueadas pela rede da empresa, então fiz de um computador pessoal com Windows.

### `uv.lock` sem o índice interno da MELI

O `uv sync` do Fé dava 403 porque o `uv.lock`, gerado na minha máquina do trabalho, gravava `pypi.artifacts.furycloud.io` como origem de cada pacote (1.370 referências). Apaguei o lock e gerei de novo pelo PyPI público (`a9d2c0d`). Só rodar `uv lock` não basta: sem mudança no `pyproject.toml`, o uv reaproveita o lock antigo.

Também adicionei `sys_platform == 'win32'` em `[tool.uv] environments` (`a3fbe55`), senão o `uv sync` recusa rodar no Windows.

### Primeiro `dvc push`

Os MD5 dos três CSVs batem com os `.dvc` commitados, então foi só `dvc add` (não altera os `.dvc`) e `dvc push`.

O Google bloqueia o login padrão do DVC ("Se bloqueó esta app"). Contorno: app OAuth próprio no Google Cloud (`fiap-tech-challenge`, público Externo, modo teste), com a Google Drive API ativada, eu e o Fé como usuários de teste (em **Público → Usuarios de prueba**) e um cliente do tipo **App de escritorio**. Cada um configura na própria máquina:

```bash
uv run dvc remote modify --local gdrive_storage gdrive_client_id "<id>"
uv run dvc remote modify --local gdrive_storage gdrive_client_secret "<secret>"
```

O aviso "Google não verificou este app" é esperado (**Avançado → Acessar**). O login expira a cada 7 dias.

No Drive, o DVC guarda por hash (`files/md5/77`, `79`, `98`), não pelo nome do arquivo. Não mexer direto por lá.

---

## 04/10/2026 - Fellipe

Quatro frentes: setup do DVC, lint no projeto inteiro, CI no GitHub Actions e smoke test do container. O CI do meu escopo ficou montado; o **CD não será implementado** e fica documentado no README como evolução.

### DVC (setup na minha máquina)

Segui o passo a passo do German. As credenciais do Google ficam só em `.dvc/config.local` (não vai para o Git) e não são registradas aqui. O login expira a cada 7 dias.

```bash
git pull origin develop
uv sync --group dev --group training
uv run dvc remote modify --local gdrive_storage gdrive_client_id "<id>"
uv run dvc remote modify --local gdrive_storage gdrive_client_secret "<secret>"
uv run dvc pull
uv run pytest
```

Regra para quem alterar dado: `dvc add` → `git add` do `.dvc` → `git commit` → `dvc push` → `git push`. Esquecer o `dvc push` quebra o `dvc pull` de quem recebe o ponteiro.

### Lint no projeto inteiro

`ruff check` achou 1 problema (imports fora de ordem em `training/train_baseline.py`) e `ruff format --check` reformataria 13 arquivos, todos cosméticos. Apliquei `ruff check --fix .` e `ruff format .` no projeto **inteiro**, incluindo `training/`: padrão só vale se for para o time todo. Foi num commit separado só de estilo. **German:** faz `git pull`/rebase antes de continuar mexendo em `training/`.

### CI no GitHub Actions

Arquivo `.github/workflows/ci.yml`. Gatilhos: `push` em `develop` e `main`, e `pull_request` para qualquer branch. Dois jobs em **paralelo**, sem `needs`, porque respondem perguntas diferentes (o código está certo? a imagem sobe?):

| Job     | Steps                                                                                                                                 |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `tests` | checkout@v5 → setup-uv → `uv sync --group dev --group training` → `ruff check .` → `ruff format --check .` → `pytest`                 |
| `build` | checkout@v5 → `docker build -t medical-triage .` → `docker run -d --name api -p 8000:8000` → smoke test → logs do container se falhar |

Máquina fixada em `ubuntu-24.04` (o `ubuntu-latest` migra para o Ubuntu 26 em 19/10/2026 e poderia mudar o comportamento do CI sem mexer no código).

Erros do caminho: o primeiro run quebrou por indentação do YAML (`steps` fora do job) e depois por `uv sync --group dev` sem o grupo `training` (faltavam `pandas` e `pyyaml`).

**Resultado do último run:** dois testes estão falhando no job de tests devido ao DVC e o job de build está funcionando com smoke test validando.

### Smoke test do container

`scripts/smoke_test.sh [url]` (padrão `http://localhost:8000`). Espera o `/health` responder (até 30 tentativas, 1s) e depois chama `/predict` e confere o campo `label`. O `/health` sozinho não basta: ele responde mesmo sem o modelo, e só o `/predict` prova que o `.joblib` foi copiado e carregado. O script **não sobe nem derruba container**: recebe só uma URL, então no futuro serve contra um ambiente implantado.

```bash
docker run -d --name api -p 8000:8000 medical-triage
./scripts/smoke_test.sh
docker rm -f api
```

### Decisões

- **CD fora do escopo.** Build + push no registry + deploy ficam como evolução planejada, descrita no README. O `build` do CI só **verifica** (não publica).
- **Imagem como artefato adiada:** hoje não há quem a consuma.
- **Proteção de branch** (`develop` e `main`: exigir PR e status checks) só depois que o `tests` estiver estável. Os checks se chamam pelo `name:` dos jobs ("Checagem dos testes" e "Build da imagem Docker").

### Para alinhar com o German

1. **Testes que dependem de** `data/raw/` **no CI:** proposta de marker `@pytest.mark.requires_data` (registrado no `pyproject.toml`) com `pytest -m "not requires_data"` no CI, e/ou testes de lógica com CSV sintético pequeno. Os testes são dele, então a mudança é combinada.
2. **Versionar o** `.joblib` **no DVC em vez do Git:** cada retreino adiciona um binário novo ao histórico (~22,5 MB hoje). Se mudar, o `COPY models/` do Dockerfile no CI quebra, então precisa ser combinado antes.

---

## 05/10/2026 - German

### DAG do Airflow para treino

Branch `feature/airflow-dag-treino`, mesclada na `develop` (PR #5). Arquivo `airflow/dags/training_dag.py`, DAG `medical_triage_training`:

```
ingest_data → validate_data → train_model → evaluate_model → deploy_model
```

- A DAG só orquestra. A lógica de cada task fica em `training/dag_tasks.py`, reaproveitando `ingest`, `validate`, `train` e `evaluate`, pra poder testar sem o Airflow instalado.
- Entre as tasks só passam caminhos e números (XCom). DataFrames e modelos ficam numa pasta de staging por execução, `data/processed/<run_id>/`, que está no `.gitignore`.
- **Quality gate:** o `evaluate_model` só deixa publicar se o F1-macro no teste for ≥ `quality_gate.min_f1_macro` (0.50 no `config.yaml`; o modelo atual faz 0.558). Se reprovar, a task falha sem retry e o `deploy_model` não roda.
- O `deploy_model` grava o Pipeline em `models/`, no mesmo arquivo que a API carrega. A parte de salvar artefatos virou `pipeline.save_artifacts()`, usada tanto pelo `run_pipeline()` quanto pela DAG.
- Execução só manual (`schedule=None`), 2 retries com backoff, timeout de 30 min e callback de alerta em falha (por enquanto só registra no log).

**Airflow local** (`airflow/docker-compose.yaml`): versão 2.10.4 com LocalExecutor (sem redis nem worker) e imagem própria (`airflow/Dockerfile`) com `scikit-learn==1.9.0`, a mesma versão da API, senão o `.joblib` gerado pela DAG não carrega na API. `numpy` e `pandas` ficaram fixos nas versões que a imagem já traz, pra não quebrar o pandas dela.

```bash
cd airflow
docker compose up airflow-init   # só na primeira vez
docker compose up -d             # UI em http://localhost:8080 (airflow / airflow)
```

No Mac/Linux, criar `airflow/.env` com `AIRFLOW_UID=<saída de id -u>`.

**Testes:** 5 novos em `tests/test_dag_tasks.py`, com dataset sintético (não dependem do DVC, então rodam no CI): encadeamento das tasks, retornos cabendo no XCom, quality gate reprovado sem publicar o modelo e nome da pasta de staging válido no Windows. Total agora: 22 testes.

---

## 05/10/2026 - Fellipe

Instrumentei a API com `prometheus_client`, validei as métricas localmente e estimei o limite de capacidade na conta. O `docker-compose.yml` com Prometheus e Grafana ainda **não** foi feito.

### Como o Prometheus funciona (modelo pull)

A API não envia nada para ninguém: ela só expõe suas métricas em texto na rota `/metrics`. Quem vem buscar é o Prometheus (*scrape*), de tempos em tempos, e guarda tudo como séries temporais. O Grafana não coleta nada: ele consulta o Prometheus (linguagem PromQL) e desenha os gráficos.

O `prometheus_client` é só uma biblioteca com as peças (tipos de métrica, `generate_latest()`, etc.). Ela **não instrumenta a API sozinha**: quem mede cada requisição é o middleware que escrevi.

### Métricas criadas

Arquivo `src/medical_triage/infra/api/metrics.py`, na camada de infra: FastAPI e `prometheus_client` são detalhes externos e não podem vazar para application/domain.

| Métrica                         | Tipo      | Labels                     | Para quê                    |
| ------------------------------- | --------- | -------------------------- | --------------------------- |
| `http_requests_total`           | Counter   | `method`, `path`, `status` | Contar chamadas e erros     |
| `http_request_duration_seconds` | Histogram | `path`                     | Medir o tempo e achar o p95 |

Os três tipos que o Prometheus oferece:

- **Counter**: só cresce (`.inc()`). Serve para contar. Para ver a taxa usa-se `rate()` na consulta.
- **Gauge**: sobe e desce. Serve para um valor "de agora" (memória, itens em fila).
- **Histogram**: guarda observações (`.observe()`) em faixas e gera `_bucket`, `_sum` e `_count`.

Tempo é **Histogram, não Gauge**: o Gauge guarda só o último valor e eu perderia a distribuição, e sem ela não existe p95.

Os nomes seguem a convenção da comunidade (`snake_case`, unidade no nome como `_seconds`, Counter terminando em `_total`). Não são obrigatórios, mas seguir facilita usar dashboards prontos.

**Cardinalidade.** Cada combinação de valores de labels cria uma série separada. Se o `path` fosse a URL crua (com ids ou query string), o número de séries cresceria sem limite. Por isso uso o **template da rota** (`/predict`) e, para rotas que não existem, o valor fixo `"unmatched"`.

### Middleware e rota `/metrics`

O middleware mede o tempo com `time.perf_counter()` e registra as métricas dentro de um `try/finally`, para que requisições que lançam exceção também entrem na contagem (status padrão 500). O registro correto no FastAPI é:

```python
app.middleware("http")(metrics_middleware)
```

A forma `app.middleware(metrics_middleware)` que escrevi primeiro **não registra nada**: o argumento do decorator é o tipo (`"http"`), não a função.

A rota devolve o texto no formato do Prometheus, sem aparecer no Swagger:

```python
@router.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

Com `-> str` a rota devolveria JSON, e o Prometheus espera texto puro (`text/plain; version=0.0.4`).

Alguns termos que me deram dúvida no código:

| Termo                                      | O que é                                                                                                                                                                                                          |
| ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Callable[[Request], Awaitable[Response]]` | Tipo de uma função que recebe um `Request` e devolve algo "esperável" (`await`) que resulta em `Response`. Em Java seria `Function<Request, CompletableFuture<Response>>`.                                       |
| `a if cond else b`                         | Operador ternário do Python (o `cond ? a : b` do Java). Em `route.path if route else "unmatched"`, um `route` não nulo conta como verdadeiro.                                                                    |
| `time.perf_counter()`                      | Relógio monotônico e de alta resolução, feito para medir **diferenças**. O `time.time()` pode ser ajustado pelo sistema operacional e pular. O valor absoluto não significa nada, só a subtração `fim - início`. |
| `try/finally`                              | Garante que o registro das métricas aconteça mesmo se a requisição falhar. Sem ele, as falhas ficariam fora da contagem.                                                                                         |

### Buckets do histograma

Cada bucket é uma faixa "menor ou igual a" (`le`) e eles são **cumulativos**: `le="0.02"` conta tudo que levou até 20 ms, incluindo o que já estava nos buckets menores. Para saber quantas caíram numa faixa, subtraio o bucket anterior.

Com os buckets padrão do Prometheus eu não tinha resolução na faixa de ~15 ms. Defini os meus à mão, de 1 ms a 1 s, mais densos entre 10 e 25 ms e com um bucket em **0,1 s**, que é o meu limite. A média sai de `_sum / _count`, e o p95 é uma **estimativa** por interpolação dentro do bucket (buckets estreitos dão mais precisão).

Validação local: 10 chamadas ao `/predict` com o `scripts/benchmark.sh`.

| Faixa (tempo no servidor) | Requisições                |
| ------------------------- | -------------------------- |
| até 10 ms                 | 0                          |
| 10 a 15 ms                | 7                          |
| 15 a 20 ms                | 2                          |
| 25 a 50 ms                | 1 (a primeira, cold start) |

Média no servidor: `0,1678 s / 10 ≈ 16,8 ms`. No cliente (`curl`) deu ≈ 17,7 ms, e a diferença é o custo de conexão e do próprio `curl`. A primeira requisição é mais lenta (cold start) e, com poucas amostras, ela domina o p95. Por isso o p95 deve ser lido em janelas (`[1m]`, `[5m]`) com tráfego contínuo.

### Capacidade: até onde a API aguenta

O `benchmark.sh` **não é teste de carga**: ele faz uma chamada de cada vez (concorrência 1) e só mede a latência isolada. Para saber o limite é preciso subir a concorrência em degraus e ver onde o p95 estoura. Ainda não rodei esse teste. Fiz a estimativa na conta, como hipótese para validar depois (ferramenta candidata: Locust).

| Termo         | O que é                                                                                                                |
| ------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Latência      | Tempo de uma requisição.                                                                                               |
| Throughput    | Requisições atendidas por segundo (RPS).                                                                               |
| Concorrência  | Quantas requisições estão em andamento ao mesmo tempo.                                                                 |
| Saturação     | Ponto em que um recurso (CPU, threads) esgota: as requisições entram em fila, a latência sobe e o RPS para de crescer. |
| Lei de Little | `concorrência = throughput × latência`. Liga as três grandezas acima.                                                  |
| SLO           | A meta que eu aceitei: **p95 < 100 ms** e taxa de erro < 1% (o 1% é uma escolha minha, ainda a justificar).            |

Uso o p95 e não a média porque a média esconde a cauda: com 19 requisições de 20 ms e uma de 600 ms a média dá ~49 ms, mas 5% dos usuários tiveram uma resposta péssima. E olho também os erros, porque uma API que quebra devolve 500 rapidinho e passaria no teste de latência.

Modelo simples: o `/predict` ocupa o servidor ~16,8 ms por requisição, atendida uma de cada vez.

- Throughput máximo ≈ 1000 ÷ 16,8 ≈ **60 req/s**. Esse teto **não depende** dos 100 ms, só do tempo que cada requisição ocupa o servidor.
- Concorrência em que o p95 passa de 100 ms ≈ 100 ÷ 16,8 ≈ **6 clientes simultâneos** (confere com Little: 60 req/s × 0,1 s = 6). Errei nessa conta no início ao responder 62: misturei a taxa (req/s) com a concorrência (clientes).
- 6 requisições em andamento **não são 6 usuários**: usuários reais pensam entre uma chamada e outra. A conta vale para o teste de carga, não para estimar quantos usuários a API aguenta.

É só um modelo: o sklearn pode paralelizar, o FastAPI roda rotas `def` numa pool de threads e o uvicorn aceita mais workers, então a medição real pode ficar acima disso.

### Decisões

- Latência é Histogram, com buckets definidos à mão e um bucket exatamente no limite de 100 ms.
- O `path` do label usa o template da rota, e rota inexistente vira `"unmatched"`.
- Métricas de negócio (ex.: predições por classe) ficam para depois e entrariam por uma port na camada application, com um adaptador Prometheus em `infra/`.
- Dentro do Docker Compose, `localhost` é o próprio container. O Prometheus vai usar o **nome do serviço** (`api:8000`) como alvo, pelo DNS interno da rede do Compose. Ainda vou confirmar isso na prática.

### Para alinhar com o German

1. **Baseline de latência:** o documentado antes (~0,27 ms) não bate com o que medi agora (~15 ms). Pode ser o modelo novo do tuning (289 árvores), Docker contra local ou diferença de unidade. Vou medir de novo no Docker com o modelo atual antes da comparação da Etapa 4.
2. **Buckets:** foram escolhidos para a latência atual. Se o modelo mudar muito (mais árvores, outro algoritmo), é preciso revisar.

---

## Aberto

1. `docker-compose.yml` com API + Prometheus + Grafana, `prometheus.yml` e datasource do Grafana por provisioning
2. Dashboard simples (taxa de requisições, p95, contagem por status), com export do JSON e print
3. Testes de `/metrics` e do middleware, e conferir que o `prometheus-client` está nas dependências de produção (o Dockerfile usa `uv sync --frozen --no-dev`)
4. Decidir se o middleware ignora o próprio `/metrics` (hoje ele mede os scrapes do Prometheus, o que gera ruído) e se desligo as séries `_created`
5. Refazer o baseline de latência no Docker e atualizar `docs/baseline_latency.md`
6. Teste de carga para validar a hipótese de ~60 req/s e saturação com ~6 clientes
7. Opcional: "aquecer" o modelo no `lifespan` para reduzir o cold start

---

## 11/10/2026 - German

### Testes que dependem do dataset pulados no CI

Branch `fix/testes-dataset-ci`. Dois testes falhavam no GitHub Actions com `FileNotFoundError`:

- `tests/test_data.py::test_load_data_retorna_train_test_e_label_map`
- `tests/test_pipeline.py::test_pipeline_end_to_end_roda_e_retorna_metricas`

Os dois leem os CSVs de `data/raw/`. Os CSVs estão no DVC: o Git guarda só os ponteiros `.dvc`, e o CI faz `git checkout` mas nunca `dvc pull`, então os arquivos não existem lá. Reproduzi isso numa cópia limpa da `develop` (2 failed, 20 passed).

**O que fiz:** nos dois arquivos, um marcador `requires_dataset` (`pytest.mark.skipif`) que pula o teste quando falta algum dos 3 CSVs em `data/raw/`. O caminho vem do `DATA_DIR` do `training/ingest.py`, então acompanha se um dia mudar. Só esses 2 testes levam o marcador; as validações com dados sintéticos continuam rodando sempre.

| Onde | Antes | Depois |
| --- | --- | --- |
| CI (sem os CSVs) | 2 failed, 20 passed | 20 passed, 2 skipped |
| Máquina com os CSVs | 22 passed | 22 passed |

Pra ver o motivo de cada teste pulado: `uv run pytest -rs`.

**Por que não fazer `dvc pull` no CI:** o CI precisaria de credencial do Google Drive como segredo do GitHub. O login OAuth que configuramos é de "app em teste" e expira a cada 7 dias, então quebraria toda semana. O certo seria uma service account com acesso à pasta, e ainda mexeria no `ci.yaml`. Não compensa: o que esses 2 testes cobrem (ingestão, treino, avaliação e o encadeamento) já é testado com dados sintéticos em `test_train.py` e `test_dag_tasks.py`. Os 2 testes com o dataset real ficam como testes de integração locais.

**Formatação dos arquivos do Prometheus (Fé, só pra você saber):** o `ruff format --check` do CI falhava em `src/medical_triage/infra/api/metrics.py`, `routes.py` e `main.py`, e como ele roda antes do `pytest`, o passo de testes nem chegava a executar. Rodei `ruff format` nos três: só 3 linhas em branco antes de funções e `'http'` → `"http"` no `main.py`, nenhuma mudança de lógica.

### DAG quebrada no Airflow do Windows: leitura lenta (PR #8)

Primeira vez que subi o Airflow via Docker, no computador Windows. A UI mostrou `Broken DAG: [/opt/airflow/dags/training_dag.py]` e a DAG não aparecia. Não era erro de código: dentro do container, `import training.dag_tasks` funcionava e o `airflow dags list-import-errors` já não mostrava nada pouco depois.

**Causa:** tempo. O Airflow corta a leitura de cada arquivo de DAG em 30 s (`dagbag_import_timeout`) e relê o arquivo a cada 30 s e no início de cada task. A DAG fazia `from training import dag_tasks` no topo, o que carrega `pandas`, `scikit-learn` e `scipy` em toda leitura. No Windows isso levou ~19 s com cache e passou dos 30 s na primeira leitura, a fria.

**Correção** (branch `fix/airflow-dag-import-lento`):
- `airflow/dags/training_dag.py`: cada task importa o `training.dag_tasks` dentro da própria função, só quando executa. O arquivo da DAG fica leve.
- `airflow/docker-compose.yaml`: `AIRFLOW__CORE__DAGBAG_IMPORT_TIMEOUT=120` e `AIRFLOW__CORE__DAG_FILE_PROCESSOR_TIMEOUT=180` (o segundo precisa ser maior que o primeiro), como margem.

| Leitura do arquivo da DAG (medido no Mac) | Antes | Depois |
| --- | --- | --- |
| Tempo | 7 s na primeira, ~1,2 s depois | 0,06 s |
| Bibliotecas pesadas carregadas | `pandas`, `sklearn`, `scipy` | nenhuma |

O comportamento da DAG não mudou: mesmas 5 tasks, mesmos retries (o `evaluate_model` continua sem retry), mesmas métricas no `airflow dags test`.

Não precisa reconstruir a imagem: a pasta `dags/` é montada no container e o Airflow pega o arquivo novo sozinho. Pras variáveis novas do compose, basta `docker compose up -d`.

### DAG rodando no Airflow via Docker

Com isso, a DAG rodou de ponta a ponta no Airflow 2.10.4 via Docker Compose, no Windows: **2 execuções, as 2 com sucesso**, as 5 tasks verdes em cada uma. Print em `docs/airflow_runs.jpeg`. É a evidência do item "DAG Airflow funcional" do Tech Challenge (Boas Práticas Obrigatórias e critério Orquestração, 15%), que marquei no checklist do README.

A execução mais longa levou 3 min 50 s, bem mais que no Mac (segundos). Boa parte disso é o mesmo custo de importar as bibliotecas no início de cada task, agravado pela pasta do Windows montada no Docker.

No README também marquei "API instrumentada com `prometheus_client`", que estava feito desde o PR #6 do Fé (`metrics.py`, com `http_requests_total` e `http_request_duration_seconds`) mas tinha ficado desmarcado.

---

## 15/10/2026 - Fellipe

Duas frentes da Etapa 3: um teste de carga rápido, que validou a hipótese de capacidade que eu tinha anotado em 05/10, e a stack de monitoramento no `docker-compose.yml` (API + Prometheus + Grafana) com o `prometheus.yml`. O Grafana ainda não tem datasource nem dashboard.

### Teste de carga rápido (validou a hipótese)

Script `scripts/load_test.sh`, com a ferramenta `hey`. Ele aquece a API por 3 s e depois sobe a concorrência em degraus de 10 s cada (1, 2, 4, 6, 8, 12, 20 e 50 clientes simultâneos), imprimindo req/s, p95, p99 e respostas fora de 2xx.

```bash
brew install hey
./scripts/load_test.sh
```

Resultado (API e gerador de carga na mesma máquina, então os números são de referência local):

| Clientes | req/s | Não-2xx | Latência esperada (clientes ÷ req/s) |
| -------- | ----- | ------- | ------------------------------------ |
| 1        | 64,8  | 0       | ~14 ms                               |
| 2        | 70,1  | 0       | ~29 ms                               |
| 4        | 69,4  | 0       | ~57 ms                               |
| 6        | 69,9  | 0       | ~86 ms                               |
| 8        | 71,0  | 0       | ~114 ms                              |
| 12       | 69,6  | 0       | ~171 ms                              |
| 20       | 71,0  | 0       | ~286 ms                              |
| 50       | 69,8  | 0       | ~714 ms                              |

- O throughput trava em **~70 req/s** já com 2 clientes: o servidor atende praticamente uma requisição de cada vez, e isso é a saturação. A previsão era ~60 req/s (a diferença vem de a requisição levar ~14 ms e não 16,8 ms).
- Pela Lei de Little (`concorrência = throughput × latência`), o limite de 100 ms para o p95 fica entre **6 e 8 clientes simultâneos**, que bate com a previsão de ~6.
- Não houve nenhum erro: sob sobrecarga a API **degrada em latência, mas não quebra**.
- O `/metrics` confirmou a mesma curva: das 5818 requisições do teste, 3170 (54%) terminaram em até 100 ms e nenhuma passou de 1 s.
- Esse histograma acumula o teste inteiro e mistura todos os degraus, então um p95 calculado sobre ele não significa nada. Para olhar cada degrau é preciso a taxa por janela de tempo (`rate()` sobre o histograma), que é o que o Grafana vai fazer.
- **Problema no script:** as colunas p95 e p99 saíram vazias, porque o `awk` não encontrou essas linhas na saída real do `hey`. As outras colunas funcionam. Falta colar uma saída bruta de uma rodada para ajustar o parsing.

### Stack de monitoramento no Docker Compose

O Compose cria uma rede interna onde cada serviço é alcançável pelo **nome da chave** do serviço (DNS interno). Três serviços:

| Serviço                 | Imagem                          | Porta no host | Papel                                    |
| ----------------------- | ------------------------------- | ------------- | ---------------------------------------- |
| `medical-triage`        | construída do `Dockerfile` (`build: .`) | 8000   | A API                                    |
| `prometheus`            | `prom/prometheus:v3.5.0`        | 9090          | Coleta e guarda as métricas              |
| `grafana`               | `grafana/grafana:12.2.0`        | 3000          | Consulta o Prometheus e desenha os gráficos |

```yaml
services:
  medical-triage:
    build: .
    ports:
      - "8000:8000"
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 10s
      timeout: 3s
      retries: 3
      start_period: 20s

  prometheus:
    image: prom/prometheus:v3.5.0
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    restart: unless-stopped

  grafana:
    image: grafana/grafana:12.2.0
    ports:
      - "3000:3000"
    depends_on:
      - prometheus
    volumes:
      - grafana-data:/var/lib/grafana
    restart: unless-stopped

volumes:
  grafana-data:
  prometheus-data:
```

### `prometheus.yml`

Arquivo em `monitoring/prometheus/prometheus.yml`, montado dentro do container do Prometheus.

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: medical-triage-api
    static_configs:
      - targets:
          - medical-triage:8000
```

| Termo             | O que é                                                                                                                                                              |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `scrape_interval` | De quanto em quanto tempo o Prometheus faz uma busca (`GET /metrics`). Ele busca para sempre; o valor é o intervalo entre as buscas, não uma duração. Limita a janela útil do `rate()`: com 15 s, `[1m]` tem ~4 pontos. |
| `job`             | Um nome para um grupo de alvos com a mesma função. Vira o label `job` em toda série coletada dali. Cada alvo também ganha o label `instance` (`host:porta`).        |
| `target`          | Só `host:porta`, **sem** o `/metrics`. O caminho é outra configuração (`metrics_path`), cujo padrão já é `/metrics`.                                                |
| `up`              | Métrica que o Prometheus cria sozinho para cada alvo: 1 se a última coleta funcionou, 0 se falhou. O jeito mais rápido de conferir se a configuração está certa.     |

O target tem que ser **idêntico à chave do serviço no compose** (`medical-triage`), com a **porta do container** (8000). Dentro do container do Prometheus, `localhost` é o próprio Prometheus, e não a API.

### Conceitos de Compose que eu precisei fixar

| Termo                      | O que é                                                                                                                                                                      |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `image` vs `build`         | `image` baixa uma imagem pronta de um registry. `build: .` constrói a partir do `Dockerfile` da pasta indicada. `image: DockerFile` está errado: `image` não aceita arquivo.  |
| `ports: "H:C"`             | Publica uma porta do container na **minha máquina**. Entre containers da mesma rede não é necessário: eles se falam pelo nome do serviço e pela porta do container.          |
| Bind mount                 | `./arquivo:/destino:ro` liga um arquivo do host a um caminho dentro do container (`ro` = somente leitura). Usei para entregar o `prometheus.yml` ao Prometheus.              |
| Volume nomeado             | `nome:/destino`, declarado na seção `volumes:` do final. É criado e gerenciado pelo Docker (fora do container), cresce conforme se grava e sobrevive à recriação do container. `docker compose down` não o apaga, só `down -v`. |
| `depends_on`               | Só controla a **ordem de início**. Se o Prometheus cair depois, o Grafana continua rodando. Esperar o serviço ficar pronto exige `condition: service_healthy` e um healthcheck. |
| `restart: unless-stopped`  | Reinicia o container se o processo cair ou se o Docker/máquina reiniciar, a menos que eu o tenha parado de propósito. Reage à **saída do processo**, não ao estado `unhealthy`. |
| `healthcheck`              | Comando que o Docker roda **dentro** do container de tempos em tempos. Código de saída 0 = saudável. Outros serviços, orquestradores e o `docker compose ps` consomem esse estado. |

O healthcheck da API usa só a biblioteca padrão do Python, porque a imagem `python:3.12-slim` não traz `curl`:

```
python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
```

O `urlopen` lança exceção se a resposta for 4xx/5xx ou se a conexão for recusada, e a exceção não tratada faz o Python sair com código 1 (falha). Aqui o `localhost` está certo, porque o comando roda dentro do próprio container da API.

### Decisões

- **URL do datasource do Grafana:** `http://prometheus:9090` (nome do serviço mais porta do container), e não `localhost`. Quem consulta o Prometheus é o servidor do Grafana (o container), não o meu navegador.
- Versões das imagens **fixadas** em tags (reprodutibilidade, mesmo motivo do `ubuntu-24.04` no CI). As tags `v3.5.0` e `12.2.0` precisam ser conferidas no Docker Hub; se o `pull` falhar com "manifest unknown", é a tag.
- O healthcheck da API usa o `/health`, que é rota de liveness. Cada chamada dele passa pelo middleware e é contada nas métricas; é a mesma questão do ruído do `/metrics` medindo a si mesmo.
- Prometheus e Grafana ficam **sem healthcheck** por enquanto: ninguém consome esse estado na nossa stack. Quando for necessário, os endpoints são `/-/ready` (Prometheus, pronto para consultas), `/-/healthy` (processo vivo) e `/api/health` (Grafana). Preciso confirmar se a imagem traz `wget` antes de usar.
- Volume `prometheus-data` para preservar o histórico de séries entre recriações do container.

### Para alinhar com o German

- Se o `.joblib` passar a ser versionado no DVC (item que já estava combinado), o `docker compose up --build` e o `COPY models/` do `Dockerfile` passam a exigir `dvc pull` antes do build.
- O teste de carga mede o `random_forest_pipeline.joblib` atual (modelo do tuning). Se o modelo mudar, os ~70 req/s e a faixa de buckets precisam ser reavaliados.

### Em aberto

1. [CONFIRMAR] Resultado de `docker compose up -d --build`: estado no `docker compose ps` (API `healthy`), alvo UP em Status → Targets e resultado da consulta `up`.
2. Datasource do Grafana por provisioning (arquivo em `monitoring/grafana/provisioning/datasources/`, montado no container).
3. Dashboard simples: taxa de requisições (`rate(http_requests_total[1m])`), p95 de latência (`histogram_quantile` sobre o histograma) e contagem por status; export do JSON e print como entrega da etapa.
4. Ajustar o parsing do `load_test.sh` (p95/p99) a partir de uma saída bruta do `hey`.
5. Investigar o teto de ~70 req/s: mais workers do uvicorn ajudam, mas só até o limite da CPU da própria máquina.
6. Decidir se o middleware ignora `/metrics` e `/health` (ruído) e se desligo as séries `_created`.
7. Testes de `/metrics` e do middleware; confirmar o `prometheus-client` nas dependências de produção (o Dockerfile usa `--no-dev`).
8. Refazer o baseline de latência no Docker e atualizar `docs/baseline_latency.md`.
9. README: seção de monitoramento (como subir a stack, portas, como achar o dashboard).

---

## 18/10/2026 - Fellipe

Fechei o dashboard do Grafana da Etapa 3. O datasource e o dashboard agora são **configuração versionada** (provisioning): sobem sozinhos com `docker compose up`, em qualquer máquina, sem clicar em nada.

### Datasource por provisioning

Primeiro criei o datasource à mão pela interface, para ver o problema: tudo que é criado pela UI fica no banco do Grafana (volume `grafana-data`) e some com um `docker compose down -v`. Depois fiz o mesmo por arquivo.

Arquivo `monitoring/grafana/provisioning/datasources/datasource.yml`:

```yaml
apiVersion: 1

datasources:
  - name: Prometheus
    uid: prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: false
```

| Campo | Por quê |
| --- | --- |
| `uid: prometheus` | O dashboard aponta para o datasource pelo uid. Pela UI ele saiu aleatório (`eg1qdp3zazitca`), o que quebraria o dashboard em outra máquina. |
| `access: proxy` | Quem consulta o Prometheus é o servidor do Grafana (o container), não o navegador. |
| `url: http://prometheus:9090` | Nome do serviço mais porta do container, pelo DNS interno da rede do compose. Com `localhost:9090` o teste falha: dentro do container do Grafana, `localhost` é o próprio Grafana. |
| `editable: false` | A UI não altera o datasource; a fonte da verdade é o arquivo. |

### Dashboard "Medical Triage API"

Organizado em duas rows, seguindo os padrões RED (serviço, visto de fora) e USE (recursos, vistos de dentro). Cabe numa tela só.

| Row | Painel | Consulta | Unidade |
| --- | --- | --- | --- |
| Serviço (RED) | Requisições por segundo | `sum by (path) (rate(http_requests_total[1m]))` | req/s |
| Serviço (RED) | Taxa de erro (5xx) | `(sum(rate(http_requests_total{path="/predict", status=~"5.."}[1m])) or vector(0)) / sum(rate(http_requests_total{path="/predict"}[1m]))` | %, linha em 1% |
| Serviço (RED) | Requisições por status | `sum by (status) (rate(http_requests_total[1m]))` | req/s |
| Serviço (RED) | Latência — /predict | `histogram_quantile` 0.95 e 0.99 sobre `sum by (le) (rate(http_request_duration_seconds_bucket{path="/predict"}[1m]))` | ms, linha em 100 ms |
| Recursos (USE) | CPU do processo da API | `rate(process_cpu_seconds_total{job="medical-triage-api"}[1m])` | núcleos, linha em 1 |
| Recursos (USE) | Memória do processo da API | `process_resident_memory_bytes{job="medical-triage-api"}` | bytes (IEC) |

CPU e memória vêm das métricas de processo que o `prometheus_client` já expõe em Linux. Não precisou de cAdvisor nem de outro serviço.

### Dashboard como código

Arquivo `monitoring/grafana/provisioning/dashboards/dashboards.yml` (o provider, que só aponta para a pasta dos JSONs):

```yaml
apiVersion: 1

providers:
  - name: medical-triage
    folder: Medical Triage
    type: file
    disableDeletion: true
    allowUiUpdates: false
    options:
      path: /etc/grafana/dashboards
```

O dashboard em si é `monitoring/grafana/dashboards/medical-triage.json`. Volumes no serviço `grafana` do compose:

```yaml
    volumes:
      - grafana-data:/var/lib/grafana
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning:ro
      - ./monitoring/grafana/dashboards:/etc/grafana/dashboards:ro
```

**Como alterar o dashboard daqui para a frente:** editar na UI → **Export → Export as code**, formato **Classic**, com "export externally" desligado → substituir o `medical-triage.json` → commit. A UI não salva por cima (`allowUiUpdates: false`), e o Grafana relê a pasta a cada ~10 s, sem reiniciar. Os formatos *Resource* do export e a opção externa (que troca o datasource por `${DS_PROMETHEUS}`) não funcionam com esse provider.

Validação: `docker compose down -v` + `docker compose up -d`. O datasource e o dashboard voltaram sozinhos, sem erro nos logs, e a UI recusou salvar alteração por cima do dashboard provisionado.

### O que o dashboard mostrou (tráfego contínuo, ~7,3 req/s no /predict)

| Sinal | Valor |
| --- | --- |
| p95 do /predict | ~48 ms |
| p99 do /predict | ~50 ms, com um pico de ~60 ms |
| Taxa de erro 5xx | 0% |
| CPU | ~0,14 núcleo |
| Memória | ~190–200 MiB, estável |

- 0,14 núcleo ÷ 7,3 req/s ≈ **19 ms de CPU por requisição**, perto da latência medida. Hipótese: o `/predict` é limitado por CPU, o que explicaria o teto de ~70 req/s do teste de carga (um processo, um núcleo). Falta confirmar rodando o teste de carga com o painel aberto.
- O p95 de ~48 ms está bem acima da média de ~15 ms que eu tinha medido. Parte disso é interpolação: os buckets pulam de 25 para 50 ms, e o `histogram_quantile` estima dentro da faixa. Falta comparar com a média (`rate(_sum) / rate(_count)`).

### Decisões

- **Erro do serviço é só 5xx.** Um 422 é a API recusando entrada inválida, ou seja, funcionando. O SLO de erro < 1% conta só 5xx.
- A taxa de erro filtra `path="/predict"`: sem isso, os scrapes do `/metrics` e os healthchecks entram no total e diluem a taxa.
- `or vector(0)` na taxa de erro: a série de 5xx só existe depois do primeiro erro, e sem isso o painel mostraria "No data" justamente quando está tudo bem.
- Unidades informadas na forma crua (segundos, bytes, fração), com o Grafana convertendo na exibição. Eixos com *Soft min* 0 para não ampliar variações minúsculas.
- Dashboard só por provisioning; a interface serve para desenhar, não para guardar.

### Para alinhar com o German

- Mudanças no dashboard entram por PR, editando o JSON (fluxo acima). Alteração feita só na UI se perde.
- Se o modelo mudar (mais árvores, outro algoritmo), os buckets do histograma e a leitura do painel de CPU precisam ser revistos.

### Em aberto

1. Senha do Grafana via `.env` (`GF_SECURITY_ADMIN_USER` / `GF_SECURITY_ADMIN_PASSWORD`) com `.env.example` no repo; hoje é o padrão `admin`/`admin`.
2. Print do dashboard como entrega da etapa.
3. README: seção de monitoramento (como subir a stack, portas, onde achar o dashboard, fluxo de alteração).
4. Teste de carga com o painel de CPU aberto, para confirmar o teto por CPU.
5. Comparar p95 com a média recente e avaliar buckets entre 25 e 50 ms.
6. Ajustar o parsing do `load_test.sh` (p95/p99) a partir de uma saída bruta do `hey`.
7. Decidir se o middleware ignora `/metrics` e `/health` (ruído nas métricas) e se desligo as séries `_created`.
8. Testes de `/metrics` e do middleware; confirmar o `prometheus-client` nas dependências de produção (o Dockerfile usa `--no-dev`).
9. Refazer o baseline de latência no Docker e atualizar `docs/baseline_latency.md`.

---

## 19/10/2026 - German

### Otimização de latência com ONNX (Etapa 4)

Branch `feature/onnx-otimizacao-latencia`. Detalhes e tabela completa em `docs/otimizacao_onnx.md`.

O Pipeline (TF-IDF + Random Forest) é exportado para ONNX e executado com o ONNX Runtime. Escolhi ONNX e não quantização porque a quantização da Etapa 6 (Aula 04) age sobre pesos de redes neurais, e o Random Forest não tem pesos. O ONNX se aplica ao modelo atual sem retreinar.

- `training/export_onnx.py` (novo): converte com `skl2onnx` (o TF-IDF vai junto no grafo, então o modelo recebe texto cru e devolve a classe em texto, igual ao `.joblib`) e confere a **paridade** com o scikit-learn.
- `pipeline.save_artifacts()` agora gera o `models/random_forest_pipeline.onnx` junto com o `.joblib`, então o `pipeline.py` e a task `deploy_model` da DAG publicam os dois. O ONNX é conferido **antes** de gravar qualquer arquivo: se bater em menos de 99% (`onnx.min_parity` no `config.yaml`), nada em `models/` muda. Hoje bate 100% (2.888/2.888).
- `training/benchmark_onnx.py` (novo): mede a latência de um laudo por vez e grava `docs/latency_benchmark.json`.

| Variante (só a inferência, 1 laudo) | p50 | p99 |
| --- | --- | --- |
| scikit-learn como publicado (`n_jobs=-1`) | 13,50 ms | 14,69 ms |
| scikit-learn `n_jobs=1` | 5,49 ms | 5,82 ms |
| **ONNX Runtime** | **0,050 ms** | **0,079 ms** |

Ganho de 268x no p50 sobre o publicado (109x sobre `n_jobs=1`). O `.onnx` tem 13,9 MB (o `.joblib`, 22,5 MB).

**Dependências:** `skl2onnx` e `onnxruntime` no grupo `training` do `pyproject.toml`, e fixados no `airflow/requirements.txt` (`skl2onnx==1.20.0`, `onnxruntime==1.30.0`). Testei a imagem do Airflow de novo: com o numpy 1.26.4 da imagem o pip escolhe o `onnx 1.22.0`, a DAG roda inteira e publica o `.onnx` com 100% de paridade, e esse `.onnx` funciona no ambiente com numpy 2. A imagem do Airflow precisa ser reconstruída (`docker compose build`) por causa do `requirements.txt`.

**`uv.lock`:** não dá pra gerar nesta máquina (a rede da empresa bloqueia o PyPI público e o `uv` grava o índice interno). Fica pra gerar no Windows antes do PR, só ele num commit.

**Testes:** 4 novos em `tests/test_export_onnx.py` (modelo sintético, rodam no CI): predições iguais ao scikit-learn, rótulos em texto, paridade aprovada e paridade reprovada. `test_dag_tasks.py` e `test_pipeline.py` passaram a conferir o `.onnx` publicado.

### Pro Fellipe

1. **Baseline de latência (o ponto que você deixou pra alinhar comigo):** os ~15 ms que você mediu na API são quase todos do modelo: só a inferência leva 13,5 ms, por causa do `n_jobs=-1` do Random Forest. Os ~0,27 ms antigos devem ser do teu modelo sintético (`classifier.joblib`, Regressão Logística): medi agora e ele faz 0,18 ms.
2. **Servir o ONNX na API (decisão tua):** o `.onnx` já está em `models/`. Precisaria mover o `onnxruntime` para as `dependencies` de produção e criar um adaptador em `infra/ml/` que carregue o `.onnx` (a entrada se chama `text`, formato `[n, 1]`; a saída `label` já vem em texto). Se servir, os buckets do histograma vão precisar de faixas abaixo de 1 ms.

---

## 25/10/2026 - German

### README no formato padrão e completo

O README era só o checklist do Tech Challenge. Reescrevi no formato de um README de projeto, com o checklist mantido no fim (`30d09da`):

- **Apresentação e arquitetura:** o problema, o que o projeto cobre, diagrama (Mermaid) e estrutura do repositório.
- **Como rodar:** pré-requisitos, início rápido com Docker Compose (API + Prometheus + Grafana), API sem Docker, dataset (download público ou `dvc pull`), treino, Airflow e testes.
- **Resultados:** métricas do modelo, latência scikit-learn vs. ONNX e números do monitoramento.
- **Decisão de arquitetura em nuvem:** seção criada, com o "tempo real, não batch" já decidido e o provedor a completar.
- **Documentação e Equipe.**

Depois, pelo GitHub (`eccfed0`, `d2d43f5`), troquei a coluna "Responsável por" da tabela da equipe por "E-Mail".

---

## 31/10/2026 - German

### Decisão de arquitetura em nuvem (Etapa 1)

Completei a seção **Decisão de arquitetura em nuvem** do README, que estava como "A completar (Fellipe)". Era o último item da Etapa 1 no checklist, que agora está marcado. Ela parte da tua entrada de 13/09 (tempo real, não batch) e da tua escolha de AWS pela familiaridade com a plataforma, e se apoia no material da Etapa 1 (Deploy em nuvem): padrões batch, tempo real e serverless (Aula 01), modelos leves em API (Aula 02), serviços da AWS (Aula 03), equivalentes no Azure e no GCP (Aulas 04 e 05) e custo e segurança (Aula 06).

Está escrito como **proposta**: não subimos nada na nuvem, e o desafio só pede a análise textual.

Resumo do que ficou no README:

- **Tempo real:** a triagem precisa da resposta na hora, e o modelo é leve (só CPU, ~200 MiB, ~13,5 ms por laudo). Batch fica como complemento (reprocessar lote histórico, retreino pela DAG).
- **Por que não serverless (Lambda):** (1) o Prometheus faz pull do `/metrics` e precisa de um processo de pé; numa função efêmera os contadores e o histograma se perdem a cada execução; (2) cold start carregando scikit-learn e o modelo de 22,5 MB; (3) hospital tem fluxo contínuo, não intermitente. O argumento (1) é nosso, o material só fala de cold start.
- **AWS:** tabela de equivalentes nos três provedores (registry, VM, serverless, batch, plataforma gerenciada). Como nenhum tem vantagem técnica decisiva para um modelo leve em CPU, o desempate foi a experiência da equipe. Escrevi "um dos integrantes trabalha com AWS no dia a dia", sem citar nome.
- **Arquitetura proposta:** imagem da API no **ECR** → **EC2** rodando o mesmo `docker-compose.yml` (API + Prometheus + Grafana), região `sa-east-1` (proximidade e dados de saúde, sensíveis pela LGPD). SageMaker descartado: custo e complexidade desproporcionais para um modelo de 22,5 MB, e exigiria adaptar o monitoramento.
- **Escala:** mais workers do uvicorn, instância maior ou várias atrás de load balancer; servir o ONNX tiraria o modelo da posição de gargalo.
- **Custo e segurança:** custo constante do tempo real (instância pequena, desligar ambientes de teste); HTTPS, criptografia de volumes, IAM com menor privilégio, Prometheus/Grafana sem acesso público e, numa evolução, autenticação e limite de requisições na API.

### Pro Fellipe

A seção era tua, então revisa e muda o que não concordar:

1. **Serviço de execução:** usei **EC2** porque é o que a Aula 03 ensina para tempo real e porque roda o nosso `docker-compose.yml` sem mudança. Se você prefere outro serviço que usa no trabalho (ECS/Fargate, por exemplo), é só trocar.
2. **Frase da experiência da equipe:** veja se está ok do jeito que ficou.
3. **Recomendações de segurança:** autenticação e limite de requisições estão como "numa evolução", porque a API hoje não tem nenhum dos dois.
4. Se a gente chegar a subir na AWS, a seção muda de "proposta" para "implantado".

---
