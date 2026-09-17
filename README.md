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
