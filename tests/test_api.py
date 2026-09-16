# tests/test_api.py

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from medical_triage.infra.api.dependencies import get_model
from medical_triage.main import app


# --- Fixture: cliente HTTP em memória ---
# pytest.fixture cria um recurso reutilizável entre testes.
# "yield" entrega o client para o teste; após o teste, limpa o override.
@pytest.fixture
def client():
    mock = MagicMock()
    mock.predict.return_value = ["cardiovascular"]

    # Substitui get_model pelo mock — FastAPI nunca chama a versão real
    app.dependency_overrides[get_model] = lambda: mock

    with TestClient(app) as c:
        yield c

    # Limpeza: remove o override para não vazar entre testes
    app.dependency_overrides.clear()


# --- Testes do /health ---

def test_health_return_200(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_return_status_ok(client):
    response = client.get("/health")
    assert response.json() == {"status": "ok"}


# --- Testes do /predict ---

def test_predict_return_200(client):
    response = client.post(
        "/predict",
        json={"text": "patient with cardiac chest pain and heart failure"},
    )
    assert response.status_code == 200


def test_predict_return_label(client):
    response = client.post(
        "/predict",
        json={"text": "patient with cardiac chest pain and heart failure"},
    )
    assert response.json()["label"] == "cardiovascular"


def test_predict_not_text_return_422(client):
    # 422 Unprocessable Entity: FastAPI valida o schema automaticamente
    # Se o campo obrigatório "text" não vier, a requisição é rejeitada
    response = client.post("/predict", json={})
    assert response.status_code == 422


# --- Teste do modelo indisponível ---

def test_predict_no_model_return_503():
    # Sem override: get_model vai encontrar app.state.model = None
    # e deve lançar HTTPException 503
    app.dependency_overrides.clear()

    with TestClient(app) as c:
        # Garante que o modelo está None (lifespan pode ter tentado carregar)
        app.state.model = None
        response = c.post(
            "/predict",
            json={"text": "patient with cardiac chest pain"},
        )

    assert response.status_code == 503