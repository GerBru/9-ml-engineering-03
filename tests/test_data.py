# tests/test_data.py

import pandas as pd
import pytest

from training.ingest import load_data
from training.validate import validate_data

# --- Testes de ingest ---


def test_load_data_retorna_train_test_e_label_map():
    train_df, test_df, label_map = load_data()

    assert len(train_df) > 0
    assert len(test_df) > 0
    assert {"condition_label", "medical_abstract"}.issubset(train_df.columns)
    assert label_map == {
        1: "neoplasms",
        2: "digestive system diseases",
        3: "nervous system diseases",
        4: "cardiovascular diseases",
        5: "general pathological conditions",
    }


# --- Testes de validate ---


@pytest.fixture
def label_map():
    return {1: "a", 2: "b"}


@pytest.fixture
def dados_validos():
    return pd.DataFrame(
        {
            "condition_label": [1, 2, 1, 2],
            "medical_abstract": ["texto a", "texto b", "texto c", "texto d"],
        }
    )


def test_validate_data_aceita_dados_validos(dados_validos, label_map):
    validate_data(dados_validos, label_map)  # não deve levantar


def test_validate_data_rejeita_dataset_vazio(label_map):
    vazio = pd.DataFrame({"condition_label": [], "medical_abstract": []})
    with pytest.raises(AssertionError, match="vazio"):
        validate_data(vazio, label_map)


def test_validate_data_rejeita_coluna_faltando(label_map):
    sem_coluna = pd.DataFrame({"condition_label": [1, 2]})
    with pytest.raises(AssertionError, match="Colunas faltando"):
        validate_data(sem_coluna, label_map)


def test_validate_data_rejeita_texto_nulo(label_map):
    com_nulo = pd.DataFrame(
        {"condition_label": [1, 2], "medical_abstract": ["texto", None]}
    )
    with pytest.raises(AssertionError, match="nulos"):
        validate_data(com_nulo, label_map)


def test_validate_data_rejeita_label_desconhecido(dados_validos, label_map):
    dados_validos.loc[0, "condition_label"] = 99
    with pytest.raises(AssertionError, match="fora do mapa"):
        validate_data(dados_validos, label_map)


def test_validate_data_rejeita_menos_de_duas_classes(label_map):
    uma_classe = pd.DataFrame(
        {"condition_label": [1, 1, 1], "medical_abstract": ["a", "b", "c"]}
    )
    with pytest.raises(AssertionError, match="Menos de 2 classes"):
        validate_data(uma_classe, label_map)
