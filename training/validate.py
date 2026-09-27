"""Validação de qualidade dos dados antes do treino."""
import logging

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"condition_label", "medical_abstract"}


def validate_data(df: pd.DataFrame, label_map: dict[int, str]) -> None:
    """Valida que o dataset está pronto para treino/avaliação.

    Args:
        df: DataFrame com colunas condition_label e medical_abstract.
        label_map: mapa condition_label -> condition_name conhecido.

    Raises:
        AssertionError: se alguma validação falhar.
    """
    logger.info("Validando dados...")

    assert len(df) > 0, "Dataset vazio!"
    assert REQUIRED_COLUMNS.issubset(df.columns), (
        f"Colunas faltando: {REQUIRED_COLUMNS - set(df.columns)}"
    )
    assert df["medical_abstract"].isna().sum() == 0, "Textos nulos em medical_abstract!"
    assert df["condition_label"].isna().sum() == 0, "Labels nulos em condition_label!"

    labels_desconhecidos = set(df["condition_label"].unique()) - set(label_map.keys())
    assert not labels_desconhecidos, (
        f"Labels fora do mapa conhecido: {labels_desconhecidos}"
    )
    assert df["condition_label"].nunique() >= 2, "Menos de 2 classes no dataset!"

    logger.info(
        f"✅ Dados válidos: {len(df)} amostras, "
        f"{df['condition_label'].nunique()} classes"
    )
