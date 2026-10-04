"""Ingestão de dados - carrega o Medical Abstracts TC Corpus."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[int, str]]:
    """Carrega train/test/labels do Medical Abstracts TC Corpus.

    Returns:
        tuple: (train_df, test_df, label_map), onde label_map mapeia
        condition_label (int) -> condition_name (str).
    """
    logger.info("Carregando dataset Medical Abstracts TC Corpus...")
    train_df = pd.read_csv(DATA_DIR / "medical_tc_train.csv")
    test_df = pd.read_csv(DATA_DIR / "medical_tc_test.csv")
    labels_df = pd.read_csv(DATA_DIR / "medical_tc_labels.csv")

    label_map = dict(
        zip(labels_df["condition_label"], labels_df["condition_name"], strict=True)
    )
    logger.info(f"Treino: {len(train_df)} amostras | Teste: {len(test_df)} amostras")
    return train_df, test_df, label_map
