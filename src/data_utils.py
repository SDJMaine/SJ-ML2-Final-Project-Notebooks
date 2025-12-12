from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


@dataclass
class EmotionDatasetSplits:
    """
    Container for emotion dataset splits and label mappings.
    """
    train_texts: List[str]
    train_labels: List[int]
    val_texts: List[str]
    val_labels: List[int]
    test_texts: List[str]
    test_labels: List[int]
    label2id: Dict[str, int]
    id2label: Dict[int, str]


# -------------------------------------------------------------------
# 1. Loading CSV splits
# -------------------------------------------------------------------

def load_emotion_csv_splits(
        data_dir: Path,
        train_name: str = "training.csv",
        val_name: str = "validation.csv",
        test_name: str = "test.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the emotion dataset CSV splits as pandas DataFrames.

    :param data_dir: Path to the directory containing the CSV files.
    :param train_name: Filename of the training split.
    :param val_name: Filename of the validation split.
    :param test_name: Filename of the test split.
    :return: (train_df, val_df, test_df)
    """
    data_dir = Path(data_dir)

    train_df = pd.read_csv(data_dir / train_name)
    val_df = pd.read_csv(data_dir / val_name)
    test_df = pd.read_csv(data_dir / test_name)

    return train_df, val_df, test_df


# -------------------------------------------------------------------
# 2. Label mappings (common across all models)
# -------------------------------------------------------------------

def build_label_mappings(labels: pd.Series) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build consistent label <-> id mappings from a label Series.

    The mapping is built from the unique labels in sorted order to
    keep it stable.

    :param labels: Pandas Series of string labels (e.g., 'anger', 'joy').
    :return: (label2id, id2label)
    """
    unique_labels = sorted(labels.unique())
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def encode_labels(
        labels: pd.Series,
        label2id: Dict[str, int],
) -> List[int]:
    """
    Encode string labels as integer ids using a provided mapping.

    :param labels: Pandas Series of string labels.
    :param label2id: Mapping from label string to integer id.
    :return: List of integer label ids.
    """
    return [label2id[label] for label in labels]


# -------------------------------------------------------------------
# 3. text normalization
# -------------------------------------------------------------------

def basic_text_normalization(text: str) -> str:
    """
    Basic text normalization used across models where appropriate.

    :param text: Input tweet text.
    :return: Normalized text.
    """
    if not isinstance(text, str):
        text = str(text)
    return text.strip()


def normalize_texts(texts: pd.Series) -> List[str]:
    """
    Apply basic_text_normalization to a Series of texts.

    :param texts: Pandas Series of tweet text.
    :return: List of normalized strings.
    """
    return [basic_text_normalization(t) for t in texts]


# -------------------------------------------------------------------
# 4. High-level helper for all splits
# -------------------------------------------------------------------

def load_and_prepare_emotion_splits(
        data_dir: Path,
        normalize: bool = True,
) -> EmotionDatasetSplits:
    """
    Load emotion dataset CSVs, build label mappings, and return
    standardized splits ready for modeling.

    This function:
      - Loads train/val/test DataFrames.
      - Builds label2id/id2label from the train labels.
      - Encodes labels to integers for all splits.
      - applies light text normalization.

    :param data_dir: Directory containing train/val/test CSVs.
    :param normalize: Whether to apply basic_text_normalization.
    :return: EmotionDatasetSplits dataclass with texts, labels, mappings.
    """
    train_df, val_df, test_df = load_emotion_csv_splits(data_dir)

    # Build mappings from training labels only (standard practice)
    label2id, id2label = build_label_mappings(train_df["label"])

    # Extract and optionally normalize texts
    if normalize:
        train_texts = normalize_texts(train_df["text"])
        val_texts = normalize_texts(val_df["text"])
        test_texts = normalize_texts(test_df["text"])
    else:
        train_texts = train_df["text"].astype(str).tolist()
        val_texts = val_df["text"].astype(str).tolist()
        test_texts = test_df["text"].astype(str).tolist()

    # Encode labels
    train_labels = encode_labels(train_df["label"], label2id)
    val_labels = encode_labels(val_df["label"], label2id)
    test_labels = encode_labels(test_df["label"], label2id)

    return EmotionDatasetSplits(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        test_texts=test_texts,
        test_labels=test_labels,
        label2id=label2id,
        id2label=id2label,
    )