"""
Data loader module for Medical AI Diagnosis System v2.0

Handles loading of Wisconsin Breast Cancer dataset from sklearn.
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from typing import Tuple

from utils.logger import data_logger
from utils.config import FEATURE_NAMES


def load_breast_cancer_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load Wisconsin Breast Cancer dataset from sklearn.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: (X, y)
            X: DataFrame with 30 features (569 samples)
            y: Series with binary labels (0=Benign, 1=Malignant)

    Raises:
        Exception: If dataset loading fails
    """
    try:
        data_logger.info("Loading Wisconsin Breast Cancer dataset from sklearn")

        # Load from sklearn
        data = load_breast_cancer()

        # Create DataFrame with feature names
        X = pd.DataFrame(data.data, columns=FEATURE_NAMES)

        # Create Series for target (already 0/1 encoded)
        # Note: sklearn convention is 0=malignant, 1=benign
        # We need to flip: 0=benign, 1=malignant (medical convention)
        y = pd.Series(1 - data.target, name="diagnosis")

        data_logger.info(
            f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features"
        )
        data_logger.info(f"Class distribution: {y.value_counts().to_dict()}")

        # Validate
        assert X.shape[1] == 30, "Dataset must have 30 features"
        assert len(X) == len(y), "X and y must have same length"
        assert set(y.unique()) == {0, 1}, "y must be binary (0, 1)"

        return X, y

    except Exception as e:
        data_logger.error(f"Failed to load dataset: {e}")
        raise


def get_feature_info() -> pd.DataFrame:
    """
    Get information about dataset features.

    Returns:
        DataFrame with feature metadata
    """
    data = load_breast_cancer()
    return pd.DataFrame({
        "feature": FEATURE_NAMES,
        "description": data.feature_names,
    })
