"""
Data preprocessor module for Medical AI Diagnosis System v2.0

Handles data splitting, scaling, and preprocessing.
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict
from pathlib import Path

from utils.logger import data_logger
from utils.config import (
    RANDOM_STATE,
    TRAIN_SIZE,
    VAL_SIZE,
    TEST_SIZE,
    STRATIFY,
    SCALER_PATH,
)


class DataPreprocessor:
    """
    Handles all data preprocessing operations.

    Responsibilities:
    - Train/val/test splitting (stratified)
    - Feature scaling (StandardScaler)
    - Saving/loading scaler
    """

    def __init__(self):
        """Initialize preprocessor."""
        self.scaler = StandardScaler()
        self._is_fitted = False

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Split data into train/val/test sets (60/20/20).

        Args:
            X: Features (DataFrame)
            y: Labels (Series)

        Returns:
            Tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        data_logger.info("Splitting data into train/val/test (60/20/20)")

        # First split: 80% temp (train+val), 20% test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            stratify=y if STRATIFY else None,
            random_state=RANDOM_STATE,
        )

        # Second split: 75% train, 25% val (of temp)
        # 75% of 80% = 60% total (train)
        # 25% of 80% = 20% total (val)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=VAL_SIZE / (TRAIN_SIZE + VAL_SIZE),
            stratify=y_temp if STRATIFY else None,
            random_state=RANDOM_STATE,
        )

        data_logger.info(f"Train set: {len(X_train)} samples")
        data_logger.info(f"Val set: {len(X_val)} samples")
        data_logger.info(f"Test set: {len(X_test)} samples")

        # Validate proportions
        total = len(X)
        train_ratio = len(X_train) / total
        val_ratio = len(X_val) / total
        test_ratio = len(X_test) / total

        data_logger.info(
            f"Actual ratios - Train: {train_ratio:.2%}, "
            f"Val: {val_ratio:.2%}, Test: {test_ratio:.2%}"
        )

        # Validate class distributions
        self._log_class_distribution("Train", y_train)
        self._log_class_distribution("Val", y_val)
        self._log_class_distribution("Test", y_test)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def fit_scaler(self, X_train: pd.DataFrame) -> None:
        """
        Fit StandardScaler on training data.

        Args:
            X_train: Training features

        Important: Only fit on training data to avoid data leakage
        """
        data_logger.info("Fitting StandardScaler on training data")
        self.scaler.fit(X_train)
        self._is_fitted = True
        data_logger.info("StandardScaler fitted successfully")

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted scaler.

        Args:
            X: Features to transform

        Returns:
            Scaled features (numpy array)

        Raises:
            ValueError: If scaler not fitted
        """
        if not self._is_fitted:
            raise ValueError("Scaler not fitted. Call fit_scaler() first.")

        data_logger.debug(f"Transforming {len(X)} samples")
        return self.scaler.transform(X)

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        """
        Fit scaler and transform training data.

        Args:
            X_train: Training features

        Returns:
            Scaled training features
        """
        self.fit_scaler(X_train)
        return self.transform(X_train)

    def save_scaler(self, path: Path = SCALER_PATH) -> None:
        """
        Save fitted scaler to disk.

        Args:
            path: Path to save scaler (default: config.SCALER_PATH)

        Raises:
            ValueError: If scaler not fitted
        """
        if not self._is_fitted:
            raise ValueError("Scaler not fitted. Cannot save.")

        data_logger.info(f"Saving scaler to {path}")
        joblib.dump(self.scaler, path)
        data_logger.info("Scaler saved successfully")

    def load_scaler(self, path: Path = SCALER_PATH) -> None:
        """
        Load fitted scaler from disk.

        Args:
            path: Path to load scaler from

        Raises:
            FileNotFoundError: If scaler file not found
        """
        if not path.exists():
            raise FileNotFoundError(f"Scaler not found at {path}")

        data_logger.info(f"Loading scaler from {path}")
        self.scaler = joblib.load(path)
        self._is_fitted = True
        data_logger.info("Scaler loaded successfully")

    def _log_class_distribution(self, set_name: str, y: pd.Series) -> None:
        """
        Log class distribution for a dataset split.

        Args:
            set_name: Name of the set (Train/Val/Test)
            y: Labels
        """
        dist = y.value_counts()
        total = len(y)
        benign = dist.get(0, 0)
        malignant = dist.get(1, 0)

        data_logger.info(
            f"{set_name} class distribution: "
            f"Benign={benign} ({benign/total:.1%}), "
            f"Malignant={malignant} ({malignant/total:.1%})"
        )
