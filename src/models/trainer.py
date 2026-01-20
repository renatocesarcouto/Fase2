"""
Model trainer module for Medical AI Diagnosis System v2.0

Handles training of Logistic Regression and Random Forest models.
"""
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
from typing import Literal, Dict

from utils.logger import model_logger
from utils.config import (
    LR_CONFIG,
    RF_CONFIG,
    MODEL_LOGISTIC_REGRESSION,
    MODEL_RANDOM_FOREST,
)


class ModelTrainer:
    """
    Trains and saves ML models for breast cancer diagnosis.

    Supports:
    - Logistic Regression
    - Random Forest
    """

    def __init__(self, model_type: Literal["logistic_regression", "random_forest"]):
        """
        Initialize trainer.

        Args:
            model_type: Type of model to train
        """
        self.model_type = model_type
        self.model = self._create_model()

    def _create_model(self):
        """Create model instance based on type."""
        if self.model_type == "logistic_regression":
            model_logger.info("Creating Logistic Regression model")
            return LogisticRegression(**LR_CONFIG)
        elif self.model_type == "random_forest":
            model_logger.info("Creating Random Forest model")
            return RandomForestClassifier(**RF_CONFIG)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train model on training data.

        Args:
            X_train: Training features (scaled)
            y_train: Training labels

        Raises:
            Exception: If training fails
        """
        try:
            model_logger.info(
                f"Training {self.model_type} on {len(X_train)} samples"
            )

            # Train model
            self.model.fit(X_train, y_train)

            model_logger.info(f"{self.model_type} trained successfully")

            # Log training score
            train_score = self.model.score(X_train, y_train)
            model_logger.info(f"Training accuracy: {train_score:.4f}")

        except Exception as e:
            model_logger.error(f"Training failed: {e}")
            raise

    def save_model(self, path: Path = None) -> None:
        """
        Save trained model to disk.

        Args:
            path: Path to save model (default: config paths)

        Raises:
            ValueError: If model not trained
        """
        if not hasattr(self.model, "classes_"):
            raise ValueError("Model not trained. Call train() first.")

        # Use default path if not provided
        if path is None:
            if self.model_type == "logistic_regression":
                path = MODEL_LOGISTIC_REGRESSION
            elif self.model_type == "random_forest":
                path = MODEL_RANDOM_FOREST

        model_logger.info(f"Saving {self.model_type} to {path}")
        joblib.dump(self.model, path)
        model_logger.info("Model saved successfully")

    def get_model(self):
        """Get trained model instance."""
        return self.model
