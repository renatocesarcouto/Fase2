"""
Model predictor module for Medical AI Diagnosis System v2.0

Handles model loading and inference.
"""
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Tuple

from utils.logger import model_logger
from utils.config import (
    get_model_path,
    SCALER_PATH,
    CLASS_NAMES,
    DEFAULT_MODEL,
)


class ModelPredictor:
    """
    Loads trained model and performs inference.

    Responsibilities:
    - Load model and scaler from disk
    - Predict diagnosis for new patients
    - Return probabilities and class labels
    """

    def __init__(self, model_type: str = DEFAULT_MODEL):
        """
        Initialize predictor.

        Args:
            model_type: Type of model to load ("logistic_regression" or "random_forest")
        """
        self.model_type = model_type
        self.model = None
        self.scaler = None

    def load_model(self, model_path: Path = None) -> None:
        """
        Load trained model from disk.

        Args:
            model_path: Path to model file (default: uses config)

        Raises:
            FileNotFoundError: If model file not found
        """
        if model_path is None:
            model_path = get_model_path(self.model_type)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        model_logger.info(f"Loading {self.model_type} from {model_path}")
        self.model = joblib.load(model_path)
        model_logger.info("Model loaded successfully")

    def load_scaler(self, scaler_path: Path = SCALER_PATH) -> None:
        """
        Load scaler from disk.

        Args:
            scaler_path: Path to scaler file

        Raises:
            FileNotFoundError: If scaler file not found
        """
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")

        model_logger.info(f"Loading scaler from {scaler_path}")
        self.scaler = joblib.load(scaler_path)
        model_logger.info("Scaler loaded successfully")

    def predict(
        self, features: np.ndarray, return_proba: bool = True
    ) -> Dict:
        """
        Predict diagnosis for patient features.

        Args:
            features: Patient features (30 values, unscaled or scaled)
            return_proba: Whether to return probabilities

        Returns:
            Dictionary with prediction results

        Raises:
            ValueError: If model or scaler not loaded
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Reshape if single sample
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Scale features if scaler available
        if self.scaler is not None:
            features_scaled = self.scaler.transform(features)
        else:
            model_logger.warning("Scaler not loaded, using unscaled features")
            features_scaled = features

        # Predict
        prediction = self.model.predict(features_scaled)
        proba = self.model.predict_proba(features_scaled) if return_proba else None

        # Build response
        results = []
        for i in range(len(prediction)):
            pred_class = int(prediction[i])
            result = {
                "prediction": pred_class,
                "diagnosis": CLASS_NAMES[pred_class],
            }

            if proba is not None:
                result["probability"] = {
                    "benign": float(proba[i][0]),
                    "malignant": float(proba[i][1]),
                }
                result["confidence"] = float(max(proba[i]))

            results.append(result)

        if len(results) == 1:
            return results[0]

        return {"predictions": results}

    def get_feature_contribution(self, features_scaled: np.ndarray) -> Dict[str, float]:
        """
        Calculate simplified feature importance for a single sample.
        Since real SHAP is slow for real-time API without TreeExplainer optimization,
        we roughly approximate contribution = coefficient * scaled_value for Linear models.
        For RF, we might need TreeExplainer (which is heavy).
        
        To keep API fast (<500ms), we use this approximation for the demo.
        """
        if self.model_type == "logistic_regression":
            coeffs = self.model.coef_[0]
            # Contribution = coeff * value
            contributions = coeffs * features_scaled.flatten()
            
            # Map to feature names (from config or 30 defaults)
            from utils.config import FEATURE_NAMES
            return dict(zip(FEATURE_NAMES, contributions))
            
        elif self.model_type == "random_forest":
            # RF feature importance is global, not local. 
            # Real SHAP is best, but slow.
            # Fallback: return global importance * value direction
            importances = self.model.feature_importances_
            # This is a heuristic for demo speed:
            from utils.config import FEATURE_NAMES
            return dict(zip(FEATURE_NAMES, importances * features_scaled.flatten()))
            
        return {}

    def is_ready(self) -> bool:
        """
        Check if predictor is ready for inference.

        Returns:
            True if model and scaler are loaded
        """
        return self.model is not None and self.scaler is not None
