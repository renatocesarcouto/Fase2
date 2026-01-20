"""
Model evaluator module for Medical AI Diagnosis System v2.0

Handles evaluation metrics calculation and SHAP analysis.
"""
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from typing import Dict, Tuple

from utils.logger import model_logger
from utils.config import FEATURE_NAMES, SHAP_TOP_FEATURES


class ModelEvaluator:
    """
    Evaluates model performance and generates interpretability analysis.

    Responsibilities:
    - Calculate medical metrics (sensitivity, specificity, etc)
    - Generate SHAP analysis
    - Compare with baseline metrics
    """

    def __init__(self, model):
        """
        Initialize evaluator.

        Args:
            model: Trained sklearn model
        """
        self.model = model

    def calculate_medical_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray = None,
    ) -> Dict:
        """
        Calculate comprehensive medical metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional, for AUC)

        Returns:
            Dictionary with all metrics
        """
        model_logger.info("Calculating medical metrics")

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)  # Sensitivity
        f1 = f1_score(y_true, y_pred)

        # Medical metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate

        # ROC AUC (if probabilities available)
        roc_auc = None
        if y_pred_proba is not None:
            roc_auc = roc_auc_score(y_true, y_pred_proba)

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall_sensitivity": float(recall),
            "specificity": float(specificity),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc) if roc_auc is not None else None,
            "false_negative_rate": float(fnr),
            "false_positive_rate": float(fpr),
            "confusion_matrix": {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            },
            "total_samples": int(len(y_true)),
        }

        # Log key metrics
        model_logger.info(f"Accuracy: {accuracy:.4f}")
        model_logger.info(f"Sensitivity: {recall:.4f}")
        model_logger.info(f"Specificity: {specificity:.4f}")
        model_logger.info(f"False Negatives: {fn} (critical)")
        model_logger.info(f"False Positives: {fp}")

        return metrics

    def generate_shap_analysis(
        self,
        X: np.ndarray,
        X_df: pd.DataFrame = None,
    ) -> Dict:
        """
        Generate SHAP analysis for model interpretability.

        Args:
            X: Features (scaled) for SHAP analysis
            X_df: Original DataFrame with feature names (optional)

        Returns:
            Dictionary with SHAP values and feature importance
        """
        model_logger.info("Generating SHAP analysis")

        try:
            # Create SHAP explainer
            explainer = shap.Explainer(self.model, X)

            # Calculate SHAP values
            shap_values = explainer(X)

            # Get feature importance (mean absolute SHAP values)
            if hasattr(shap_values, "values"):
                # For binary classification, get values for positive class
                if len(shap_values.values.shape) == 3:
                    values = shap_values.values[:, :, 1]  # Malignant class
                else:
                    values = shap_values.values

                importance = np.abs(values).mean(axis=0)
            else:
                importance = np.abs(shap_values).mean(axis=0)

            # Create feature importance dataframe
            feature_importance = pd.DataFrame({
                "feature": FEATURE_NAMES,
                "importance": importance,
            })
            feature_importance = feature_importance.sort_values(
                "importance", ascending=False
            ).reset_index(drop=True)

            # Top features
            top_features = feature_importance.head(SHAP_TOP_FEATURES)

            model_logger.info(
                f"Top {SHAP_TOP_FEATURES} features by SHAP importance:"
            )
            for idx, row in top_features.iterrows():
                model_logger.info(
                    f"  {idx + 1}. {row['feature']}: {row['importance']:.3f}"
                )

            return {
                "feature_importance": feature_importance.to_dict(orient="records"),
                "top_features": top_features.to_dict(orient="records"),
                "shap_values": shap_values,  # Raw SHAP object (for plotting)
            }

        except Exception as e:
            model_logger.warning(f"SHAP analysis failed: {e}")
            return {
                "feature_importance": None,
                "top_features": None,
                "error": str(e),
            }

    def compare_with_baseline(
        self, metrics: Dict, baseline: Dict = None
    ) -> Dict:
        """
        Compare metrics with Fase 1 baseline.

        Args:
            metrics: Current metrics
            baseline: Baseline metrics (if None, uses Fase 1 defaults)

        Returns:
            Comparison results
        """
        if baseline is None:
            baseline = {
                "accuracy": 0.9737,
                "recall_sensitivity": 0.9861,
                "specificity": 0.9524,
            }

        comparison = {}
        for key in ["accuracy", "recall_sensitivity", "specificity"]:
            current = metrics.get(key, 0)
            target = baseline.get(key, 0)
            diff = current - target

            comparison[key] = {
                "current": current,
                "baseline": target,
                "difference": diff,
                "meets_baseline": diff >= -0.01,  # 1% tolerance
            }

            status = "✅" if diff >= -0.01 else "⚠️"
            model_logger.info(
                f"{status} {key}: {current:.4f} vs {target:.4f} ({diff:+.4f})"
            )

        return comparison
