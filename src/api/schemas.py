"""
Pydantic schemas for Medical AI Diagnosis System v2.0 API

Defines request/response models for API endpoints.
"""
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict
import numpy as np

from utils.config import NUM_FEATURES, FEATURE_NAMES


class PatientFeatures(BaseModel):
    """
    Patient cytological features for diagnosis prediction.

    All 30 features from Wisconsin Breast Cancer Dataset.
    """

    # Mean features
    mean_radius: float = Field(..., description="Mean radius of cells")
    mean_texture: float = Field(..., description="Mean texture (std of gray-scale)")
    mean_perimeter: float = Field(..., description="Mean perimeter of cells")
    mean_area: float = Field(..., description="Mean area of cells")
    mean_smoothness: float = Field(..., description="Mean smoothness")
    mean_compactness: float = Field(..., description="Mean compactness")
    mean_concavity: float = Field(..., description="Mean concavity")
    mean_concave_points: float = Field(..., description="Mean concave points")
    mean_symmetry: float = Field(..., description="Mean symmetry")
    mean_fractal_dimension: float = Field(..., description="Mean fractal dimension")

    # Standard error features
    radius_error: float = Field(..., description="Radius standard error")
    texture_error: float = Field(..., description="Texture standard error")
    perimeter_error: float = Field(..., description="Perimeter standard error")
    area_error: float = Field(..., description="Area standard error")
    smoothness_error: float = Field(..., description="Smoothness standard error")
    compactness_error: float = Field(..., description="Compactness standard error")
    concavity_error: float = Field(..., description="Concavity standard error")
    concave_points_error: float = Field(..., description="Concave points standard error")
    symmetry_error: float = Field(..., description="Symmetry standard error")
    fractal_dimension_error: float = Field(..., description="Fractal dimension standard error")

    # Worst features
    worst_radius: float = Field(..., description="Worst (largest) radius")
    worst_texture: float = Field(..., description="Worst texture")
    worst_perimeter: float = Field(..., description="Worst perimeter")
    worst_area: float = Field(..., description="Worst area")
    worst_smoothness: float = Field(..., description="Worst smoothness")
    worst_compactness: float = Field(..., description="Worst compactness")
    worst_concavity: float = Field(..., description="Worst concavity")
    worst_concave_points: float = Field(..., description="Worst concave points")
    worst_symmetry: float = Field(..., description="Worst symmetry")
    worst_fractal_dimension: float = Field(..., description="Worst fractal dimension")

    def to_array(self) -> np.ndarray:
        """
        Convert to numpy array for model input.

        Returns:
            numpy array with 30 features in correct order
        """
        values = [
            self.mean_radius,
            self.mean_texture,
            self.mean_perimeter,
            self.mean_area,
            self.mean_smoothness,
            self.mean_compactness,
            self.mean_concavity,
            self.mean_concave_points,
            self.mean_symmetry,
            self.mean_fractal_dimension,
            self.radius_error,
            self.texture_error,
            self.perimeter_error,
            self.area_error,
            self.smoothness_error,
            self.compactness_error,
            self.concavity_error,
            self.concave_points_error,
            self.symmetry_error,
            self.fractal_dimension_error,
            self.worst_radius,
            self.worst_texture,
            self.worst_perimeter,
            self.worst_area,
            self.worst_smoothness,
            self.worst_compactness,
            self.worst_concavity,
            self.worst_concave_points,
            self.worst_symmetry,
            self.worst_fractal_dimension,
        ]
        return np.array(values)

    model_config = {
        "json_schema_extra": {
            "example": {
                "mean_radius": 17.99,
                "mean_texture": 10.38,
                "mean_perimeter": 122.8,
                "mean_area": 1001.0,
                "mean_smoothness": 0.1184,
                "mean_compactness": 0.2776,
                "mean_concavity": 0.3001,
                "mean_concave_points": 0.1471,
                "mean_symmetry": 0.2419,
                "mean_fractal_dimension": 0.07871,
                "radius_error": 1.095,
                "texture_error": 0.9053,
                "perimeter_error": 8.589,
                "area_error": 153.4,
                "smoothness_error": 0.006399,
                "compactness_error": 0.04904,
                "concavity_error": 0.05373,
                "concave_points_error": 0.01587,
                "symmetry_error": 0.03003,
                "fractal_dimension_error": 0.006193,
                "worst_radius": 25.38,
                "worst_texture": 17.33,
                "worst_perimeter": 184.6,
                "worst_area": 2019.0,
                "worst_smoothness": 0.1622,
                "worst_compactness": 0.6656,
                "worst_concavity": 0.7119,
                "worst_concave_points": 0.2654,
                "worst_symmetry": 0.4601,
                "worst_fractal_dimension": 0.1189,
            }
        }
    }


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""

    prediction: int = Field(..., description="Predicted class (0=Benign, 1=Malignant)")
    diagnosis: str = Field(..., description="Human-readable diagnosis")
    probability: Dict[str, float] = Field(..., description="Class probabilities")
    confidence: float = Field(..., description="Prediction confidence (max probability)")
    explanation: Optional[str] = Field(None, description="LLM-generated explanation of the diagnosis")


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str = Field(..., description="API status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_type: str = Field(..., description="Type of loaded model")
    version: str = Field(..., description="API version")


class ModelInfoResponse(BaseModel):
    """Response model for model info endpoint."""

    model_type: str = Field(..., description="Type of model")
    model_version: str = Field(..., description="Model version")
    features_count: int = Field(..., description="Number of input features")
    baseline_metrics: Dict[str, float] = Field(..., description="Baseline metrics from Fase 1")
