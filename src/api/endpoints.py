"""
FastAPI endpoints for Medical AI Diagnosis System v2.0

Provides REST API for breast cancer diagnosis prediction.
"""
from fastapi import APIRouter, HTTPException
from typing import Dict

from api.schemas import (
    PatientFeatures,
    PredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)
from models.predictor import ModelPredictor
from utils.logger import api_logger
from utils.config import (
    DEFAULT_MODEL,
    MODEL_VERSION,
    NUM_FEATURES,
    TARGET_ACCURACY,
    TARGET_SENSITIVITY,
    TARGET_SPECIFICITY,
)

# Create router
router = APIRouter()

# Global predictor (loaded on startup)
predictor: ModelPredictor = None


def initialize_predictor(model_type: str = DEFAULT_MODEL):
    """
    Initialize global predictor with model and scaler.

    Args:
        model_type: Type of model to load

    Raises:
        Exception: If initialization fails
    """
    global predictor

    try:
        api_logger.info(f"Initializing predictor with {model_type}")
        predictor = ModelPredictor(model_type=model_type)
        predictor.load_model()
        predictor.load_scaler()
        api_logger.info("Predictor initialized successfully")
    except Exception as e:
        api_logger.error(f"Failed to initialize predictor: {e}")
        raise


@router.post("/predict", response_model=PredictionResponse)
async def predict_diagnosis(features: PatientFeatures) -> PredictionResponse:
    """
    Predict breast cancer diagnosis based on cytological features.

    **Medical AI Support Tool - Not a replacement for medical diagnosis**

    Args:
        features: Patient cytological features (30 values)

    Returns:
        Prediction with diagnosis, probabilities, and confidence

    Raises:
        HTTPException: If prediction fails
    """
    if predictor is None or not predictor.is_ready():
        api_logger.error("Predictor not initialized")
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service unavailable.",
        )

    try:
        api_logger.info("Received prediction request")

        # Convert features to array
        features_array = features.to_array()

        # Predict
        result = predictor.predict(features_array, return_proba=True)

        api_logger.info(
            f"Prediction: {result['diagnosis']} "
            f"(confidence: {result['confidence']:.2%})"
        )

        # LLM Explanation
        try:
            from llm.interpreter import LLMInterpreter
            # Instantiate interpreter
            # INTEGRATION: We try to use Ollama first (True). If it fails (e.g. no service), 
            # the interpreter automatically falls back to the template based logic.
            interpreter = LLMInterpreter(use_ollama=True) 
            
            # Get approximated SHAP/Feature importance
            # Reshape to 2D (1 sample) to avoid sklearn warnings/errors
            features_2d = features_array.reshape(1, -1)
            features_scaled = predictor.scaler.transform(features_2d)
            shap_approx = predictor.get_feature_contribution(features_scaled)
            
            # Generate explanation
            explanation = interpreter.generate_explanation(
                prediction=result['prediction'],
                probability=result['confidence'],
                shap_features=shap_approx
            )
            result['explanation'] = explanation
            
        except Exception as llm_error:
            api_logger.warning(f"LLM Explanation failed: {llm_error}")
            result['explanation'] = "Explicação indisponível no momento."

        return PredictionResponse(**result)

    except Exception as e:
        api_logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns:
        API health status and model information
    """
    model_loaded = predictor is not None and predictor.is_ready()

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_type=predictor.model_type if predictor else "none",
        version=MODEL_VERSION,
    )


@router.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info() -> ModelInfoResponse:
    """
    Get information about loaded model.

    Returns:
        Model metadata and baseline metrics
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded",
        )

    return ModelInfoResponse(
        model_type=predictor.model_type,
        model_version=MODEL_VERSION,
        features_count=NUM_FEATURES,
        baseline_metrics={
            "accuracy": TARGET_ACCURACY,
            "sensitivity": TARGET_SENSITIVITY,
            "specificity": TARGET_SPECIFICITY,
        },
    )


@router.get("/")
async def root():
    """
    Root endpoint with API information.

    Returns:
        Welcome message and API links
    """
    return {
        "message": "Medical AI Diagnosis System API v2.0",
        "description": "Breast cancer diagnosis prediction using ML",
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)",
            "model_info": "/model-info (GET)",
            "docs": "/docs (GET)",
        },
        "warning": "This is a support tool, not a replacement for medical diagnosis",
    }
