"""
Configuration module for Medical AI Diagnosis System v2.0

Centralizes all configuration parameters, paths, and hyperparameters.
"""
import os
from pathlib import Path

# ============================================================================
# PROJECT STRUCTURE
# ============================================================================

# Root directory (Fase2/)
ROOT_DIR = Path(__file__).parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"
TESTS_DIR = ROOT_DIR / "tests"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Immutable rule: random_state=42 ALWAYS
RANDOM_STATE = 42

# Model paths
MODEL_LOGISTIC_REGRESSION = MODELS_DIR / "logistic_regression.joblib"
MODEL_RANDOM_FOREST = MODELS_DIR / "random_forest.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"

# Default model for API
DEFAULT_MODEL = "logistic_regression"

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# Dataset source (Wisconsin Breast Cancer from sklearn)
DATASET_NAME = "breast_cancer"

# Train/Val/Test split ratios
TRAIN_SIZE = 0.60
VAL_SIZE = 0.20
TEST_SIZE = 0.20

# Stratification (mandatory for imbalanced data)
STRATIFY = True

# ============================================================================
# ML HYPERPARAMETERS
# ============================================================================

# Logistic Regression
LR_CONFIG = {
    "random_state": RANDOM_STATE,
    "max_iter": 10000,
    "solver": "lbfgs",
}

# Random Forest
RF_CONFIG = {
    "n_estimators": 100,
    "random_state": RANDOM_STATE,
    "max_depth": None,
}

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

# Target metrics (from Fase 1 baseline)
TARGET_ACCURACY = 0.9737
TARGET_SENSITIVITY = 0.9861
TARGET_SPECIFICITY = 0.9524

# Tolerance for metric comparison
METRIC_TOLERANCE = 0.01

# ============================================================================
# API CONFIGURATION
# ============================================================================

# Server settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_RELOAD = os.getenv("API_RELOAD", "true").lower() == "true"

# CORS settings
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# Request limits
MAX_BATCH_SIZE = 100

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = ROOT_DIR / "medical_ai.log"

# ============================================================================
# FEATURE NAMES (Wisconsin Breast Cancer Dataset)
# ============================================================================

FEATURE_NAMES = [
    # Mean features
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",
    # Standard error features
    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error",
    # Worst features
    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension",
]

# Number of features (must be 30)
NUM_FEATURES = len(FEATURE_NAMES)
assert NUM_FEATURES == 30, "Wisconsin dataset must have exactly 30 features"

# ============================================================================
# CLASS LABELS
# ============================================================================

CLASS_NAMES = {
    0: "Benign",
    1: "Malignant",
}

# ============================================================================
# SHAP CONFIGURATION
# ============================================================================

# Number of top features to include in SHAP analysis
SHAP_TOP_FEATURES = 10

# ============================================================================
# CONSTANTS
# ============================================================================

# Model versioning
MODEL_VERSION = "2.0.0"
PROJECT_NAME = "Medical AI Diagnosis System"

# Medical context
SENSITIVITY_PRIORITY = True  # Sensitivity > Accuracy in medical context
INTERPRETABILITY_REQUIRED = True  # SHAP analysis mandatory

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_model_path(model_type: str) -> Path:
    """
    Get path to saved model file.

    Args:
        model_type: "logistic_regression" or "random_forest"

    Returns:
        Path to model file

    Raises:
        ValueError: If model_type is invalid
    """
    if model_type == "logistic_regression":
        return MODEL_LOGISTIC_REGRESSION
    elif model_type == "random_forest":
        return MODEL_RANDOM_FOREST
    else:
        raise ValueError(f"Invalid model_type: {model_type}")


def validate_config():
    """
    Validate configuration integrity.

    Raises:
        AssertionError: If configuration is invalid
    """
    # Check split ratios sum to 1.0
    assert abs(TRAIN_SIZE + VAL_SIZE + TEST_SIZE - 1.0) < 1e-6, (
        "Split ratios must sum to 1.0"
    )

    # Check random state is 42 (immutable rule)
    assert RANDOM_STATE == 42, "RANDOM_STATE must be 42 (immutable rule)"

    # Check feature count
    assert NUM_FEATURES == 30, "Wisconsin dataset must have 30 features"

    # Check model configs have random_state
    assert LR_CONFIG["random_state"] == 42, "LR must use random_state=42"
    assert RF_CONFIG["random_state"] == 42, "RF must use random_state=42"


# Validate on import
validate_config()
