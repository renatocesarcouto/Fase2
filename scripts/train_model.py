#!/usr/bin/env python3
"""
Training script for Medical AI Diagnosis System v2.0

Trains model, evaluates on test set, generates SHAP analysis, and saves everything.

Usage:
    python scripts/train_model.py --model logistic_regression
    python scripts/train_model.py --model random_forest
"""
import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.loader import load_breast_cancer_data
from data.preprocessor import DataPreprocessor
from models.trainer import ModelTrainer
from models.evaluator import ModelEvaluator
from utils.logger import model_logger
from utils.config import METADATA_PATH, RANDOM_STATE


def train_pipeline(model_type: str = "logistic_regression"):
    """
    Complete training pipeline.

    Args:
        model_type: Type of model to train

    Returns:
        Dictionary with training results
    """
    model_logger.info("=" * 60)
    model_logger.info("MEDICAL AI DIAGNOSIS SYSTEM - TRAINING PIPELINE")
    model_logger.info("=" * 60)

    # 1. Load data
    model_logger.info("\n[Step 1/6] Loading dataset...")
    X, y = load_breast_cancer_data()

    # 2. Preprocess
    model_logger.info("\n[Step 2/6] Preprocessing data...")
    preprocessor = DataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)

    # Scale
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_val_scaled = preprocessor.transform(X_val)
    X_test_scaled = preprocessor.transform(X_test)

    # Save scaler
    preprocessor.save_scaler()

    # 3. Train model
    model_logger.info(f"\n[Step 3/6] Training {model_type}...")
    trainer = ModelTrainer(model_type=model_type)
    trainer.train(X_train_scaled, y_train.values)

    # Validation score
    val_score = trainer.get_model().score(X_val_scaled, y_val.values)
    model_logger.info(f"Validation accuracy: {val_score:.4f}")

    # Save model
    trainer.save_model()

    # 4. Evaluate on test set
    model_logger.info("\n[Step 4/6] Evaluating on test set...")
    evaluator = ModelEvaluator(trainer.get_model())

    y_pred = trainer.get_model().predict(X_test_scaled)
    y_pred_proba = trainer.get_model().predict_proba(X_test_scaled)[:, 1]

    metrics = evaluator.calculate_medical_metrics(
        y_test.values, y_pred, y_pred_proba
    )

    # 5. SHAP analysis
    model_logger.info("\n[Step 5/6] Generating SHAP analysis...")
    shap_results = evaluator.generate_shap_analysis(X_test_scaled, X_test)

    # 6. Compare with baseline
    model_logger.info("\n[Step 6/6] Comparing with Fase 1 baseline...")
    comparison = evaluator.compare_with_baseline(metrics)

    # Save metadata
    metadata = {
        "model_type": model_type,
        "random_state": RANDOM_STATE,
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "metrics": metrics,
        "comparison": comparison,
        "top_features": shap_results.get("top_features"),
    }

    METADATA_PATH.parent.mkdir(exist_ok=True)
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    model_logger.info(f"\n✅ Training complete! Metadata saved to {METADATA_PATH}")

    # Summary
    model_logger.info("\n" + "=" * 60)
    model_logger.info("TRAINING SUMMARY")
    model_logger.info("=" * 60)
    model_logger.info(f"Model: {model_type}")
    model_logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    model_logger.info(f"Sensitivity: {metrics['recall_sensitivity']:.4f}")
    model_logger.info(f"Specificity: {metrics['specificity']:.4f}")
    model_logger.info(f"False Negatives: {metrics['confusion_matrix']['false_negatives']}")
    model_logger.info(f"False Positives: {metrics['confusion_matrix']['false_positives']}")
    model_logger.info("=" * 60)

    return metadata


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train Medical AI Diagnosis model"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["logistic_regression", "random_forest"],
        default="logistic_regression",
        help="Model type to train",
    )

    args = parser.parse_args()

    try:
        train_pipeline(model_type=args.model)
    except Exception as e:
        model_logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
