"""
Tests for ModelTrainer module.

Tests model training, persistence, and configuration.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from src.data.loader import load_breast_cancer_data
from src.data.preprocessor import DataPreprocessor
from src.models.trainer import ModelTrainer
from src.utils.config import RANDOM_STATE, TARGET_ACCURACY, METRIC_TOLERANCE


class TestModelTrainer:
    """Test suite for ModelTrainer class."""
    
    @pytest.fixture
    def prepared_data(self):
        """Load and preprocess data for training."""
        # Load data
        X, y = load_breast_cancer_data()
        
        # Preprocess
        preprocessor = DataPreprocessor()
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
        preprocessor.fit_scaler(X_train)
        
        X_train_scaled = preprocessor.transform(X_train)
        X_test_scaled = preprocessor.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def test_create_logistic_regression(self):
        """Test that Logistic Regression model is created correctly."""
        trainer = ModelTrainer(model_type="logistic_regression")
        
        assert trainer.model is not None
        assert trainer.model_type == "logistic_regression"
        assert trainer.model.random_state == RANDOM_STATE
    
    def test_create_random_forest(self):
        """Test that Random Forest model is created correctly."""
        trainer = ModelTrainer(model_type="random_forest")
        
        assert trainer.model is not None
        assert trainer.model_type == "random_forest"
        assert trainer.model.random_state == RANDOM_STATE
    
    def test_invalid_model_type_raises_error(self):
        """Test that invalid model type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model type"):
            ModelTrainer(model_type="invalid_model")
    
    def test_train_logistic_regression(self, prepared_data):
        """Test that Logistic Regression trains successfully."""
        X_train, X_test, y_train, y_test = prepared_data
        
        trainer = ModelTrainer(model_type="logistic_regression")
        
        # Train model
        trainer.train(X_train, y_train)
        
        # Check model is fitted
        assert hasattr(trainer.model, 'coef_')
        assert hasattr(trainer.model, 'intercept_')
    
    def test_train_random_forest(self, prepared_data):
        """Test that Random Forest trains successfully."""
        X_train, X_test, y_train, y_test = prepared_data
        
        trainer = ModelTrainer(model_type="random_forest")
        
        # Train model
        trainer.train(X_train, y_train)
        
        # Check model is fitted
        assert hasattr(trainer.model, 'estimators_')
        assert len(trainer.model.estimators_) > 0
    
    def test_trained_model_can_predict(self, prepared_data):
        """Test that trained model can make predictions."""
        X_train, X_test, y_train, y_test = prepared_data
        
        trainer = ModelTrainer(model_type="logistic_regression")
        trainer.train(X_train, y_train)
        
        # Make predictions
        predictions = trainer.model.predict(X_test)
        
        # Check predictions
        assert len(predictions) == len(y_test)
        assert set(predictions).issubset({0, 1})  # Binary predictions
    
    def test_trained_model_accuracy_in_range(self, prepared_data):
        """Test that trained model achieves target accuracy."""
        X_train, X_test, y_train, y_test = prepared_data
        
        trainer = ModelTrainer(model_type="logistic_regression")
        trainer.train(X_train, y_train)
        
        # Calculate accuracy
        predictions = trainer.model.predict(X_test)
        accuracy = (predictions == y_test).mean()
        
        # Check accuracy close to target (within tolerance)
        assert accuracy >= TARGET_ACCURACY - METRIC_TOLERANCE - 0.02, f"Accuracy too low: {accuracy:.4f}"
        assert accuracy >= 0.95, f"Accuracy below 95%: {accuracy:.4f}"
    
    def test_model_persistence(self, prepared_data):
        """Test that model can be saved and loaded."""
        X_train, X_test, y_train, y_test = prepared_data
        
        # Train model
        trainer = ModelTrainer(model_type="logistic_regression")
        trainer.train(X_train, y_train)
        
        # Make predictions before saving
        predictions_before = trainer.model.predict(X_test)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            trainer.save_model(tmp_path)
            
            # Load into new trainer
            trainer2 = ModelTrainer(model_type="logistic_regression")
            import joblib
            trainer2.model = joblib.load(tmp_path)
            
            # Make predictions after loading
            predictions_after = trainer2.model.predict(X_test)
            
            # Check identical predictions
            np.testing.assert_array_equal(predictions_before, predictions_after)
            
        finally:
            # Cleanup
            Path(tmp_path).unlink(missing_ok=True)
    
    def test_save_untrained_model_raises_error(self):
        """Test that saving untrained model raises ValueError."""
        trainer = ModelTrainer(model_type="logistic_regression")
        
        with tempfile.NamedTemporaryFile(suffix='.joblib') as tmp:
            with pytest.raises(ValueError, match="Model not trained"):
                trainer.save_model(tmp.name)
    
    def test_get_model(self, prepared_data):
        """Test that get_model returns the trained model."""
        X_train, X_test, y_train, y_test = prepared_data
        
        trainer = ModelTrainer(model_type="logistic_regression")
        trainer.train(X_train, y_train)
        
        model = trainer.get_model()
        
        assert model is not None
        assert model is trainer.model
    
    def test_random_state_consistency(self, prepared_data):
        """Test that models with same random_state produce same results."""
        X_train, X_test, y_train, y_test = prepared_data
        
        # Train first model
        trainer1 = ModelTrainer(model_type="logistic_regression")
        trainer1.train(X_train, y_train)
        predictions1 = trainer1.model.predict(X_test)
        
        # Train second model (same random_state)
        trainer2 = ModelTrainer(model_type="logistic_regression")
        trainer2.train(X_train, y_train)
        predictions2 = trainer2.model.predict(X_test)
        
        # Check identical predictions (deterministic)
        np.testing.assert_array_equal(predictions1, predictions2)
    
    def test_immutable_random_state_in_config(self):
        """Test that RANDOM_STATE in config is 42 (immutable rule)."""
        from src.utils.config import LR_CONFIG, RF_CONFIG
        
        assert LR_CONFIG['random_state'] == 42
        assert RF_CONFIG['random_state'] == 42
    
    def test_logistic_regression_has_probability(self, prepared_data):
        """Test that Logistic Regression can predict probabilities."""
        X_train, X_test, y_train, y_test = prepared_data
        
        trainer = ModelTrainer(model_type="logistic_regression")
        trainer.train(X_train, y_train)
        
        # Predict probabilities
        probabilities = trainer.model.predict_proba(X_test)
        
        # Check shape
        assert probabilities.shape == (len(X_test), 2)
        
        # Check probabilities sum to 1
        assert np.allclose(probabilities.sum(axis=1), 1.0)
        
        # Check probabilities in [0, 1]
        assert (probabilities >= 0).all()
        assert (probabilities <= 1).all()
    
    def test_random_forest_has_probability(self, prepared_data):
        """Test that Random Forest can predict probabilities."""
        X_train, X_test, y_train, y_test = prepared_data
        
        trainer = ModelTrainer(model_type="random_forest")
        trainer.train(X_train, y_train)
        
        # Predict probabilities
        probabilities = trainer.model.predict_proba(X_test)
        
        # Check shape
        assert probabilities.shape == (len(X_test), 2)
        
        # Check probabilities sum to 1
        assert np.allclose(probabilities.sum(axis=1), 1.0)
    
    def test_both_models_achieve_high_accuracy(self, prepared_data):
        """Test that both LR and RF achieve >95% accuracy."""
        X_train, X_test, y_train, y_test = prepared_data
        
        # Train Logistic Regression
        trainer_lr = ModelTrainer(model_type="logistic_regression")
        trainer_lr.train(X_train, y_train)
        accuracy_lr = (trainer_lr.model.predict(X_test) == y_test).mean()
        
        # Train Random Forest
        trainer_rf = ModelTrainer(model_type="random_forest")
        trainer_rf.train(X_train, y_train)
        accuracy_rf = (trainer_rf.model.predict(X_test) == y_test).mean()
        
        # Both should be >95%
        assert accuracy_lr >= 0.95, f"LR accuracy too low: {accuracy_lr:.4f}"
        assert accuracy_rf >= 0.95, f"RF accuracy too low: {accuracy_rf:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
