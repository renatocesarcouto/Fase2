"""
Tests for DataPreprocessor module.

Tests split ratios, stratification, scaling, and persistence.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from src.data.loader import load_breast_cancer_data
from src.data.preprocessor import DataPreprocessor
from src.utils.config import TRAIN_SIZE, VAL_SIZE, TEST_SIZE, RANDOM_STATE


class TestDataPreprocessor:
    """Test suite for DataPreprocessor class."""
    
    @pytest.fixture
    def sample_data(self):
        """Load real Wisconsin dataset for testing."""
        X, y = load_breast_cancer_data()
        return X, y
    
    @pytest.fixture
    def preprocessor(self):
        """Create DataPreprocessor instance."""
        return DataPreprocessor()
    
    def test_split_data_ratios(self, preprocessor, sample_data):
        """Test that split_data returns correct split ratios."""
        X, y = sample_data
        
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
        
        total_samples = len(X)
        
        # Check train split (60%)
        expected_train = int(total_samples * TRAIN_SIZE)
        assert len(X_train) == expected_train
        assert len(y_train) == expected_train
        
        # Check val split (20%) - Allow ±1 for rounding
        expected_val = int(total_samples * VAL_SIZE)
        assert abs(len(X_val) - expected_val) <= 1
        assert abs(len(y_val) - expected_val) <= 1
        
        # Check test split (20%) - Calculate based on actual remaining
        assert len(X_test) == total_samples - len(X_train) - len(X_val)
        assert len(y_test) == total_samples - len(y_train) - len(y_val)
        
        # Check total adds up
        assert len(X_train) + len(X_val) + len(X_test) == total_samples
    
    def test_split_data_stratification(self, preprocessor, sample_data):
        """Test that stratification maintains class proportions."""
        X, y = sample_data
        
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
        
        # Original proportions
        original_ratio = y.value_counts(normalize=True)
        
        # Train proportions
        train_ratio = pd.Series(y_train).value_counts(normalize=True)
        
        # Val proportions
        val_ratio = pd.Series(y_val).value_counts(normalize=True)
        
        # Test proportions
        test_ratio = pd.Series(y_test).value_counts(normalize=True)
        
        # Check all splits maintain proportions (within 5% tolerance)
        tolerance = 0.05
        for cls in original_ratio.index:
            assert abs(train_ratio[cls] - original_ratio[cls]) < tolerance
            assert abs(val_ratio[cls] - original_ratio[cls]) < tolerance
            assert abs(test_ratio[cls] - original_ratio[cls]) < tolerance
    
    def test_split_data_reproducibility(self, preprocessor, sample_data):
        """Test that splits are reproducible with same random_state."""
        X, y = sample_data
        
        # First split
        X_train1, X_val1, X_test1, y_train1, y_val1, y_test1 = preprocessor.split_data(X, y)
        
        # Second split (same preprocessor)
        preprocessor2 = DataPreprocessor()
        X_train2, X_val2, X_test2, y_train2, y_val2, y_test2 = preprocessor2.split_data(X, y)
        
        # Check identical splits
        pd.testing.assert_frame_equal(X_train1, X_train2)
        pd.testing.assert_frame_equal(X_val1, X_val2)
        pd.testing.assert_frame_equal(X_test1, X_test2)
        pd.testing.assert_series_equal(pd.Series(y_train1), pd.Series(y_train2))
    
    def test_fit_scaler(self, preprocessor, sample_data):
        """Test that scaler fits correctly on training data."""
        X, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
        
        # Fit scaler
        preprocessor.fit_scaler(X_train)
        
        # Check scaler is fitted
        assert preprocessor.scaler is not None
        assert hasattr(preprocessor.scaler, 'mean_')
        assert hasattr(preprocessor.scaler, 'scale_')
        
        # Check mean and scale shapes
        assert len(preprocessor.scaler.mean_) == X_train.shape[1]
        assert len(preprocessor.scaler.scale_) == X_train.shape[1]
    
    def test_transform(self, preprocessor, sample_data):
        """Test that transform scales data correctly."""
        X, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
        
        # Fit and transform
        preprocessor.fit_scaler(X_train)
        X_train_scaled = preprocessor.transform(X_train)
        
        # Check scaled data has mean ≈ 0 and std ≈ 1
        # transform() returns numpy array, not DataFrame
        assert abs(X_train_scaled.mean(axis=0)).max() < 0.1  # Mean close to 0
        assert abs(X_train_scaled.std(axis=0) - 1).max() < 0.1  # Std close to 1
        
        # Check shape preserved
        assert X_train_scaled.shape == X_train.shape
    
    def test_transform_without_fit_raises_error(self, preprocessor, sample_data):
        """Test that transform without fit raises ValueError."""
        X, y = sample_data
        
        with pytest.raises(ValueError, match="Scaler not fitted"):
            preprocessor.transform(X)
    
    def test_scaler_persistence(self, preprocessor, sample_data):
        """Test that scaler can be saved and loaded."""
        X, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
        
        # Fit scaler
        preprocessor.fit_scaler(X_train)
        
        # Transform data
        X_train_scaled = preprocessor.transform(X_train)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            preprocessor.save_scaler(tmp_path)
            
            # Load into new preprocessor
            preprocessor2 = DataPreprocessor()
            preprocessor2.load_scaler(Path(tmp_path))
            
            # Transform with loaded scaler
            X_train_scaled2 = preprocessor2.transform(X_train)
            
            # Check identical results (both are numpy arrays)
            np.testing.assert_array_almost_equal(X_train_scaled, X_train_scaled2)
            
        finally:
            # Cleanup
            Path(tmp_path).unlink(missing_ok=True)
    
    def test_save_scaler_without_fit_raises_error(self, preprocessor):
        """Test that saving unfitted scaler raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix='.joblib') as tmp:
            with pytest.raises(ValueError, match="Scaler not fitted"):
                preprocessor.save_scaler(tmp.name)
    
    def test_load_nonexistent_scaler_raises_error(self, preprocessor):
        """Test that loading nonexistent scaler raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            preprocessor.load_scaler(Path("/nonexistent/path/scaler.joblib"))
    
    def test_immutable_random_state(self):
        """Test that RANDOM_STATE is immutable (always 42)."""
        assert RANDOM_STATE == 42, "RANDOM_STATE must be 42 (immutable rule)"
    
    def test_split_ratios_sum_to_one(self):
        """Test that split ratios sum to 1.0."""
        total = TRAIN_SIZE + VAL_SIZE + TEST_SIZE
        assert abs(total - 1.0) < 1e-10, "Split ratios must sum to 1.0"
    
    def test_no_data_leakage(self, preprocessor, sample_data):
        """Test that train/val/test sets have no overlap."""
        X, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
        
        # Get indices
        train_idx = set(X_train.index)
        val_idx = set(X_val.index)
        test_idx = set(X_test.index)
        
        # Check no overlap
        assert train_idx.isdisjoint(val_idx), "Train and val sets overlap!"
        assert train_idx.isdisjoint(test_idx), "Train and test sets overlap!"
        assert val_idx.isdisjoint(test_idx), "Val and test sets overlap!"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
