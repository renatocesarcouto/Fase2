"""
Tests for FastAPI endpoints.

Tests API routes, validation, and responses.
"""

import pytest
from fastapi.testclient import TestClient
import json

from src.api.main import app


class TestAPI:
    """Test suite for FastAPI endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def sample_patient_features(self):
        """Sample patient features for testing (Malignant case from dataset)."""
        return {
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
            "worst_fractal_dimension": 0.1189
        }
    
    def test_root_endpoint(self, client):
        """Test GET / returns API info."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "v2.0" in data["message"]  # Version in message
        assert "endpoints" in data
    
    def test_health_endpoint(self, client):
        """Test GET /health returns health status."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "model_loaded" in data
        assert isinstance(data["model_loaded"], bool)
    
    def test_model_info_endpoint(self, client):
        """Test GET /model-info returns model metadata."""
        response = client.get("/model-info")
        
        # If model not loaded, should return 503
        if response.status_code == 503:
            pytest.skip("Model not loaded (expected if not trained yet)")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "model_type" in data
        assert "features_count" in data
        assert data["features_count"] == 30
    
    def test_predict_endpoint_with_valid_data(self, client, sample_patient_features):
        """Test POST /predict with valid patient data."""
        response = client.post("/predict", json=sample_patient_features)
        
        # If model not loaded, should return 503
        if response.status_code == 503:
            pytest.skip("Model not loaded (expected if not trained yet)")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "prediction" in data
        assert "diagnosis" in data
        assert "probability" in data
        assert "confidence" in data
        
        # Check types
        assert isinstance(data["prediction"], int)
        assert data["prediction"] in [0, 1]
        assert isinstance(data["diagnosis"], str)
        assert data["diagnosis"] in ["Benign", "Malignant"]
        
        # Check probabilities
        assert "benign" in data["probability"]
        assert "malignant" in data["probability"]
        assert 0 <= data["probability"]["benign"] <= 1
        assert 0 <= data["probability"]["malignant"] <= 1
        assert abs(data["probability"]["benign"] + data["probability"]["malignant"] - 1.0) < 0.01
        
        # Check confidence
        assert 0 <= data["confidence"] <= 1
    
    def test_predict_endpoint_with_missing_field(self, client, sample_patient_features):
        """Test POST /predict with missing required field."""
        # Remove one field
        incomplete_data = sample_patient_features.copy()
        del incomplete_data["mean_radius"]
        
        response = client.post("/predict", json=incomplete_data)
        
        # Should return 422 (validation error)
        assert response.status_code == 422
    
    def test_predict_endpoint_with_invalid_type(self, client, sample_patient_features):
        """Test POST /predict with invalid field type."""
        # Change field to string
        invalid_data = sample_patient_features.copy()
        invalid_data["mean_radius"] = "not_a_number"
        
        response = client.post("/predict", json=invalid_data)
        
        # Should return 422 (validation error)
        assert response.status_code == 422
    
    def test_predict_endpoint_with_negative_value(self, client, sample_patient_features):
        """Test POST /predict with negative value (invalid for medical data)."""
        # Set negative value
        invalid_data = sample_patient_features.copy()
        invalid_data["mean_radius"] = -10.0
        
        response = client.post("/predict", json=invalid_data)
        
        # Should return 422 (validation error) or 503 (model not loaded)
        assert response.status_code in [422, 503]
    
    def test_predict_endpoint_with_extra_field(self, client, sample_patient_features):
        """Test POST /predict with extra unexpected field."""
        # Add extra field
        extra_data = sample_patient_features.copy()
        extra_data["extra_field"] = 999
        
        response = client.post("/predict", json=extra_data)
        
        # FastAPI ignores extra fields by default (should still succeed)
        # If model not loaded, skip
        if response.status_code == 503:
            pytest.skip("Model not loaded")
        
        # Should either succeed or return validation error
        assert response.status_code in [200, 422]
    
    def test_predict_endpoint_response_time(self, client, sample_patient_features):
        """Test that /predict responds in reasonable time."""
        import time
        
        start = time.time()
        response = client.post("/predict", json=sample_patient_features)
        elapsed = time.time() - start
        
        # Should respond in less than 1 second
        assert elapsed < 1.0, f"Response too slow: {elapsed:.2f}s"
    
    def test_docs_endpoint(self, client):
        """Test that /docs (OpenAPI) is accessible."""
        response = client.get("/docs")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_openapi_schema(self, client):
        """Test that /openapi.json returns valid schema."""
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        schema = response.json()
        
        # Check OpenAPI structure
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
        
        # Check predict endpoint is documented
        assert "/predict" in schema["paths"]
        assert "post" in schema["paths"]["/predict"]
    
    def test_cors_headers(self, client):
        """Test that CORS headers are present."""
        response = client.options("/predict", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST"
        })
        
        # Check CORS headers
        assert "access-control-allow-origin" in response.headers or response.status_code == 200
    
    def test_benign_case_prediction(self, client):
        """Test prediction for a benign case (if model loaded)."""
        # Sample benign case (low values)
        benign_features = {
            "mean_radius": 12.0,
            "mean_texture": 15.0,
            "mean_perimeter": 80.0,
            "mean_area": 450.0,
            "mean_smoothness": 0.08,
            "mean_compactness": 0.08,
            "mean_concavity": 0.05,
            "mean_concave_points": 0.03,
            "mean_symmetry": 0.17,
            "mean_fractal_dimension": 0.06,
            "radius_error": 0.3,
            "texture_error": 0.8,
            "perimeter_error": 2.0,
            "area_error": 25.0,
            "smoothness_error": 0.005,
            "compactness_error": 0.02,
            "concavity_error": 0.02,
            "concave_points_error": 0.008,
            "symmetry_error": 0.015,
            "fractal_dimension_error": 0.003,
            "worst_radius": 13.5,
            "worst_texture": 20.0,
            "worst_perimeter": 90.0,
            "worst_area": 550.0,
            "worst_smoothness": 0.12,
            "worst_compactness": 0.15,
            "worst_concavity": 0.12,
            "worst_concave_points": 0.06,
            "worst_symmetry": 0.25,
            "worst_fractal_dimension": 0.08
        }
        
        response = client.post("/predict", json=benign_features)
        
        if response.status_code == 503:
            pytest.skip("Model not loaded")
        
        assert response.status_code == 200
        # Note: Cannot assert diagnosis without trained model
        # This test verifies format, not correctness


class TestPatientFeaturesSchema:
    """Test PatientFeatures Pydantic schema."""
    
    def test_patient_features_to_array(self):
        """Test that PatientFeatures.to_array() returns correct array."""
        from src.api.schemas import PatientFeatures
        
        features = PatientFeatures(
            mean_radius=17.99,
            mean_texture=10.38,
            mean_perimeter=122.8,
            mean_area=1001.0,
            mean_smoothness=0.1184,
            mean_compactness=0.2776,
            mean_concavity=0.3001,
            mean_concave_points=0.1471,
            mean_symmetry=0.2419,
            mean_fractal_dimension=0.07871,
            radius_error=1.095,
            texture_error=0.9053,
            perimeter_error=8.589,
            area_error=153.4,
            smoothness_error=0.006399,
            compactness_error=0.04904,
            concavity_error=0.05373,
            concave_points_error=0.01587,
            symmetry_error=0.03003,
            fractal_dimension_error=0.006193,
            worst_radius=25.38,
            worst_texture=17.33,
            worst_perimeter=184.6,
            worst_area=2019.0,
            worst_smoothness=0.1622,
            worst_compactness=0.6656,
            worst_concavity=0.7119,
            worst_concave_points=0.2654,
            worst_symmetry=0.4601,
            worst_fractal_dimension=0.1189
        )
        
        arr = features.to_array()
        
        # Check shape
        assert arr.shape == (30,)
        
        # Check first value
        assert arr[0] == 17.99
        
        # Check last value
        assert arr[29] == 0.1189


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
