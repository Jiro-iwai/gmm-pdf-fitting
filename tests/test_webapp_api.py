"""
Tests for Web Application API endpoints.
"""

import pytest
import json
import numpy as np
from io import BytesIO

# Add src directory to path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

# Try to import TestClient, skip tests if httpx is not available
try:
    from fastapi.testclient import TestClient
    from webapp.api import app
    TESTCLIENT_AVAILABLE = True
except ImportError:
    TESTCLIENT_AVAILABLE = False
    pytestmark = pytest.mark.skip(reason="httpx not installed, required for TestClient")

if TESTCLIENT_AVAILABLE:
    client = TestClient(app)


class TestHealthEndpoint:
    """Tests for /api/health endpoint."""
    
    @pytest.mark.skipif(not TESTCLIENT_AVAILABLE, reason="httpx not available")
    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestRootEndpoint:
    """Tests for root endpoint."""
    
    @pytest.mark.skipif(not TESTCLIENT_AVAILABLE, reason="httpx not available")
    def test_root(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data


class TestComputeEndpoint:
    """Tests for /api/compute endpoint."""
    
    @pytest.mark.skipif(not TESTCLIENT_AVAILABLE, reason="httpx not available")
    def test_compute_em_method(self):
        """Test compute endpoint with EM method."""
        request_data = {
            "bivariate_params": {
                "mu_x": 0.0,
                "sigma_x": 0.8,
                "mu_y": 0.0,
                "sigma_y": 1.6,
                "rho": 0.9
            },
            "grid_params": {
                "z_range": [-6.0, 8.0],
                "z_npoints": 100  # Small for faster test
            },
            "K": 3,
            "method": "em",
            "em_params": {
                "max_iter": 100,
                "tol": 1e-6,
                "n_init": 2,
                "init": "quantile",
                "use_moment_matching": False
            }
        }
        
        response = client.post("/api/compute", json=request_data)
        if response.status_code != 200:
            print(f"Response status: {response.status_code}")
            print(f"Response body: {response.text}")
            # Try to get error details
            try:
                error_data = response.json()
                print(f"Error details: {error_data}")
            except:
                pass
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()
        
        assert data["success"] is True
        assert data["method"] == "em"
        assert "z" in data
        assert "f_true" in data
        assert "f_hat" in data
        assert "gmm_components" in data
        assert "statistics_true" in data
        assert "statistics_hat" in data
        assert "error_metrics" in data
        assert "execution_time" in data
        
        # Check data types and structure
        assert isinstance(data["z"], list)
        assert isinstance(data["f_true"], list)
        assert isinstance(data["f_hat"], list)
        assert len(data["z"]) == len(data["f_true"]) == len(data["f_hat"])
        assert len(data["gmm_components"]) == 3
        
        # Check statistics
        assert "mean" in data["statistics_true"]
        assert "std" in data["statistics_true"]
        assert "skewness" in data["statistics_true"]
        assert "kurtosis" in data["statistics_true"]
        
        # Check execution time
        assert "total_time" in data["execution_time"]
        assert data["execution_time"]["em_time"] is not None
    
    @pytest.mark.skipif(not TESTCLIENT_AVAILABLE, reason="httpx not available")
    def test_compute_lp_method(self):
        """Test compute endpoint with LP method."""
        request_data = {
            "bivariate_params": {
                "mu_x": 0.1,
                "sigma_x": 0.4,
                "mu_y": 0.15,
                "sigma_y": 0.9,
                "rho": 0.9
            },
            "grid_params": {
                "z_range": [-4, 4],
                "z_npoints": 64  # Small for faster test
            },
            "K": 5,
            "method": "lp",
            "lp_params": {
                "L": 5,
                "objective_mode": "pdf",
                "solver": "highs"
            }
        }
        
        response = client.post("/api/compute", json=request_data)
        if response.status_code != 200:
            print(f"Response status: {response.status_code}")
            print(f"Response body: {response.text}")
            try:
                error_data = response.json()
                print(f"Error details: {error_data}")
            except:
                pass
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()
        
        assert data["success"] is True
        assert data["method"] == "lp"
        assert "z" in data
        assert "f_true" in data
        assert "f_hat" in data
        assert data["execution_time"]["lp_time"] is not None
    
    @pytest.mark.skipif(not TESTCLIENT_AVAILABLE, reason="httpx not available")
    def test_compute_hybrid_method(self):
        """Test compute endpoint with Hybrid method."""
        request_data = {
            "bivariate_params": {
                "mu_x": 0.0,
                "sigma_x": 0.8,
                "mu_y": 0.0,
                "sigma_y": 1.6,
                "rho": 0.9
            },
            "grid_params": {
                "z_range": [-4, 4],
                "z_npoints": 64  # Small for faster test
            },
            "K": 5,
            "method": "hybrid",
            "hybrid_params": {
                "dict_J": 20,
                "dict_L": 5,
                "objective_mode": "raw_moments"
            },
            "em_params": {
                "max_iter": 50,
                "n_init": 1,
                "init": "custom"
            }
        }
        
        response = client.post("/api/compute", json=request_data)
        if response.status_code != 200:
            print(f"Response status: {response.status_code}")
            print(f"Response body: {response.text}")
            try:
                error_data = response.json()
                print(f"Error details: {error_data}")
            except:
                pass
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()
        
        assert data["success"] is True
        assert data["method"] == "hybrid"
        assert data["execution_time"]["lp_time"] is not None
        assert data["execution_time"]["em_time"] is not None
    
    @pytest.mark.skipif(not TESTCLIENT_AVAILABLE, reason="httpx not available")
    def test_compute_invalid_params(self):
        """Test compute endpoint with invalid parameters."""
        request_data = {
            "bivariate_params": {
                "mu_x": 0.0,
                "sigma_x": -0.8,  # Invalid: negative sigma
                "mu_y": 0.0,
                "sigma_y": 1.6,
                "rho": 0.9
            },
            "grid_params": {
                "z_range": [-6.0, 8.0],
                "z_npoints": 100
            },
            "K": 3,
            "method": "em"
        }
        
        response = client.post("/api/compute", json=request_data)
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.skipif(not TESTCLIENT_AVAILABLE, reason="httpx not available")
    def test_compute_invalid_z_range(self):
        """Test compute endpoint with invalid z_range."""
        request_data = {
            "bivariate_params": {
                "mu_x": 0.0,
                "sigma_x": 0.8,
                "mu_y": 0.0,
                "sigma_y": 1.6,
                "rho": 0.9
            },
            "grid_params": {
                "z_range": [8.0, -6.0],  # Invalid: z_min > z_max
                "z_npoints": 100
            },
            "K": 3,
            "method": "em"
        }
        
        response = client.post("/api/compute", json=request_data)
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.skipif(not TESTCLIENT_AVAILABLE, reason="httpx not available")
    def test_compute_invalid_rho(self):
        """Test compute endpoint with invalid rho."""
        request_data = {
            "bivariate_params": {
                "mu_x": 0.0,
                "sigma_x": 0.8,
                "mu_y": 0.0,
                "sigma_y": 1.6,
                "rho": 1.5  # Invalid: rho > 1
            },
            "grid_params": {
                "z_range": [-6.0, 8.0],
                "z_npoints": 100
            },
            "K": 3,
            "method": "em"
        }
        
        response = client.post("/api/compute", json=request_data)
        assert response.status_code == 422  # Validation error


class TestLoadConfigEndpoint:
    """Tests for /api/load-config endpoint."""
    
    @pytest.mark.skipif(not TESTCLIENT_AVAILABLE, reason="httpx not available")
    def test_load_config_valid(self):
        """Test loading a valid config file."""
        config_data = {
            "mu_x": 0.0,
            "sigma_x": 0.8,
            "mu_y": 0.0,
            "sigma_y": 1.6,
            "rho": 0.9,
            "z_range": [-6.0, 8.0],
            "z_npoints": 2500,
            "K": 3,
            "method": "em",
            "max_iter": 400,
            "tol": 1e-10,
            "n_init": 8,
            "init": "quantile",
            "use_moment_matching": False
        }
        
        # Create a file-like object
        file_content = json.dumps(config_data).encode("utf-8")
        files = {"file": ("config.json", BytesIO(file_content), "application/json")}
        
        response = client.post("/api/load-config", files=files)
        assert response.status_code == 200
        data = response.json()
        
        # Check that the response contains the expected structure
        assert "bivariate_params" in data
        assert "grid_params" in data
        assert "K" in data
        assert "method" in data
        assert data["bivariate_params"]["mu_x"] == 0.0
        assert data["bivariate_params"]["sigma_x"] == 0.8
        assert data["K"] == 3
        assert data["method"] == "em"
    
    @pytest.mark.skipif(not TESTCLIENT_AVAILABLE, reason="httpx not available")
    def test_load_config_invalid_json(self):
        """Test loading an invalid JSON file."""
        file_content = b"invalid json content"
        files = {"file": ("config.json", BytesIO(file_content), "application/json")}
        
        response = client.post("/api/load-config", files=files)
        assert response.status_code == 400
    
    @pytest.mark.skipif(not TESTCLIENT_AVAILABLE, reason="httpx not available")
    def test_load_config_missing_file(self):
        """Test load-config endpoint without file."""
        response = client.post("/api/load-config")
        assert response.status_code == 422  # Missing file parameter


class TestAPIErrorHandling:
    """Tests for API error handling."""
    
    @pytest.mark.skipif(not TESTCLIENT_AVAILABLE, reason="httpx not available")
    def test_compute_with_infeasible_lp(self):
        """Test compute endpoint with parameters that may cause LP infeasibility."""
        # Use parameters that might cause issues
        request_data = {
            "bivariate_params": {
                "mu_x": 0.0,
                "sigma_x": 0.1,  # Very small sigma
                "mu_y": 0.0,
                "sigma_y": 0.1,
                "rho": 0.99  # Very high correlation
            },
            "grid_params": {
                "z_range": [-1, 1],
                "z_npoints": 32
            },
            "K": 10,
            "method": "lp",
            "lp_params": {
                "L": 10,
                "objective_mode": "raw_moments",
                "pdf_tolerance": 0.001  # Very strict tolerance
            }
        }
        
        response = client.post("/api/compute", json=request_data)
        # Should either succeed or return a proper error response
        assert response.status_code in [200, 500]
        if response.status_code == 500:
            data = response.json()
            assert "detail" in data or "message" in data

