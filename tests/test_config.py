"""
Tests for configuration and setup functions.
"""
import json
import tempfile
import os
import pytest
from gmm_fitting import (
    load_config,
    prepare_init_params,
    DEFAULT_MU_X,
    DEFAULT_SIGMA_X
)


class TestLoadConfig:
    """Tests for load_config function."""
    
    def test_load_config_defaults(self):
        """Test loading config with defaults."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({}, f)
            config_path = f.name
        
        try:
            config = load_config(config_path)
            
            assert config['mu_x'] == DEFAULT_MU_X
            assert config['sigma_x'] == DEFAULT_SIGMA_X
            assert config['K'] == 3
            assert config['init'] == "quantile"
        finally:
            os.unlink(config_path)
    
    def test_load_config_custom_values(self):
        """Test loading config with custom values."""
        custom_config = {
            "mu_x": 1.0,
            "sigma_x": 0.5,
            "K": 5,
            "init": "qmi"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(custom_config, f)
            config_path = f.name
        
        try:
            config = load_config(config_path)
            
            assert config['mu_x'] == 1.0
            assert config['sigma_x'] == 0.5
            assert config['K'] == 5
            assert config['init'] == "qmi"
        finally:
            os.unlink(config_path)
    
    def test_load_config_missing_file(self):
        """Test loading config when file doesn't exist."""
        config = load_config("nonexistent_file.json")
        
        # Should use defaults
        assert config['mu_x'] == DEFAULT_MU_X
        assert config['sigma_x'] == DEFAULT_SIGMA_X
    
    def test_load_config_invalid_json(self):
        """Test loading config with invalid JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json }")
            config_path = f.name
        
        try:
            with pytest.raises(json.JSONDecodeError):
                load_config(config_path)
        finally:
            os.unlink(config_path)
    
    def test_load_config_all_parameters(self):
        """Test loading config with all parameters."""
        full_config = {
            "mu_x": 0.1,
            "sigma_x": 0.4,
            "mu_y": 0.15,
            "sigma_y": 0.9,
            "rho": 0.9,
            "z_range": [-4, 4],
            "z_npoints": 128,
            "K": 4,
            "max_iter": 20000,
            "tol": 1e-6,
            "reg_var": 1e-6,
            "n_init": 4,
            "seed": 1,
            "init": "quantile",
            "use_moment_matching": True,
            "qp_mode": "hard",
            "soft_lambda": 1e4,
            "output_path": "test_output",
            "show_grid_points": True,
            "max_grid_points_display": 200
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(full_config, f)
            config_path = f.name
        
        try:
            config = load_config(config_path)
            
            for key, value in full_config.items():
                assert config[key] == value
        finally:
            os.unlink(config_path)


class TestPrepareInitParams:
    """Tests for prepare_init_params function."""
    
    def test_prepare_init_params_wqmi(self):
        """Test preparing init params for WQMI."""
        config = {"_raw_config": {}}
        mu_x, sigma_x = 0.1, 0.4
        mu_y, sigma_y = 0.15, 0.9
        rho = 0.9
        
        init_params = prepare_init_params(
            config, "wqmi", mu_x, sigma_x, mu_y, sigma_y, rho
        )
        
        assert init_params is not None
        assert init_params['mu_x'] == mu_x
        assert init_params['var_x'] == sigma_x**2
        assert init_params['mu_y'] == mu_y
        assert init_params['var_y'] == sigma_y**2
        assert init_params['rho'] == rho
    
    def test_prepare_init_params_non_wqmi(self):
        """Test preparing init params for non-WQMI methods."""
        config = {}
        
        for init_method in ["quantile", "random", "qmi"]:
            init_params = prepare_init_params(
                config, init_method, 0.0, 1.0, 0.0, 1.0, 0.0
            )
            
            assert init_params is None
    
    def test_prepare_init_params_override(self):
        """Test that init_params can be overridden from config."""
        config = {
            "_raw_config": {
                "init_params": {
                    "mu_x": 999.0,  # Override value
                    "mass_floor": 1e-20
                }
            }
        }
        
        init_params = prepare_init_params(
            config, "wqmi", 0.1, 0.4, 0.15, 0.9, 0.9
        )
        
        assert init_params is not None
        assert init_params['mu_x'] == 999.0  # Overridden
        assert init_params['mass_floor'] == 1e-20  # From override
        assert init_params['var_x'] == 0.4**2  # Not overridden

