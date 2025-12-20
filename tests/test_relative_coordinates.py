"""Tests for relative coordinate mode in MDN training and inference.

This module tests the relative coordinate system where:
- Training data is generated with μ_x = 0 (fixed)
- Inference applies coordinate transformation (shift by M1)
"""
import numpy as np
import pytest
import tempfile
from pathlib import Path

# Skip tests if PyTorch is not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")

from src.gmm_fitting import max_pdf_bivariate_normal, normalize_pdf_on_grid


# =============================================================================
# Test: Dataset Generation with Relative Coordinates
# =============================================================================

class TestDatasetRelativeCoordinates:
    """Tests for dataset generation with μ_x = 0 fixed."""
    
    def test_generate_dataset_mu_x_is_zero(self):
        """Test that generated dataset has μ_x = 0 for all samples."""
        from src.ml_init.dataset import generate_dataset, COORDINATE_MODE
        
        # Skip if not in relative mode
        if COORDINATE_MODE != "relative":
            pytest.skip("Dataset not in relative coordinate mode")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            generate_dataset(
                output_dir=output_dir,
                n_train=100,
                n_val=10,
                n_test=10,
                seed_train=0,
                seed_val=1,
                seed_test=2,
            )
            
            # Load and check
            train_data = np.load(output_dir / "train.npz")
            params = train_data["params"]
            
            # params format: (mu_x, sigma_x, mu_y, sigma_y, rho)
            mu_x = params[:, 0]
            
            # All μ_x should be 0
            assert np.allclose(mu_x, 0.0), f"μ_x should be 0, got {mu_x[:5]}"
    
    def test_generate_dataset_delta_mu_range(self):
        """Test that Δμ = μ_y - μ_x covers the expected range."""
        from src.ml_init.dataset import generate_dataset, COORDINATE_MODE, DELTA_MU_RANGE
        
        if COORDINATE_MODE != "relative":
            pytest.skip("Dataset not in relative coordinate mode")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            generate_dataset(
                output_dir=output_dir,
                n_train=1000,  # More samples to check range coverage
                n_val=10,
                n_test=10,
                seed_train=42,
                seed_val=1,
                seed_test=2,
            )
            
            train_data = np.load(output_dir / "train.npz")
            params = train_data["params"]
            
            mu_x = params[:, 0]
            mu_y = params[:, 2]
            delta_mu = mu_y - mu_x  # Should equal mu_y since mu_x = 0
            
            # Check range
            assert np.all(delta_mu >= DELTA_MU_RANGE[0]), \
                f"Δμ min {delta_mu.min():.2f} < {DELTA_MU_RANGE[0]}"
            assert np.all(delta_mu <= DELTA_MU_RANGE[1]), \
                f"Δμ max {delta_mu.max():.2f} > {DELTA_MU_RANGE[1]}"
            
            # Check that range is reasonably covered
            assert delta_mu.min() < DELTA_MU_RANGE[0] + 1.0, \
                "Δμ should cover lower range"
            assert delta_mu.max() > DELTA_MU_RANGE[1] - 1.0, \
                "Δμ should cover upper range"
    
    def test_pdf_centered_at_origin(self):
        """Test that generated PDFs have M1 close to 0."""
        from src.ml_init.dataset import generate_dataset, COORDINATE_MODE
        
        if COORDINATE_MODE != "relative":
            pytest.skip("Dataset not in relative coordinate mode")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            generate_dataset(
                output_dir=output_dir,
                n_train=100,
                n_val=10,
                n_test=10,
                seed_train=0,
                seed_val=1,
                seed_test=2,
            )
            
            train_data = np.load(output_dir / "train.npz")
            z = train_data["z"]
            f = train_data["f"]
            
            # Compute M1 for each sample
            m1_values = []
            for i in range(len(f)):
                m1 = np.trapezoid(z * f[i], z)
                m1_values.append(m1)
            
            m1_values = np.array(m1_values)
            
            # M1 should be relatively small (centered around 0)
            # Note: M1 won't be exactly 0 because E[max(X,Y)] != 0 even when μ_x=μ_y=0
            # But it should be much smaller than when μ_x, μ_y vary over [-3, 3]
            # V5 has wider parameter ranges (Δμ ∈ [-10, 10]), so M1 can be larger
            assert np.abs(np.mean(m1_values)) < 5.0, \
                f"Mean M1 = {np.mean(m1_values):.2f}, expected < 5.0"


# =============================================================================
# Test: Coordinate Transformation in Inference
# =============================================================================

class TestInferenceCoordinateTransform:
    """Tests for coordinate transformation in inference."""
    
    def test_compute_m1(self):
        """Test M1 (first moment) computation."""
        from src.ml_init.infer import _compute_m1
        
        z = np.linspace(-8, 8, 64)
        
        # Test 1: Standard normal (M1 ≈ 0)
        f = np.exp(-z**2 / 2) / np.sqrt(2 * np.pi)
        f = f / np.trapezoid(f, z)
        m1 = _compute_m1(z, f)
        assert np.abs(m1) < 0.01, f"Standard normal M1 = {m1}, expected ~0"
        
        # Test 2: Shifted normal (M1 ≈ 2)
        f = np.exp(-(z - 2)**2 / 2) / np.sqrt(2 * np.pi)
        f = f / np.trapezoid(f, z)
        m1 = _compute_m1(z, f)
        assert np.abs(m1 - 2.0) < 0.1, f"Shifted normal M1 = {m1}, expected ~2"
    
    def test_shift_pdf_to_relative(self):
        """Test PDF shifting to relative coordinates."""
        from src.ml_init.infer import _compute_m1, _shift_pdf_to_relative
        
        z = np.linspace(-8, 8, 64)
        
        # Create a shifted PDF
        mu_shift = 2.0
        f_orig = np.exp(-(z - mu_shift)**2 / 2) / np.sqrt(2 * np.pi)
        f_orig = f_orig / np.trapezoid(f_orig, z)
        
        m1_orig = _compute_m1(z, f_orig)
        assert np.abs(m1_orig - mu_shift) < 0.1
        
        # Shift to relative coordinates
        f_shifted = _shift_pdf_to_relative(z, f_orig, m1_orig, z)
        
        # M1 of shifted PDF should be ~0
        m1_shifted = _compute_m1(z, f_shifted)
        assert np.abs(m1_shifted) < 0.1, \
            f"Shifted PDF M1 = {m1_shifted}, expected ~0"
        
        # PDF should still be normalized
        integral = np.trapezoid(f_shifted, z)
        assert np.abs(integral - 1.0) < 0.01, \
            f"Shifted PDF integral = {integral}, expected 1.0"
    
    def test_coordinate_transform_preserves_shape(self):
        """Test that coordinate transformation preserves PDF shape."""
        from src.ml_init.infer import _compute_m1, _shift_pdf_to_relative
        
        z = np.linspace(-8, 8, 64)
        
        # Two PDFs with same shape but different positions
        sigma = 1.0
        f1 = np.exp(-(z - 0)**2 / (2 * sigma**2))
        f1 = f1 / np.trapezoid(f1, z)
        
        f2 = np.exp(-(z - 1.5)**2 / (2 * sigma**2))
        f2 = f2 / np.trapezoid(f2, z)
        
        # Shift both to relative coordinates
        m1_1 = _compute_m1(z, f1)
        m1_2 = _compute_m1(z, f2)
        
        f1_shifted = _shift_pdf_to_relative(z, f1, m1_1, z)
        f2_shifted = _shift_pdf_to_relative(z, f2, m1_2, z)
        
        # Both shifted PDFs should have similar shape
        # (allowing for some interpolation error)
        max_diff = np.max(np.abs(f1_shifted - f2_shifted))
        assert max_diff < 0.05, \
            f"Shape difference = {max_diff}, expected < 0.05"


# =============================================================================
# Test: Model Metadata
# =============================================================================

class TestModelMetadata:
    """Tests for model metadata with coordinate mode."""
    
    def test_metadata_includes_coordinate_mode(self):
        """Test that trained model metadata includes coordinate_mode."""
        from src.ml_init.dataset import generate_dataset
        from src.ml_init.train import train_mdn_model
        import json
        
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            output_dir = Path(tmpdir) / "checkpoints"
            data_dir.mkdir()
            output_dir.mkdir()
            
            # Generate dataset
            generate_dataset(
                output_dir=data_dir,
                n_train=50,
                n_val=10,
                n_test=10,
                seed_train=0,
                seed_val=1,
                seed_test=2,
            )
            
            # Train model
            train_mdn_model(
                data_dir=data_dir,
                output_dir=output_dir,
                batch_size=16,
                epochs=1,
            )
            
            # Check metadata
            metadata_path = output_dir / "metadata.json"
            assert metadata_path.exists()
            
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            assert "coordinate_mode" in metadata, \
                "metadata should include coordinate_mode"
            assert metadata["coordinate_mode"] in ["absolute", "relative"], \
                f"coordinate_mode should be 'absolute' or 'relative', got {metadata['coordinate_mode']}"


# =============================================================================
# Test: End-to-End with Relative Coordinates
# =============================================================================

class TestEndToEndRelativeCoordinates:
    """End-to-end tests for relative coordinate mode."""
    
    def test_same_shape_gives_same_relative_output(self):
        """Test that PDFs with same shape give same output after coordinate transform."""
        from src.ml_init.dataset import generate_dataset, COORDINATE_MODE
        from src.ml_init.train import train_mdn_model
        from src.ml_init.infer import mdn_predict_init, _compute_m1
        
        if COORDINATE_MODE != "relative":
            pytest.skip("Dataset not in relative coordinate mode")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            output_dir = Path(tmpdir) / "checkpoints"
            data_dir.mkdir()
            output_dir.mkdir()
            
            # Generate dataset and train
            generate_dataset(
                output_dir=data_dir,
                n_train=100,
                n_val=10,
                n_test=10,
                seed_train=0,
                seed_val=1,
                seed_test=2,
            )
            
            train_mdn_model(
                data_dir=data_dir,
                output_dir=output_dir,
                batch_size=16,
                epochs=5,  # More epochs for better training
            )
            
            # Get N from the dataset (V5 uses N=96)
            train_data = np.load(data_dir / "train.npz")
            N = len(train_data["z"])
            K = 5
            
            # Model file name is based on N from data
            model_path = output_dir / f"mdn_init_v1_N{N}_K{K}.pt"
            z = np.linspace(-15, 15, N)  # Use V5 grid range
            
            # Two PDFs with same relative parameters but different absolute positions
            # Same: Δμ = 1.0, σ_x = 1.0, σ_y = 0.8, ρ = 0.5
            f1 = max_pdf_bivariate_normal(z, 0.0, 1.0, 1.0, 0.8, 0.5)
            f1 = normalize_pdf_on_grid(z, f1)
            
            f2 = max_pdf_bivariate_normal(z, 1.5, 1.0, 2.5, 0.8, 0.5)
            f2 = normalize_pdf_on_grid(z, f2)
            
            # Get predictions (mdn_predict_init expects directory path, not file path)
            result1 = mdn_predict_init(z, f1, K, str(output_dir), device="cpu")
            result2 = mdn_predict_init(z, f2, K, str(output_dir), device="cpu")
            
            # Convert to relative coordinates for comparison
            m1_1 = _compute_m1(z, f1)
            m1_2 = _compute_m1(z, f2)
            
            mu_rel_1 = result1["mu"] - m1_1
            mu_rel_2 = result2["mu"] - m1_2
            
            # Relative μ should be similar (with some tolerance due to training)
            max_mu_diff = np.max(np.abs(mu_rel_1 - mu_rel_2))
            
            # Note: With a small training set, tolerance needs to be larger
            # In production, this should be < 0.3
            assert max_mu_diff < 1.0, \
                f"Relative μ difference = {max_mu_diff}, expected < 1.0\n" \
                f"μ_rel_1 = {mu_rel_1}\nμ_rel_2 = {mu_rel_2}"
    
    def test_backward_compatibility_absolute_model(self):
        """Test that inference works with models trained in absolute mode."""
        from src.ml_init.infer import mdn_predict_init
        
        # Use existing model (if available)
        model_path = Path("ml_init/checkpoints/mdn_init_v1_N64_K5.pt")
        if not model_path.exists():
            pytest.skip("No existing model for backward compatibility test")
        
        z = np.linspace(-8, 8, 64)
        f = max_pdf_bivariate_normal(z, 0.5, 1.0, 1.0, 0.8, 0.3)
        f = normalize_pdf_on_grid(z, f)
        
        # Should work without errors
        result = mdn_predict_init(z, f, K=5, model_path=str(model_path), device="cpu")
        
        assert "pi" in result
        assert "mu" in result
        assert "var" in result
        assert len(result["pi"]) == 5
        assert np.allclose(np.sum(result["pi"]), 1.0)


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases in coordinate transformation."""
    
    def test_large_m1_shift(self):
        """Test handling of PDFs with large M1 (near grid boundary)."""
        from src.ml_init.infer import _compute_m1, _shift_pdf_to_relative
        
        z = np.linspace(-8, 8, 64)
        
        # PDF with M1 near boundary
        mu_shift = 5.0
        f = np.exp(-(z - mu_shift)**2 / (2 * 0.5**2))
        f = f / np.trapezoid(f, z)
        
        m1 = _compute_m1(z, f)
        
        # Shift should work but may lose some tail
        f_shifted = _shift_pdf_to_relative(z, f, m1, z)
        
        # Should still be valid PDF
        assert np.all(f_shifted >= 0)
        integral = np.trapezoid(f_shifted, z)
        assert integral > 0.9, f"Lost too much mass: integral = {integral}"
    
    def test_very_narrow_pdf(self):
        """Test handling of very narrow PDFs (small variance)."""
        from src.ml_init.infer import _compute_m1, _shift_pdf_to_relative
        
        z = np.linspace(-8, 8, 64)
        
        # Very narrow PDF
        sigma = 0.2
        mu = 1.0
        f = np.exp(-(z - mu)**2 / (2 * sigma**2))
        f = f / np.trapezoid(f, z)
        
        m1 = _compute_m1(z, f)
        f_shifted = _shift_pdf_to_relative(z, f, m1, z)
        
        # M1 should be close to 0
        m1_shifted = _compute_m1(z, f_shifted)
        assert np.abs(m1_shifted) < 0.2, \
            f"Narrow PDF: shifted M1 = {m1_shifted}"

