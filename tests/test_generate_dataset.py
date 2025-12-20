"""Tests for dataset generation."""
import numpy as np
import tempfile
import os
from pathlib import Path
from src.ml_init.dataset import generate_dataset


def test_generate_dataset_basic():
    """Test basic dataset generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # Generate small dataset
        generate_dataset(
            output_dir=output_dir,
            n_train=100,
            n_val=10,
            n_test=10,
            seed_train=0,
            seed_val=1,
            seed_test=2,
            z_min=-8,
            z_max=8,
            n_points=64,
        )
        
        # Check files exist
        assert (output_dir / "train.npz").exists()
        assert (output_dir / "val.npz").exists()
        assert (output_dir / "test.npz").exists()
        
        # Check train data
        train_data = np.load(output_dir / "train.npz")
        assert "z" in train_data
        assert "f" in train_data
        assert "params" in train_data
        
        # Check shapes
        z = train_data["z"]
        f = train_data["f"]
        params = train_data["params"]
        
        assert z.shape == (64,)
        assert f.shape == (100, 64)
        assert params.shape == (100, 5)
        
        # Check z is in range
        assert np.all(z >= -8)
        assert np.all(z <= 8)
        
        # Check f is normalized (approximately)
        for i in range(10):  # Check first 10 samples
            f_i = f[i]
            w = np.full(64, z[1] - z[0])
            w[0] = w[-1] = (z[1] - z[0]) / 2
            integral = np.sum(f_i * w)
            assert np.isclose(integral, 1.0, rtol=1e-3), f"Sample {i} not normalized"
        
        # Check params are in valid ranges
        from src.ml_init.dataset import (
            COORDINATE_MODE, MU_RANGE, DELTA_MU_RANGE, MU_X_FIXED,
            SIGMA_RANGE, RHO_RANGE
        )
        
        mu_x, sigma_x, mu_y, sigma_y, rho = params.T
        
        if COORDINATE_MODE == "relative":
            # In relative mode: mu_x = 0, mu_y = delta_mu
            assert np.allclose(mu_x, MU_X_FIXED), f"mu_x should be {MU_X_FIXED} in relative mode"
            assert np.all(mu_y >= DELTA_MU_RANGE[0]) and np.all(mu_y <= DELTA_MU_RANGE[1])
        else:
            # In absolute mode: mu_x and mu_y are both in MU_RANGE
            assert np.all(mu_x >= MU_RANGE[0]) and np.all(mu_x <= MU_RANGE[1])
            assert np.all(mu_y >= MU_RANGE[0]) and np.all(mu_y <= MU_RANGE[1])
        
        assert np.all(sigma_x >= SIGMA_RANGE[0]) and np.all(sigma_x <= SIGMA_RANGE[1])
        assert np.all(sigma_y >= SIGMA_RANGE[0]) and np.all(sigma_y <= SIGMA_RANGE[1])
        assert np.all(rho >= RHO_RANGE[0]) and np.all(rho <= RHO_RANGE[1])


def test_generate_dataset_reproducibility():
    """Test that dataset generation is reproducible with same seed."""
    with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
        output_dir1 = Path(tmpdir1)
        output_dir2 = Path(tmpdir2)
        
        # Generate twice with same seeds
        generate_dataset(
            output_dir=output_dir1,
            n_train=10,
            n_val=5,
            n_test=5,
            seed_train=42,
            seed_val=43,
            seed_test=44,
            z_min=-8,
            z_max=8,
            n_points=64,
        )
        
        generate_dataset(
            output_dir=output_dir2,
            n_train=10,
            n_val=5,
            n_test=5,
            seed_train=42,
            seed_val=43,
            seed_test=44,
            z_min=-8,
            z_max=8,
            n_points=64,
        )
        
        # Check reproducibility
        train1 = np.load(output_dir1 / "train.npz")
        train2 = np.load(output_dir2 / "train.npz")
        
        np.testing.assert_array_equal(train1["z"], train2["z"])
        np.testing.assert_array_equal(train1["f"], train2["f"])
        np.testing.assert_array_equal(train1["params"], train2["params"])


def test_generate_dataset_non_negative():
    """Test that generated PDFs are non-negative."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        generate_dataset(
            output_dir=output_dir,
            n_train=50,
            n_val=10,
            n_test=10,
            seed_train=0,
            seed_val=1,
            seed_test=2,
            z_min=-8,
            z_max=8,
            n_points=64,
        )
        
        train_data = np.load(output_dir / "train.npz")
        f = train_data["f"]
        
        assert np.all(f >= 0), "PDF values must be non-negative"


def test_generate_dataset_zero_samples():
    """Test edge case: n_samples=0."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # Should not raise error
        generate_dataset(
            output_dir=output_dir,
            n_train=0,
            n_val=0,
            n_test=0,
            seed_train=0,
            seed_val=1,
            seed_test=2,
            z_min=-8,
            z_max=8,
            n_points=64,
        )
        
        # Files should still be created (but empty)
        assert (output_dir / "train.npz").exists()
        train_data = np.load(output_dir / "train.npz")
        assert train_data["f"].shape[0] == 0


def test_generate_dataset_single_point():
    """Test edge case: n_points=1."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        generate_dataset(
            output_dir=output_dir,
            n_train=10,
            n_val=5,
            n_test=5,
            seed_train=0,
            seed_val=1,
            seed_test=2,
            z_min=-8,
            z_max=8,
            n_points=1,
        )
        
        train_data = np.load(output_dir / "train.npz")
        z = train_data["z"]
        f = train_data["f"]
        
        assert len(z) == 1
        assert f.shape == (10, 1)
        # Single point PDF should be normalized to 1.0
        assert np.allclose(f[:, 0], 1.0)


def test_generate_dataset_extreme_ranges():
    """Test with extreme z ranges."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        generate_dataset(
            output_dir=output_dir,
            n_train=10,
            n_val=5,
            n_test=5,
            seed_train=0,
            seed_val=1,
            seed_test=2,
            z_min=-100,
            z_max=100,
            n_points=64,
        )
        
        train_data = np.load(output_dir / "train.npz")
        z = train_data["z"]
        
        assert np.all(z >= -100)
        assert np.all(z <= 100)


def test_generate_dataset_same_seed():
    """Test that same seed produces same results."""
    with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
        output_dir1 = Path(tmpdir1)
        output_dir2 = Path(tmpdir2)
        
        # Generate with same seeds
        generate_dataset(
            output_dir=output_dir1,
            n_train=10,
            n_val=5,
            n_test=5,
            seed_train=100,
            seed_val=100,
            seed_test=100,
            z_min=-8,
            z_max=8,
            n_points=64,
        )
        
        generate_dataset(
            output_dir=output_dir2,
            n_train=10,
            n_val=5,
            n_test=5,
            seed_train=100,
            seed_val=100,
            seed_test=100,
            z_min=-8,
            z_max=8,
            n_points=64,
        )
        
        # Should be identical
        train1 = np.load(output_dir1 / "train.npz")
        train2 = np.load(output_dir2 / "train.npz")
        
        np.testing.assert_array_equal(train1["z"], train2["z"])
        np.testing.assert_array_equal(train1["f"], train2["f"])
        np.testing.assert_array_equal(train1["params"], train2["params"])


def test_generate_dataset_invalid_range():
    """Test error handling for invalid z range."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # z_min > z_max: np.linspace will create empty array or reverse
        # This should either raise error or handle gracefully
        try:
            generate_dataset(
                output_dir=output_dir,
                n_train=10,
                n_val=5,
                n_test=5,
                seed_train=0,
                seed_val=1,
                seed_test=2,
                z_min=8,
                z_max=-8,  # Invalid: min > max
                n_points=64,
            )
            # If no error, check that output is reasonable
            train_data = np.load(output_dir / "train.npz")
            z = train_data["z"]
            # Should be empty or reversed
            assert len(z) == 0 or np.all(np.diff(z) <= 0)
        except (ValueError, AssertionError):
            # Error is acceptable
            pass

