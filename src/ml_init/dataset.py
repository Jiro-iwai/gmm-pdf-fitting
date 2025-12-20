"""Dataset generation for MDN training."""
import numpy as np
from pathlib import Path
from typing import Optional
import json

from src.gmm_fitting import max_pdf_bivariate_normal, normalize_pdf_on_grid


# Parameter ranges (hardcoded for reproducibility)
# V5: Extended parameter ranges with systematic sampling
COORDINATE_MODE = "relative"  # "absolute" or "relative"

# For absolute mode (legacy)
MU_RANGE = (-3.0, 3.0)

# For relative mode (V5)
MU_X_FIXED = 0.0  # μ_x is fixed at 0
DELTA_MU_RANGE = (-10.0, 10.0)  # Δμ = μ_y - μ_x range (extended from ±6)

# Common parameters (V5)
SIGMA_RANGE = (0.1, 5.0)  # Extended: [0.1, 5.0] for broader coverage
RHO_RANGE = (-0.999, 0.999)  # Extended to ±0.999

# Grid parameters (V5)
DEFAULT_Z_MIN = -15.0  # Extended from -10.0
DEFAULT_Z_MAX = 15.0   # Extended from 10.0
DEFAULT_N_POINTS = 96  # Increased from 64

# Sampling strategy options
SAMPLING_STRATEGY = "stratified"  # "uniform", "improved", or "stratified"


def generate_dataset(
    output_dir: Path,
    n_train: int = 150000,
    n_val: int = 15000,
    n_test: int = 15000,
    seed_train: int = 0,
    seed_val: int = 1,
    seed_test: int = 2,
    z_min: float = DEFAULT_Z_MIN,
    z_max: float = DEFAULT_Z_MAX,
    n_points: int = DEFAULT_N_POINTS,
) -> None:
    """
    Generate training dataset for MDN.
    
    Parameters:
    -----------
    output_dir : Path
        Output directory for .npz files
    n_train : int
        Number of training samples
    n_val : int
        Number of validation samples
    n_test : int
        Number of test samples
    seed_train : int
        Random seed for training set
    seed_val : int
        Random seed for validation set
    seed_test : int
        Random seed for test set
    z_min : float
        Minimum z value for grid
    z_max : float
        Maximum z value for grid
    n_points : int
        Number of grid points
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Fixed grid
    z = np.linspace(z_min, z_max, n_points)
    
    # Generate each split
    splits = [
        ("train", n_train, seed_train),
        ("val", n_val, seed_val),
        ("test", n_test, seed_test),
    ]
    
    for split_name, n_samples, seed in splits:
        z_array, f_array, params_array = _generate_split(
            z, n_samples, seed
        )
        
        # Save as .npz
        output_path = output_dir / f"{split_name}.npz"
        np.savez(
            output_path,
            z=z_array,
            f=f_array,
            params=params_array,
        )
        
        print(f"Generated {split_name}: {n_samples} samples -> {output_path}")


def _sample_sigma_log_uniform(sigma_min: float, sigma_max: float, rng: np.random.Generator) -> float:
    """Sample sigma from log-uniform distribution (more small values)."""
    log_min = np.log(sigma_min)
    log_max = np.log(sigma_max)
    return float(np.exp(rng.uniform(log_min, log_max)))


def _sample_mu(rng: np.random.Generator) -> tuple[float, float]:
    """Sample μ_x and μ_y based on coordinate mode."""
    if COORDINATE_MODE == "relative":
        mu_x = MU_X_FIXED
        delta_mu = rng.uniform(*DELTA_MU_RANGE)
        mu_y = mu_x + delta_mu
    else:
        mu_x = rng.uniform(*MU_RANGE)
        mu_y = rng.uniform(*MU_RANGE)
    return mu_x, mu_y


def _sample_params_improved(rng: np.random.Generator) -> tuple[float, float, float, float, float]:
    """
    Improved parameter sampling strategy.
    
    - 50% standard cases: symmetric parameters
    - 30% asymmetric cases: different X/Y parameters
    - 20% edge cases: extreme correlations or variance ratios
    
    In relative coordinate mode, μ_x is always 0.
    """
    case_type = rng.random()
    
    if case_type < 0.5:
        # Standard symmetric case
        if COORDINATE_MODE == "relative":
            mu_x = MU_X_FIXED
            mu_y = MU_X_FIXED  # Same as mu_x for symmetric case
        else:
            mu = rng.uniform(*MU_RANGE)
            mu_x = mu
            mu_y = mu
        sigma = _sample_sigma_log_uniform(SIGMA_RANGE[0], SIGMA_RANGE[1], rng)
        rho = rng.uniform(*RHO_RANGE)
        return mu_x, sigma, mu_y, sigma, rho
    
    elif case_type < 0.8:
        # Asymmetric case: different X/Y parameters
        mu_x, mu_y = _sample_mu(rng)
        sigma_x = _sample_sigma_log_uniform(SIGMA_RANGE[0], SIGMA_RANGE[1], rng)
        sigma_y = _sample_sigma_log_uniform(SIGMA_RANGE[0], SIGMA_RANGE[1], rng)
        rho = rng.uniform(*RHO_RANGE)
        return mu_x, sigma_x, mu_y, sigma_y, rho
    
    else:
        # Edge case: extreme parameters
        edge_type = rng.integers(0, 3)
        
        if edge_type == 0:
            # Extreme correlation
            mu_x, mu_y = _sample_mu(rng)
            sigma_x = _sample_sigma_log_uniform(SIGMA_RANGE[0], SIGMA_RANGE[1], rng)
            sigma_y = _sample_sigma_log_uniform(SIGMA_RANGE[0], SIGMA_RANGE[1], rng)
            # High correlation (positive or negative)
            if rng.random() < 0.5:
                rho = rng.uniform(0.85, 0.99)
            else:
                rho = rng.uniform(-0.99, -0.85)
            return mu_x, sigma_x, mu_y, sigma_y, rho
        
        elif edge_type == 1:
            # Extreme variance ratio
            mu_x, mu_y = _sample_mu(rng)
            # One sigma very small, one larger
            if rng.random() < 0.5:
                sigma_x = _sample_sigma_log_uniform(SIGMA_RANGE[0], 0.3, rng)
                sigma_y = _sample_sigma_log_uniform(0.8, SIGMA_RANGE[1], rng)
            else:
                sigma_x = _sample_sigma_log_uniform(0.8, SIGMA_RANGE[1], rng)
                sigma_y = _sample_sigma_log_uniform(SIGMA_RANGE[0], 0.3, rng)
            rho = rng.uniform(*RHO_RANGE)
            return mu_x, sigma_x, mu_y, sigma_y, rho
        
        else:
            # Very small variance (both)
            if COORDINATE_MODE == "relative":
                mu_x = MU_X_FIXED
                # Narrower Δμ range for small variance
                delta_mu = rng.uniform(-2.0, 2.0)
                mu_y = mu_x + delta_mu
            else:
                mu_x = rng.uniform(-1.0, 1.0)
                mu_y = rng.uniform(-1.0, 1.0)
            sigma_x = _sample_sigma_log_uniform(SIGMA_RANGE[0], 0.25, rng)
            sigma_y = _sample_sigma_log_uniform(SIGMA_RANGE[0], 0.25, rng)
            rho = rng.uniform(*RHO_RANGE)
            return mu_x, sigma_x, mu_y, sigma_y, rho


# Stratified sampling bins (V5)
DELTA_MU_BINS = [(-10, -6), (-6, -3), (-3, -1), (-1, 1), (1, 3), (3, 6), (6, 10)]  # 7 bins
SIGMA_BINS = [(0.1, 0.3), (0.3, 0.7), (0.7, 1.5), (1.5, 3.0), (3.0, 5.0)]  # 5 bins (log-like)
RHO_BINS = [(-0.999, -0.9), (-0.9, -0.7), (-0.7, -0.3), (-0.3, 0.3), (0.3, 0.7), (0.7, 0.9), (0.9, 0.999)]  # 7 bins


def _sample_params_stratified(rng: np.random.Generator, sample_idx: int, total_samples: int) -> tuple[float, float, float, float, float]:
    """
    Stratified sampling strategy (V5).
    
    Ensures coverage of the entire parameter space by:
    1. 60% basic samples: uniform random (smooth coverage)
    2. 25% stratified samples: sample from each cell of parameter grid
    3. 15% edge samples: explicitly sample extreme combinations
    
    Parameters:
    -----------
    rng : np.random.Generator
        Random number generator
    sample_idx : int
        Current sample index
    total_samples : int
        Total number of samples in this split
    
    Returns:
    --------
    mu_x, sigma_x, mu_y, sigma_y, rho : tuple of floats
    """
    # Determine sample type based on index ratio
    ratio = sample_idx / total_samples
    
    if ratio < 0.60:
        # Basic samples: uniform random
        return _sample_params_uniform(rng)
    
    elif ratio < 0.85:
        # Stratified samples: sample from grid cells
        return _sample_params_from_grid(rng)
    
    else:
        # Edge samples: extreme combinations
        return _sample_params_edge_cases(rng)


def _sample_params_uniform(rng: np.random.Generator) -> tuple[float, float, float, float, float]:
    """Uniform random sampling."""
    mu_x = MU_X_FIXED
    delta_mu = rng.uniform(*DELTA_MU_RANGE)
    mu_y = mu_x + delta_mu
    sigma_x = _sample_sigma_log_uniform(SIGMA_RANGE[0], SIGMA_RANGE[1], rng)
    sigma_y = _sample_sigma_log_uniform(SIGMA_RANGE[0], SIGMA_RANGE[1], rng)
    rho = rng.uniform(*RHO_RANGE)
    return mu_x, sigma_x, mu_y, sigma_y, rho


def _sample_params_from_grid(rng: np.random.Generator) -> tuple[float, float, float, float, float]:
    """Sample from a random cell in the parameter grid."""
    # Randomly select bins
    dm_bin = DELTA_MU_BINS[rng.integers(len(DELTA_MU_BINS))]
    sx_bin = SIGMA_BINS[rng.integers(len(SIGMA_BINS))]
    sy_bin = SIGMA_BINS[rng.integers(len(SIGMA_BINS))]
    rho_bin = RHO_BINS[rng.integers(len(RHO_BINS))]
    
    # Sample uniformly within each bin
    mu_x = MU_X_FIXED
    delta_mu = rng.uniform(*dm_bin)
    mu_y = mu_x + delta_mu
    sigma_x = rng.uniform(*sx_bin)
    sigma_y = rng.uniform(*sy_bin)
    rho = rng.uniform(*rho_bin)
    
    return mu_x, sigma_x, mu_y, sigma_y, rho


def _sample_params_edge_cases(rng: np.random.Generator) -> tuple[float, float, float, float, float]:
    """
    Sample extreme parameter combinations that are important but rare.
    
    Edge cases:
    1. High σ ratio + high |ρ|
    2. Low σ ratio + high |ρ|
    3. Large |Δμ| + small σ
    4. Very high |ρ| (any σ)
    5. Both σ at extremes
    """
    edge_type = rng.integers(0, 5)
    mu_x = MU_X_FIXED
    
    if edge_type == 0:
        # High σ ratio (≥3) + high |ρ| (≥0.8)
        sigma_x = rng.uniform(0.1, 1.0)
        sigma_y = rng.uniform(sigma_x * 3, min(sigma_x * 10, 5.0))
        if rng.random() < 0.5:
            # Swap
            sigma_x, sigma_y = sigma_y, sigma_x
        rho = rng.choice([-1, 1]) * rng.uniform(0.8, 0.999)
        delta_mu = rng.uniform(*DELTA_MU_RANGE)
        
    elif edge_type == 1:
        # Very high |ρ| (≥0.95) with any σ
        sigma_x = _sample_sigma_log_uniform(0.1, 5.0, rng)
        sigma_y = _sample_sigma_log_uniform(0.1, 5.0, rng)
        rho = rng.choice([-1, 1]) * rng.uniform(0.95, 0.999)
        delta_mu = rng.uniform(*DELTA_MU_RANGE)
        
    elif edge_type == 2:
        # Large |Δμ| (≥6) + small σ (≤1)
        delta_mu = rng.choice([-1, 1]) * rng.uniform(6, 10)
        sigma_x = rng.uniform(0.1, 1.0)
        sigma_y = rng.uniform(0.1, 1.0)
        rho = rng.uniform(*RHO_RANGE)
        
    elif edge_type == 3:
        # Both σ at extremes (both small or both large)
        if rng.random() < 0.5:
            # Both small
            sigma_x = rng.uniform(0.1, 0.3)
            sigma_y = rng.uniform(0.1, 0.3)
        else:
            # Both large
            sigma_x = rng.uniform(3.0, 5.0)
            sigma_y = rng.uniform(3.0, 5.0)
        rho = rng.uniform(*RHO_RANGE)
        delta_mu = rng.uniform(*DELTA_MU_RANGE)
        
    else:
        # Mixed extreme: high ratio + large Δμ + high ρ
        sigma_x = rng.uniform(0.1, 0.5)
        sigma_y = rng.uniform(2.0, 5.0)
        if rng.random() < 0.5:
            sigma_x, sigma_y = sigma_y, sigma_x
        rho = rng.choice([-1, 1]) * rng.uniform(0.7, 0.999)
        delta_mu = rng.choice([-1, 1]) * rng.uniform(4, 10)
    
    mu_y = mu_x + delta_mu
    return mu_x, sigma_x, mu_y, sigma_y, rho


def _generate_split(
    z: np.ndarray,
    n_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate one split of the dataset.
    
    Parameters:
    -----------
    z : np.ndarray
        Fixed grid points, shape (N,)
    n_samples : int
        Number of samples to generate
    seed : int
        Random seed
    
    Returns:
    --------
    z_array : np.ndarray
        Grid points (same for all samples), shape (N,)
    f_array : np.ndarray
        PDF values, shape (n_samples, N)
    params_array : np.ndarray
        Parameters (mu_x, sigma_x, mu_y, sigma_y, rho), shape (n_samples, 5)
    """
    rng = np.random.default_rng(seed)
    
    N = len(z)
    f_array = np.zeros((n_samples, N))
    params_array = np.zeros((n_samples, 5))
    
    for i in range(n_samples):
        if SAMPLING_STRATEGY == "stratified":
            mu_x, sigma_x, mu_y, sigma_y, rho = _sample_params_stratified(rng, i, n_samples)
        elif SAMPLING_STRATEGY == "improved":
            mu_x, sigma_x, mu_y, sigma_y, rho = _sample_params_improved(rng)
        else:
            # Uniform sampling
            if COORDINATE_MODE == "relative":
                # Relative mode: μ_x = 0, sample Δμ
                mu_x = MU_X_FIXED
                delta_mu = rng.uniform(*DELTA_MU_RANGE)
                mu_y = mu_x + delta_mu
            else:
                # Absolute mode (legacy)
                mu_x = rng.uniform(*MU_RANGE)
                mu_y = rng.uniform(*MU_RANGE)
            sigma_x = rng.uniform(*SIGMA_RANGE)
            sigma_y = rng.uniform(*SIGMA_RANGE)
            rho = rng.uniform(*RHO_RANGE)
        
        # Clamp rho to avoid numerical issues (V5: extended to ±0.999)
        rho = np.clip(rho, -0.999, 0.999)
        
        # Generate PDF
        f = max_pdf_bivariate_normal(z, mu_x, sigma_x, mu_y, sigma_y, rho)
        
        # Ensure non-negative
        f = np.maximum(f, 0.0)
        
        # Normalize (handle single point case)
        if len(z) == 1:
            # Single point: PDF is just 1.0 (normalized)
            f = np.array([1.0])
        else:
            f = normalize_pdf_on_grid(z, f)
        
        f_array[i] = f
        params_array[i] = [mu_x, sigma_x, mu_y, sigma_y, rho]
    
    return z, f_array, params_array


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate MDN training dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for dataset files")
    parser.add_argument("--n_train", type=int, default=150000, help="Number of training samples")
    parser.add_argument("--n_val", type=int, default=15000, help="Number of validation samples")
    parser.add_argument("--n_test", type=int, default=15000, help="Number of test samples")
    parser.add_argument("--seed_train", type=int, default=0, help="Random seed for training set")
    parser.add_argument("--seed_val", type=int, default=1, help="Random seed for validation set")
    parser.add_argument("--seed_test", type=int, default=2, help="Random seed for test set")
    parser.add_argument("--z_min", type=float, default=DEFAULT_Z_MIN, help="Minimum z value")
    parser.add_argument("--z_max", type=float, default=DEFAULT_Z_MAX, help="Maximum z value")
    parser.add_argument("--n_points", type=int, default=64, help="Number of grid points")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generate_dataset(
        output_dir=output_dir,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        seed_train=args.seed_train,
        seed_val=args.seed_val,
        seed_test=args.seed_test,
        z_min=args.z_min,
        z_max=args.z_max,
        n_points=args.n_points,
    )

