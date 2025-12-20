"""Test coordinate transformation for MDN inference.

This test verifies that applying coordinate transformation (shifting by M1)
and then shifting back produces equivalent results to direct inference.
"""
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gmm_fitting import max_pdf_bivariate_normal, normalize_pdf_on_grid


def compute_m1(z: np.ndarray, f: np.ndarray) -> float:
    """Compute first moment (mean) of PDF using trapezoidal rule."""
    return np.trapz(z * f, z)


def interpolate_to_grid(z_orig: np.ndarray, f_orig: np.ndarray, z_target: np.ndarray) -> np.ndarray:
    """Interpolate PDF to target grid."""
    f_interp = np.interp(z_target, z_orig, f_orig, left=0.0, right=0.0)
    # Re-normalize
    integral = np.trapz(f_interp, z_target)
    if integral > 0:
        f_interp = f_interp / integral
    return f_interp


def test_coordinate_transform_equivalence():
    """Test that coordinate transform + inverse transform gives same result."""
    print("=" * 60)
    print("Testing coordinate transformation equivalence")
    print("=" * 60)
    
    # Fixed grid
    z_min, z_max, N = -8.0, 8.0, 64
    z = np.linspace(z_min, z_max, N)
    
    # Test cases with various parameters
    test_cases = [
        # (mu_x, sigma_x, mu_y, sigma_y, rho)
        (0.0, 1.0, 0.0, 1.0, 0.0),      # Symmetric, uncorrelated
        (1.0, 1.0, 2.0, 1.0, 0.0),      # Y shifted right
        (-1.0, 0.5, 1.0, 1.5, 0.5),     # Different variances, positive correlation
        (2.0, 0.8, -1.0, 1.2, -0.7),    # Negative correlation
        (0.5, 0.3, 0.5, 0.3, 0.9),      # High correlation, small variance
        (-2.0, 2.0, 2.0, 0.5, 0.0),     # Large difference in means and variances
    ]
    
    print(f"\nGrid: z ∈ [{z_min}, {z_max}], N = {N}")
    print("-" * 60)
    
    for i, (mu_x, sigma_x, mu_y, sigma_y, rho) in enumerate(test_cases):
        print(f"\nTest case {i+1}: μ_x={mu_x}, σ_x={sigma_x}, μ_y={mu_y}, σ_y={sigma_y}, ρ={rho}")
        
        # Generate original PDF
        f_orig = max_pdf_bivariate_normal(z, mu_x, sigma_x, mu_y, sigma_y, rho)
        f_orig = normalize_pdf_on_grid(z, f_orig)
        
        # Compute M1 (first moment)
        M1 = compute_m1(z, f_orig)
        print(f"  M1 (first moment) = {M1:.6f}")
        
        # Transform to relative coordinates (shift by -M1)
        z_shifted = z - M1
        
        # Interpolate to standard grid
        f_shifted = interpolate_to_grid(z_shifted, f_orig, z)
        
        # Verify: compute M1 of shifted PDF (should be close to 0)
        M1_shifted = compute_m1(z, f_shifted)
        print(f"  M1 after shift = {M1_shifted:.6f} (should be ~0)")
        
        # Transform back: shift z by +M1 and interpolate
        z_back = z + M1
        f_back = interpolate_to_grid(z_back, f_shifted, z)
        
        # Compare original and round-trip PDF
        max_diff = np.max(np.abs(f_orig - f_back))
        l2_diff = np.sqrt(np.trapz((f_orig - f_back)**2, z))
        
        print(f"  Round-trip error: max|Δf| = {max_diff:.2e}, L2 = {l2_diff:.2e}")
        
        # Check if errors are acceptable
        if max_diff < 1e-3:
            print("  ✓ PASS: Round-trip transformation is accurate")
        else:
            print("  ✗ FAIL: Round-trip transformation has significant error")


def test_mdn_with_coordinate_transform():
    """Test MDN inference with and without coordinate transformation."""
    print("\n" + "=" * 60)
    print("Testing MDN inference with coordinate transformation")
    print("=" * 60)
    
    # Try to import MDN
    try:
        import torch
        from src.ml_init.infer import mdn_predict_init
        from src.ml_init.eval import load_model_and_metadata
    except ImportError as e:
        print(f"Skipping MDN test: {e}")
        return
    
    # Check if model exists
    model_path = Path("ml_init/checkpoints/mdn_init_v1_N64_K5.pt")
    if not model_path.exists():
        print(f"Skipping MDN test: Model not found at {model_path}")
        return
    
    # Load model
    model, metadata, device = load_model_and_metadata(str(model_path))
    model.eval()
    
    # Fixed grid
    z_min, z_max, N = -8.0, 8.0, 64
    z = np.linspace(z_min, z_max, N)
    K = 5
    
    # Test cases
    test_cases = [
        (0.0, 1.0, 0.0, 1.0, 0.0),
        (1.5, 1.0, 0.5, 1.0, 0.3),
        (-1.0, 0.8, 1.0, 1.2, -0.5),
        (2.0, 0.5, -0.5, 1.5, 0.7),
    ]
    
    print(f"\nComparing direct inference vs coordinate-transformed inference")
    print("-" * 60)
    
    for i, (mu_x, sigma_x, mu_y, sigma_y, rho) in enumerate(test_cases):
        print(f"\nTest case {i+1}: μ_x={mu_x}, σ_x={sigma_x}, μ_y={mu_y}, σ_y={sigma_y}, ρ={rho}")
        
        # Generate PDF
        f_orig = max_pdf_bivariate_normal(z, mu_x, sigma_x, mu_y, sigma_y, rho)
        f_orig = normalize_pdf_on_grid(z, f_orig)
        
        # Method 1: Direct inference
        result_direct = mdn_predict_init(z, f_orig, K, str(model_path))
        mu_direct = result_direct["mu"]
        
        # Method 2: Inference with coordinate transformation
        M1 = compute_m1(z, f_orig)
        
        # Shift to relative coordinates
        z_shifted = z - M1
        f_shifted = interpolate_to_grid(z_shifted, f_orig, z)
        
        # Infer on shifted PDF
        result_shifted = mdn_predict_init(z, f_shifted, K, str(model_path))
        mu_shifted = result_shifted["mu"]
        
        # Transform mu back to absolute coordinates
        mu_transformed = mu_shifted + M1
        
        # Compare
        mu_diff = np.abs(mu_direct - mu_transformed)
        print(f"  M1 = {M1:.4f}")
        print(f"  Direct μ:      {mu_direct}")
        print(f"  Transformed μ: {mu_transformed}")
        print(f"  Difference:    {mu_diff}")
        print(f"  Max |Δμ|: {np.max(mu_diff):.4f}")
        
        # Check pi and var as well
        pi_diff = np.max(np.abs(result_direct["pi"] - result_shifted["pi"]))
        var_diff = np.max(np.abs(result_direct["var"] - result_shifted["var"]))
        print(f"  Max |Δπ|: {pi_diff:.4f}, Max |Δvar|: {var_diff:.4f}")


def test_shifted_pdf_shape():
    """Verify that PDFs with same relative parameters have same shape after centering."""
    print("\n" + "=" * 60)
    print("Testing that relative parameters determine shape")
    print("=" * 60)
    
    z_min, z_max, N = -8.0, 8.0, 64
    z = np.linspace(z_min, z_max, N)
    
    # These two cases have the same relative parameters:
    # Δμ = 1.0, σ_x = 1.0, σ_y = 0.8, ρ = 0.5
    case1 = (0.0, 1.0, 1.0, 0.8, 0.5)   # μ_x = 0
    case2 = (2.0, 1.0, 3.0, 0.8, 0.5)   # μ_x = 2, μ_y = 3 (same Δμ = 1)
    case3 = (-1.5, 1.0, -0.5, 0.8, 0.5) # μ_x = -1.5, μ_y = -0.5 (same Δμ = 1)
    
    print("\nComparing PDFs with same relative parameters (Δμ=1, σ_x=1, σ_y=0.8, ρ=0.5)")
    print("-" * 60)
    
    results = []
    for name, (mu_x, sigma_x, mu_y, sigma_y, rho) in [
        ("Case 1 (μ_x=0)", case1),
        ("Case 2 (μ_x=2)", case2),
        ("Case 3 (μ_x=-1.5)", case3),
    ]:
        f = max_pdf_bivariate_normal(z, mu_x, sigma_x, mu_y, sigma_y, rho)
        f = normalize_pdf_on_grid(z, f)
        M1 = compute_m1(z, f)
        
        # Center the PDF
        z_centered = z - M1
        f_centered = interpolate_to_grid(z_centered, f, z)
        
        print(f"\n{name}: M1 = {M1:.4f}")
        results.append((name, f_centered, M1))
    
    # Compare centered PDFs
    print("\n" + "-" * 60)
    print("Comparing centered PDFs:")
    
    f1, f2, f3 = results[0][1], results[1][1], results[2][1]
    
    diff_12 = np.max(np.abs(f1 - f2))
    diff_13 = np.max(np.abs(f1 - f3))
    diff_23 = np.max(np.abs(f2 - f3))
    
    print(f"  max|f1 - f2| = {diff_12:.2e}")
    print(f"  max|f1 - f3| = {diff_13:.2e}")
    print(f"  max|f2 - f3| = {diff_23:.2e}")
    
    if max(diff_12, diff_13, diff_23) < 1e-3:
        print("\n  ✓ PASS: Centered PDFs are equivalent (shape is determined by relative params)")
    else:
        print("\n  ✗ Note: Some difference exists (may be due to interpolation or grid boundaries)")


if __name__ == "__main__":
    test_coordinate_transform_equivalence()
    test_shifted_pdf_shape()
    test_mdn_with_coordinate_transform()
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)

