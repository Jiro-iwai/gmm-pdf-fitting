"""Benchmark script comparing MDN initialization with traditional methods."""
import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gmm_fitting import (
    max_pdf_bivariate_normal,
    normalize_pdf_on_grid,
    fit_gmm1d_to_pdf_weighted_em,
    compute_pdf_statistics,
    gmm1d_pdf,
    GMM1DParams,
)


def run_single_test(
    mu_x: float,
    sigma_x: float,
    mu_y: float,
    sigma_y: float,
    rho: float,
    K: int,
    z_range: Tuple[float, float],
    z_npoints: int,
    init_method: str,
    mdn_model_path: str = None,
    max_iter: int = 100,
    tol: float = 1e-6,
    reg_var: float = 1e-6,
    n_init: int = 8,
    seed: int = 0,
) -> Dict:
    """Run a single test case."""
    z = np.linspace(z_range[0], z_range[1], z_npoints)
    f_true = max_pdf_bivariate_normal(z, mu_x, sigma_x**2, mu_y, sigma_y**2, rho)
    f_true = normalize_pdf_on_grid(z, f_true)
    
    # Adjust n_init based on init method
    if init_method == "mdn":
        n_init_test = 1
    else:
        n_init_test = n_init
    
    # Run fitting
    start_time = time.time()
    try:
        params, ll, n_iter = fit_gmm1d_to_pdf_weighted_em(
            z, f_true,
            K=K,
            max_iter=max_iter,
            tol=tol,
            reg_var=reg_var,
            n_init=n_init_test,
            seed=seed,
            init=init_method,
            mdn_model_path=mdn_model_path,
            mdn_device="cpu",
        )
        elapsed_time = time.time() - start_time
        
        # Compute statistics
        stats_true = compute_pdf_statistics(z, f_true)
        f_approx = gmm1d_pdf(z, params)
        stats_approx = compute_pdf_statistics(z, f_approx)
        
        # Compute errors
        pdf_linf = np.max(np.abs(f_true - f_approx))
        dz = z[1] - z[0] if len(z) > 1 else 1.0
        cdf_true = np.cumsum(f_true * dz)
        cdf_approx = np.cumsum(f_approx * dz)
        cdf_linf = np.max(np.abs(cdf_true - cdf_approx))
        
        # Quantile errors
        quantiles = [0.5, 0.9, 0.99]
        quantile_errors = {}
        for p in quantiles:
            q_true = np.interp(p, cdf_true, z)
            q_approx = np.interp(p, cdf_approx, z)
            quantile_errors[f"p_{p}"] = abs(q_true - q_approx)
        
        return {
            "success": True,
            "elapsed_time": elapsed_time,
            "n_iter": n_iter,
            "log_likelihood": float(ll),
            "pdf_linf_error": float(pdf_linf),
            "cdf_linf_error": float(cdf_linf),
            "quantile_errors": quantile_errors,
            "mean_error": abs(stats_true["mean"] - stats_approx["mean"]),
            "std_error": abs(stats_true["std"] - stats_approx["std"]),
            "skewness_error": abs(stats_true["skewness"] - stats_approx["skewness"]) if stats_true["skewness"] is not None else None,
            "kurtosis_error": abs(stats_true["kurtosis"] - stats_approx["kurtosis"]) if stats_true["kurtosis"] is not None else None,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "elapsed_time": time.time() - start_time,
        }


def run_benchmark(
    test_cases: List[Dict],
    mdn_model_path: str = None,
    output_path: Path = None,
) -> Dict:
    """Run benchmark on multiple test cases."""
    results = {
        "mdn": [],
        "quantile": [],
        "test_cases": test_cases,
    }
    
    print(f"Running benchmark with {len(test_cases)} test cases...")
    print("=" * 80)
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest case {i+1}/{len(test_cases)}:")
        print(f"  mu_x={test_case['mu_x']}, sigma_x={test_case['sigma_x']}, "
              f"mu_y={test_case['mu_y']}, sigma_y={test_case['sigma_y']}, "
              f"rho={test_case['rho']}, K={test_case['K']}")
        
        # Test MDN initialization
        print("  Testing MDN initialization...")
        mdn_result = run_single_test(
            init_method="mdn",
            mdn_model_path=mdn_model_path,
            **test_case
        )
        results["mdn"].append(mdn_result)
        
        if mdn_result["success"]:
            print(f"    ✓ Success: {mdn_result['n_iter']} iterations, "
                  f"{mdn_result['elapsed_time']:.4f}s, "
                  f"PDF L∞={mdn_result['pdf_linf_error']:.6f}")
        else:
            print(f"    ✗ Failed: {mdn_result.get('error', 'Unknown error')}")
        
        # Test Quantile initialization
        print("  Testing Quantile initialization...")
        quantile_result = run_single_test(
            init_method="quantile",
            **test_case
        )
        results["quantile"].append(quantile_result)
        
        if quantile_result["success"]:
            print(f"    ✓ Success: {quantile_result['n_iter']} iterations, "
                  f"{quantile_result['elapsed_time']:.4f}s, "
                  f"PDF L∞={quantile_result['pdf_linf_error']:.6f}")
        else:
            print(f"    ✗ Failed: {quantile_result.get('error', 'Unknown error')}")
    
    # Save results
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    return results


def generate_test_cases() -> List[Dict]:
    """Generate diverse test cases."""
    test_cases = []
    
    # Base parameters
    base_params = {
        "z_range": (-4.0, 4.0),
        "z_npoints": 64,
        "max_iter": 100,
        "tol": 1e-6,
        "reg_var": 1e-6,
        "n_init": 8,
        "seed": 0,
    }
    
    # Test case 1: Standard case
    test_cases.append({
        **base_params,
        "mu_x": 0.1,
        "sigma_x": 0.4,
        "mu_y": 0.15,
        "sigma_y": 0.9,
        "rho": 0.9,
        "K": 5,
    })
    
    # Test case 2: Low correlation
    test_cases.append({
        **base_params,
        "mu_x": 0.0,
        "sigma_x": 0.8,
        "mu_y": 0.0,
        "sigma_y": 1.6,
        "rho": 0.3,
        "K": 5,
    })
    
    # Test case 3: High correlation
    test_cases.append({
        **base_params,
        "mu_x": 0.0,
        "sigma_x": 0.8,
        "mu_y": 0.0,
        "sigma_y": 1.6,
        "rho": 0.99,
        "K": 5,
    })
    
    # Test case 4: Negative correlation
    test_cases.append({
        **base_params,
        "mu_x": 0.0,
        "sigma_x": 1.0,
        "mu_y": 0.0,
        "sigma_y": 1.0,
        "rho": -0.7,
        "K": 5,
    })
    
    # Test case 5: Different means
    test_cases.append({
        **base_params,
        "mu_x": 1.0,
        "sigma_x": 0.5,
        "mu_y": -1.0,
        "sigma_y": 0.5,
        "rho": 0.5,
        "K": 5,
    })
    
    # Test case 6: Small K
    test_cases.append({
        **base_params,
        "mu_x": 0.0,
        "sigma_x": 0.8,
        "mu_y": 0.0,
        "sigma_y": 1.6,
        "rho": 0.9,
        "K": 3,
    })
    
    # Test case 7: Large K
    test_cases.append({
        **base_params,
        "mu_x": 0.0,
        "sigma_x": 0.8,
        "mu_y": 0.0,
        "sigma_y": 1.6,
        "rho": 0.9,
        "K": 7,
    })
    
    # Test case 8: Small variance
    test_cases.append({
        **base_params,
        "mu_x": 0.0,
        "sigma_x": 0.3,
        "mu_y": 0.0,
        "sigma_y": 0.3,
        "rho": 0.9,
        "K": 5,
    })
    
    # Test case 9: Large variance
    test_cases.append({
        **base_params,
        "mu_x": 0.0,
        "sigma_x": 1.5,
        "mu_y": 0.0,
        "sigma_y": 2.0,
        "rho": 0.7,
        "K": 5,
    })
    
    # Test case 10: Asymmetric
    test_cases.append({
        **base_params,
        "mu_x": 0.5,
        "sigma_x": 0.6,
        "mu_y": -0.3,
        "sigma_y": 1.2,
        "rho": 0.8,
        "K": 5,
    })
    
    return test_cases


def analyze_results(results: Dict) -> Dict:
    """Analyze benchmark results."""
    mdn_results = [r for r in results["mdn"] if r.get("success", False)]
    quantile_results = [r for r in results["quantile"] if r.get("success", False)]
    
    if len(mdn_results) == 0 or len(quantile_results) == 0:
        return {
            "error": "No successful results",
            "mdn_success": len(mdn_results),
            "quantile_success": len(quantile_results),
        }
    
    analysis = {
        "n_tests": len(mdn_results),
        "mdn": {
            "mean_time": np.mean([r["elapsed_time"] for r in mdn_results]),
            "std_time": np.std([r["elapsed_time"] for r in mdn_results]),
            "mean_iterations": np.mean([r["n_iter"] for r in mdn_results]),
            "std_iterations": np.std([r["n_iter"] for r in mdn_results]),
            "mean_pdf_linf": np.mean([r["pdf_linf_error"] for r in mdn_results]),
            "std_pdf_linf": np.std([r["pdf_linf_error"] for r in mdn_results]),
            "mean_cdf_linf": np.mean([r["cdf_linf_error"] for r in mdn_results]),
            "std_cdf_linf": np.std([r["cdf_linf_error"] for r in mdn_results]),
            "mean_quantile_errors": {
                k: np.mean([r["quantile_errors"][k] for r in mdn_results])
                for k in mdn_results[0]["quantile_errors"].keys()
            },
        },
        "quantile": {
            "mean_time": np.mean([r["elapsed_time"] for r in quantile_results]),
            "std_time": np.std([r["elapsed_time"] for r in quantile_results]),
            "mean_iterations": np.mean([r["n_iter"] for r in quantile_results]),
            "std_iterations": np.std([r["n_iter"] for r in quantile_results]),
            "mean_pdf_linf": np.mean([r["pdf_linf_error"] for r in quantile_results]),
            "std_pdf_linf": np.std([r["pdf_linf_error"] for r in quantile_results]),
            "mean_cdf_linf": np.mean([r["cdf_linf_error"] for r in quantile_results]),
            "std_cdf_linf": np.std([r["cdf_linf_error"] for r in quantile_results]),
            "mean_quantile_errors": {
                k: np.mean([r["quantile_errors"][k] for r in quantile_results])
                for k in quantile_results[0]["quantile_errors"].keys()
            },
        },
    }
    
    # Compute improvements
    analysis["improvements"] = {
        "time_ratio": analysis["quantile"]["mean_time"] / analysis["mdn"]["mean_time"],
        "iteration_ratio": analysis["quantile"]["mean_iterations"] / analysis["mdn"]["mean_iterations"],
        "pdf_linf_ratio": analysis["quantile"]["mean_pdf_linf"] / analysis["mdn"]["mean_pdf_linf"],
        "cdf_linf_ratio": analysis["quantile"]["mean_cdf_linf"] / analysis["mdn"]["mean_cdf_linf"],
        "quantile_error_ratios": {
            k: analysis["quantile"]["mean_quantile_errors"][k] / analysis["mdn"]["mean_quantile_errors"][k]
            for k in analysis["mdn"]["mean_quantile_errors"].keys()
        },
    }
    
    return analysis


def print_report(analysis: Dict, output_path: Path = None):
    """Print benchmark report."""
    report = []
    report.append("=" * 80)
    report.append("MDN初期化 vs Quantile初期化 ベンチマーク結果")
    report.append("=" * 80)
    
    if "error" in analysis:
        report.append(f"\nエラー: {analysis['error']}")
        report.append(f"MDN成功: {analysis.get('mdn_success', 0)}")
        report.append(f"Quantile成功: {analysis.get('quantile_success', 0)}")
        report_text = "\n".join(report)
        print(report_text)
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
        return
    
    report.append(f"\nテストケース数: {analysis['n_tests']}")
    report.append("\n" + "-" * 80)
    report.append("実行時間")
    report.append("-" * 80)
    report.append(f"MDN初期化:")
    report.append(f"  平均: {analysis['mdn']['mean_time']:.4f}秒 ± {analysis['mdn']['std_time']:.4f}秒")
    report.append(f"Quantile初期化:")
    report.append(f"  平均: {analysis['quantile']['mean_time']:.4f}秒 ± {analysis['quantile']['std_time']:.4f}秒")
    report.append(f"時間比 (Quantile/MDN): {analysis['improvements']['time_ratio']:.2f}x")
    
    report.append("\n" + "-" * 80)
    report.append("反復回数")
    report.append("-" * 80)
    report.append(f"MDN初期化:")
    report.append(f"  平均: {analysis['mdn']['mean_iterations']:.1f}回 ± {analysis['mdn']['std_iterations']:.1f}回")
    report.append(f"Quantile初期化:")
    report.append(f"  平均: {analysis['quantile']['mean_iterations']:.1f}回 ± {analysis['quantile']['std_iterations']:.1f}回")
    report.append(f"反復回数比 (Quantile/MDN): {analysis['improvements']['iteration_ratio']:.2f}x")
    
    report.append("\n" + "-" * 80)
    report.append("PDF L∞誤差")
    report.append("-" * 80)
    report.append(f"MDN初期化:")
    report.append(f"  平均: {analysis['mdn']['mean_pdf_linf']:.6f} ± {analysis['mdn']['std_pdf_linf']:.6f}")
    report.append(f"Quantile初期化:")
    report.append(f"  平均: {analysis['quantile']['mean_pdf_linf']:.6f} ± {analysis['quantile']['std_pdf_linf']:.6f}")
    report.append(f"誤差比 (Quantile/MDN): {analysis['improvements']['pdf_linf_ratio']:.2f}x")
    
    report.append("\n" + "-" * 80)
    report.append("CDF L∞誤差")
    report.append("-" * 80)
    report.append(f"MDN初期化:")
    report.append(f"  平均: {analysis['mdn']['mean_cdf_linf']:.6f} ± {analysis['mdn']['std_cdf_linf']:.6f}")
    report.append(f"Quantile初期化:")
    report.append(f"  平均: {analysis['quantile']['mean_cdf_linf']:.6f} ± {analysis['quantile']['std_cdf_linf']:.6f}")
    report.append(f"誤差比 (Quantile/MDN): {analysis['improvements']['cdf_linf_ratio']:.2f}x")
    
    report.append("\n" + "-" * 80)
    report.append("分位点誤差")
    report.append("-" * 80)
    for k, v in analysis['mdn']['mean_quantile_errors'].items():
        p = k.replace('p_', '')
        report.append(f"\n分位点 {p}:")
        report.append(f"  MDN初期化: {v:.6f}")
        report.append(f"  Quantile初期化: {analysis['quantile']['mean_quantile_errors'][k]:.6f}")
        ratio = analysis['improvements']['quantile_error_ratios'][k]
        report.append(f"  誤差比 (Quantile/MDN): {ratio:.2f}x")
    
    report.append("\n" + "=" * 80)
    report.append("総合評価")
    report.append("=" * 80)
    
    if analysis['improvements']['iteration_ratio'] > 1.1:
        report.append(f"✓ MDN初期化は反復回数が{analysis['improvements']['iteration_ratio']:.1f}倍少ない（収束が速い）")
    
    if analysis['improvements']['pdf_linf_ratio'] < 0.9:
        report.append(f"✓ Quantile初期化はPDF L∞誤差が{1/analysis['improvements']['pdf_linf_ratio']:.1f}倍良い")
    elif analysis['improvements']['pdf_linf_ratio'] > 1.1:
        report.append(f"✓ MDN初期化はPDF L∞誤差が{analysis['improvements']['pdf_linf_ratio']:.1f}倍良い")
    
    if analysis['improvements']['cdf_linf_ratio'] < 0.9:
        report.append(f"✓ Quantile初期化はCDF L∞誤差が{1/analysis['improvements']['cdf_linf_ratio']:.1f}倍良い")
    elif analysis['improvements']['cdf_linf_ratio'] > 1.1:
        report.append(f"✓ MDN初期化はCDF L∞誤差が{analysis['improvements']['cdf_linf_ratio']:.1f}倍良い")
    
    for k, ratio in analysis['improvements']['quantile_error_ratios'].items():
        p = k.replace('p_', '')
        if ratio > 1.1:
            report.append(f"✓ MDN初期化は分位点{p}の誤差が{ratio:.1f}倍良い")
        elif ratio < 0.9:
            report.append(f"✓ Quantile初期化は分位点{p}の誤差が{1/ratio:.1f}倍良い")
    
    report_text = "\n".join(report)
    print(report_text)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
        print(f"\nレポートを保存しました: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark MDN vs Quantile initialization")
    parser.add_argument("--mdn_model_path", type=str, 
                       default="./ml_init/checkpoints/mdn_init_v1_N64_K5.pt",
                       help="Path to MDN model")
    parser.add_argument("--output_dir", type=str, default="./benchmarks/results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    mdn_model_path = args.mdn_model_path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate test cases
    test_cases = generate_test_cases()
    
    # Run benchmark
    results = run_benchmark(
        test_cases=test_cases,
        mdn_model_path=mdn_model_path,
        output_path=output_dir / "benchmark_mdn_results.json",
    )
    
    # Analyze results
    analysis = analyze_results(results)
    
    # Print and save report
    print_report(analysis, output_path=output_dir / "benchmark_mdn_report.txt")
    
    # Save analysis
    with open(output_dir / "benchmark_mdn_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2)

