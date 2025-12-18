#!/usr/bin/env python3
"""
実行例: LP法のモーメント誤差最小化モード

このスクリプトは、新しく実装したモーメント誤差最小化機能の使用例を示します。

実行方法:
    python examples/example_moments_mode.py
    または
    cd examples && python example_moments_mode.py
"""

import sys
from pathlib import Path

# Add src directory to path to import gmm_fitting package
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from gmm_fitting import fit_gmm_lp_simple, compute_pdf_statistics
from gmm_fitting.em_method import normal_pdf


def main():
    print("=" * 80)
    print("LP法のモーメント誤差最小化モード - 実行例")
    print("=" * 80)
    
    # グリッドの設定
    z = np.linspace(-5, 5, 200)
    
    # 真のPDF: 正規分布 N(0, 1)
    print("\n1. 真のPDF: 正規分布 N(0, 1)")
    f_true = normal_pdf(z, 0.0, 1.0)
    
    # 真の統計量を計算
    stats_true = compute_pdf_statistics(z, f_true)
    print(f"  平均: {stats_true['mean']:.6f}")
    print(f"  標準偏差: {stats_true['std']:.6f}")
    print(f"  分散: {stats_true['std']**2:.6f}")
    print(f"  歪度: {stats_true['skewness']:.6f}")
    print(f"  尖度: {stats_true['kurtosis']:.6f}")
    
    # ============================================================================
    # 例1: fit_gmm_lp_simple でモーメント誤差最小化モードを使用
    # ============================================================================
    print("\n" + "=" * 80)
    print("例1: fit_gmm_lp_simple - モーメント誤差最小化モード")
    print("=" * 80)
    
    lp_params_moments = {
        "solver": "highs",
        "sigma_min_scale": 0.5,
        "sigma_max_scale": 2.0,
        "lambda_mean": 1.0,
        "lambda_variance": 1.0,
        "lambda_skewness": 1.0,
        "lambda_kurtosis": 1.0,
        "pdf_tolerance": 1e-4,  # PDF誤差の上限
    }
    
    print("\nパラメータ:")
    print(f"  K (セグメント数): 5")
    print(f"  L (シグマレベル数): 3")
    print(f"  PDF tolerance: {lp_params_moments['pdf_tolerance']}")
    print(f"  Lambda (mean, variance, skewness, kurtosis): 1.0, 1.0, 1.0, 1.0")
    
    result_moments, timing_moments = fit_gmm_lp_simple(
        z, f_true,
        K=5,
        L=3,
        lp_params=lp_params_moments,
        objective_mode="moments"  # モーメント誤差最小化モード
    )
    
    print("\n結果:")
    print(f"  LP目的関数値: {result_moments['lp_objective']:.6e}")
    print(f"  PDF誤差上限 (t_pdf): {result_moments['diagnostics']['t_pdf']:.6e}")
    print(f"  平均誤差上限 (t_mean): {result_moments['diagnostics']['t_mean']:.6e}")
    print(f"  分散誤差上限 (t_var): {result_moments['diagnostics']['t_var']:.6e}")
    print(f"  歪度誤差上限 (t_skew): {result_moments['diagnostics']['t_skew']:.6e}")
    print(f"  尖度誤差上限 (t_kurt): {result_moments['diagnostics']['t_kurt']:.6e}")
    
    if "moment_errors" in result_moments['diagnostics']:
        me = result_moments['diagnostics']['moment_errors']
        print("\n実際のモーメント誤差:")
        print(f"  平均誤差: {me['mean']:.6e} (相対誤差: {me['mean_relative']:.6e})")
        print(f"  分散誤差: {me['variance']:.6e} (相対誤差: {me['variance_relative']:.6e})")
        print(f"  歪度誤差: {me['skewness']:.6e} (相対誤差: {me['skewness_relative']:.6e})")
        print(f"  尖度誤差: {me['kurtosis']:.6e} (相対誤差: {me['kurtosis_relative']:.6e})")
    
    print(f"\n実行時間:")
    print(f"  辞書生成: {timing_moments['dict_generation']:.4f}秒")
    print(f"  基底行列計算: {timing_moments['basis_computation']:.4f}秒")
    print(f"  LP求解: {timing_moments['lp_solving']:.4f}秒")
    print(f"  合計: {timing_moments['total']:.4f}秒")
    
    # ============================================================================
    # 例2: 従来のPDF/CDF誤差最小化モードと比較
    # ============================================================================
    print("\n" + "=" * 80)
    print("例2: fit_gmm_lp_simple - PDF誤差最小化モード（比較用）")
    print("=" * 80)
    
    lp_params_pdf = {
        "solver": "highs",
        "sigma_min_scale": 0.5,
        "sigma_max_scale": 2.0,
    }
    
    result_pdf, timing_pdf = fit_gmm_lp_simple(
        z, f_true,
        K=5,
        L=3,
        lp_params=lp_params_pdf,
        objective_mode="pdf"  # PDF誤差最小化モード
    )
    
    print("\n結果:")
    print(f"  LP目的関数値: {result_pdf['lp_objective']:.6e}")
    print(f"  PDF誤差上限 (t_pdf): {result_pdf['diagnostics']['t_pdf']:.6e}")
    # PDF only mode - no CDF error
    
    # 近似GMMの統計量を計算
    from em_method import gmm1d_pdf, GMM1DParams, normalize_pdf_on_grid
    params = GMM1DParams(
        pi=result_pdf['weights'],
        mu=result_pdf['mus'],
        var=result_pdf['sigmas']**2
    )
    f_approx = gmm1d_pdf(z, params)
    f_approx = normalize_pdf_on_grid(z, f_approx)
    stats_approx = compute_pdf_statistics(z, f_approx)
    
    print("\n近似GMMの統計量:")
    print(f"  平均: {stats_approx['mean']:.6f} (誤差: {stats_approx['mean'] - stats_true['mean']:.6e})")
    print(f"  分散: {stats_approx['std']**2:.6f} (誤差: {stats_approx['std']**2 - stats_true['std']**2:.6e})")
    print(f"  歪度: {stats_approx['skewness']:.6f} (誤差: {stats_approx['skewness'] - stats_true['skewness']:.6e})")
    print(f"  尖度: {stats_approx['kurtosis']:.6f} (誤差: {stats_approx['kurtosis'] - stats_true['kurtosis']:.6e})")
    
    # ============================================================================
    # 例3: 異なる重みパラメータでの比較
    # ============================================================================
    print("\n" + "=" * 80)
    print("例3: 異なる重みパラメータでの比較")
    print("=" * 80)
    
    # 平均と分散に重点を置く設定
    lp_params_mean_var = {
        "solver": "highs",
        "sigma_min_scale": 0.5,
        "sigma_max_scale": 2.0,
        "lambda_mean": 10.0,      # 平均を重視
        "lambda_variance": 10.0,  # 分散を重視
        "lambda_skewness": 0.1,   # 歪度は軽視
        "lambda_kurtosis": 0.1,   # 尖度は軽視
        "pdf_tolerance": 1e-4,
    }
    
    print("\nパラメータ（平均・分散重視）:")
    print(f"  lambda_mean: {lp_params_mean_var['lambda_mean']}")
    print(f"  lambda_variance: {lp_params_mean_var['lambda_variance']}")
    print(f"  lambda_skewness: {lp_params_mean_var['lambda_skewness']}")
    print(f"  lambda_kurtosis: {lp_params_mean_var['lambda_kurtosis']}")
    
    result_mean_var, timing_mean_var = fit_gmm_lp_simple(
        z, f_true,
        K=5,
        L=3,
        lp_params=lp_params_mean_var,
        objective_mode="moments"
    )
    
    print("\n結果:")
    print(f"  LP目的関数値: {result_mean_var['lp_objective']:.6e}")
    print(f"  PDF誤差上限 (t_pdf): {result_mean_var['diagnostics']['t_pdf']:.6e}")
    
    if "moment_errors" in result_mean_var['diagnostics']:
        me = result_mean_var['diagnostics']['moment_errors']
        print("\n実際のモーメント誤差:")
        print(f"  平均誤差: {me['mean']:.6e} (相対誤差: {me['mean_relative']:.6e})")
        print(f"  分散誤差: {me['variance']:.6e} (相対誤差: {me['variance_relative']:.6e})")
        print(f"  歪度誤差: {me['skewness']:.6e} (相対誤差: {me['skewness_relative']:.6e})")
        print(f"  尖度誤差: {me['kurtosis']:.6e} (相対誤差: {me['kurtosis_relative']:.6e})")
    
    # ============================================================================
    # まとめ
    # ============================================================================
    print("\n" + "=" * 80)
    print("まとめ")
    print("=" * 80)
    print("\nモーメント誤差最小化モードの特徴:")
    print("  - PDF誤差を制約として、モーメント（平均、分散、歪度、尖度）の相対誤差を最小化")
    print("  - 各モーメントの重み（lambda_mean, lambda_variance等）を調整可能")
    print("  - PDF toleranceでPDF誤差の上限を制御")
    print("\n使用例:")
    print("  - 統計量の一致を重視したい場合")
    print("  - PDF形状よりもモーメントの精度が重要な場合")
    print("  - 特定のモーメント（例：平均・分散）を優先したい場合")
    
    print("\n" + "=" * 80)
    print("実行例完了")
    print("=" * 80)


if __name__ == "__main__":
    main()

