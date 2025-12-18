#!/usr/bin/env python3
"""
実行例: LP法のPDF誤差最小化モード（デフォルトモード）

このスクリプトは、PDF誤差最小化モードの使用例を示します。

実行方法:
    python examples/example_pdf_mode.py
    または
    cd examples && python example_pdf_mode.py
"""

import sys
from pathlib import Path

# Add src directory to path to import gmm_fitting package
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from gmm_fitting import fit_gmm_lp_simple
from gmm_fitting.em_method import normal_pdf


def main():
    print("=" * 80)
    print("LP法のPDF誤差最小化モード - 実行例")
    print("=" * 80)
    
    # グリッドの設定
    z = np.linspace(-5, 5, 200)
    
    # 真のPDF: 正規分布 N(0, 1)
    print("\n真のPDF: 正規分布 N(0, 1)")
    f_true = normal_pdf(z, 0.0, 1.0)
    
    # ============================================================================
    # PDF誤差最小化モード（デフォルト）
    # ============================================================================
    print("\n" + "=" * 80)
    print("PDF誤差最小化モード（objective_mode='pdf'）")
    print("=" * 80)
    
    lp_params_pdf = {
        "solver": "highs",
        "sigma_min_scale": 0.5,
        "sigma_max_scale": 2.0,
    }
    
    print("\nパラメータ:")
    print(f"  K (セグメント数): 5")
    print(f"  L (シグマレベル数): 3")
    print("  注: PDF誤差のみを最小化（CDFは考慮しない）")
    
    # objective_mode="pdf" を明示的に指定（デフォルトなので省略可能）
    result_pdf, timing_pdf = fit_gmm_lp_simple(
        z, f_true,
        K=5,
        L=3,
        lp_params=lp_params_pdf,
        objective_mode="pdf"  # デフォルトモード
    )
    
    print("\n結果:")
    print(f"  LP目的関数値: {result_pdf['lp_objective']:.6e}")
    print(f"  PDF誤差上限 (t_pdf): {result_pdf['diagnostics']['t_pdf']:.6e}")
    print(f"  非ゼロ成分数: {result_pdf['diagnostics']['n_nonzero']}")
    print("  注: CDF誤差は考慮されません（PDF誤差のみ最小化）")
    
    print(f"\n実行時間:")
    print(f"  辞書生成: {timing_pdf['dict_generation']:.4f}秒")
    print(f"  基底行列計算: {timing_pdf['basis_computation']:.4f}秒")
    print(f"  LP求解: {timing_pdf['lp_solving']:.4f}秒")
    print(f"  合計: {timing_pdf['total']:.4f}秒")
    
    # ============================================================================
    # objective_modeを省略した場合（デフォルトでpdfが使われる）
    # ============================================================================
    print("\n" + "=" * 80)
    print("objective_modeを省略した場合（デフォルトでpdfが使われる）")
    print("=" * 80)
    
    result_default, timing_default = fit_gmm_lp_simple(
        z, f_true,
        K=5,
        L=3,
        lp_params=lp_params_pdf
        # objective_modeを省略 → デフォルトで "pdf" が使用される
    )
    
    print("\n結果:")
    print(f"  使用されたモード: {result_default['diagnostics']['objective_mode']}")
    print(f"  LP目的関数値: {result_default['lp_objective']:.6e}")
    print(f"  PDF誤差上限 (t_pdf): {result_default['diagnostics']['t_pdf']:.6e}")
    print("  注: CDF誤差は考慮されません（PDF誤差のみ最小化）")
    
    # ============================================================================
    # 異なる重みパラメータでの比較
    # ============================================================================
    print("\n" + "=" * 80)
    print("異なる重みパラメータでの比較")
    print("=" * 80)
    
    # 異なるシグマスケールでの比較
    lp_params_pdf_narrow = {
        "solver": "highs",
        "sigma_min_scale": 0.3,  # より狭い範囲
        "sigma_max_scale": 1.5,  # より狭い範囲
    }
    
    print("\nパラメータ（シグマ範囲を狭く）:")
    print(f"  sigma_min_scale: {lp_params_pdf_narrow['sigma_min_scale']}")
    print(f"  sigma_max_scale: {lp_params_pdf_narrow['sigma_max_scale']}")
    
    result_pdf_narrow, _ = fit_gmm_lp_simple(
        z, f_true,
        K=5,
        L=3,
        lp_params=lp_params_pdf_narrow,
        objective_mode="pdf"
    )
    
    print("\n結果:")
    print(f"  PDF誤差上限 (t_pdf): {result_pdf_narrow['diagnostics']['t_pdf']:.6e}")
    print("  注: PDF誤差のみを最小化（CDFは考慮しない）")
    
    print("\n" + "=" * 80)
    print("まとめ")
    print("=" * 80)
    print("\nPDF誤差最小化モードの特徴:")
    print("  - PDF誤差のみを最小化（CDFは考慮しない）")
    print("  - デフォルトモード（objective_modeを省略した場合）")
    print("  - PDF形状を重視した近似")
    print("\n使用例:")
    print("  - PDF形状を重視したい場合")
    print("  - CDFよりもPDFの精度が重要な場合")
    print("  - モーメントよりも分布形状を重視したい場合")


if __name__ == "__main__":
    main()

