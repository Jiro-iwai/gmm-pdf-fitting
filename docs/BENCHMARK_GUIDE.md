# 包括的性能評価ガイド

## 概要

このドキュメントでは、新しく実装された機能を含む包括的な性能評価の実行方法を説明します。

## 新機能の評価

以下の新機能が評価対象に含まれています：

1. **LP法の`raw_moments`モード**: 完全線形の生モーメントマッチング
2. **Hybrid法**: LP→EM→QPの統合手法
3. **Tail-focused辞書生成**: 右裾・左裾・両裾重視の辞書生成
4. **新しい評価指標**: CDF L∞誤差、右裾L1誤差、分位点誤差

## 実行方法

### 1. EM法のベンチマーク

```bash
python benchmarks/benchmark.py --config configs/config_em.json --output benchmarks/results/benchmark_em.json --vary-params --plot
```

### 2. LP法のベンチマーク（PDF、Raw Momentsモード）

**注意**: LP法のMomentsモードは実行速度の都合によりスキップされています。

```bash
# LP法用の設定ファイルを作成（method="lp"に設定）
python benchmarks/benchmark.py --config configs/config_lp.json --output benchmarks/results/benchmark_lp.json --vary-params --plot
```

### 3. Hybrid法のベンチマーク

```bash
# Hybrid法用の設定ファイルを作成（method="hybrid"に設定）
python benchmarks/benchmark.py --config configs/config_hybrid.json --output benchmarks/results/benchmark_hybrid.json --vary-params --plot
```

### 4. 一括実行スクリプト

```bash
./benchmarks/run_comprehensive_benchmark.sh
```

## 評価指標

### PDF/CDF誤差
- **PDF L∞誤差**: `max|f_true - f_hat|`
- **CDF L∞誤差**: `max|F_true - F_hat|`
- **PDF L2誤差**: `sqrt(∫(f_true - f_hat)² dz)`
- **右裾L1誤差**: 90パーセンタイル以降の重み付きL1誤差

### モーメント誤差（相対誤差）
- **平均誤差**: `|mean_hat - mean_true| / |mean_true| × 100%`
- **標準偏差誤差**: `|std_hat - std_true| / |std_true| × 100%`
- **歪度誤差**: `|skew_hat - skew_true| / |skew_true| × 100%`
- **尖度誤差**: `|kurt_hat - kurt_true| / |kurt_true| × 100%`

### 実行時間
- **総実行時間**: アルゴリズム全体の実行時間
- **LP実行時間**: LPソルバーの実行時間（Hybrid法のみ）
- **EM実行時間**: EMアルゴリズムの実行時間（Hybrid法のみ）
- **QP実行時間**: QP射影の実行時間（Hybrid法のみ、モーメントマッチング使用時）

## 評価パラメータ

### 基本パラメータセット
デフォルトで15種類のパラメータセットが評価されます：
- 異なる相関係数（ρ = 0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99）
- 異なる平均オフセット
- 異なる分散比

### K値（コンポーネント数）
- 全手法: K = 3, 4, 5, 10, 15, 20（統一）

### グリッド解像度
- EM法・Hybrid法: z_npoints = 8, 16, 32, 64, 128, 256, 512
- LP法: z_npoints = 32, 64, 128（最適化された設定）

### L値（LP法のシグマレベル数）
- LP法: L = 10（最適化された設定）
- EM法・Hybrid法: L = 10（使用されない）

## 結果の解釈

### サマリー統計
ベンチマーク実行後、以下の統計が表示されます：
- 各手法・モードの平均実行時間
- 平均PDF/CDF誤差
- 平均モーメント誤差（相対誤差）

### パラメータセット別統計
複数のパラメータセットを評価した場合、各パラメータセットごとの統計も表示されます。

### 可視化
`--plot`オプションを使用すると、以下のプロットが生成されます：
- 実行時間 vs K
- PDF L∞誤差 vs K
- 標準偏差誤差 vs K
- 実行時間 vs グリッド解像度
- PDF L2誤差 vs グリッド解像度
- 歪度誤差 vs K
- 尖度誤差 vs K
- 実行時間 vs 精度トレードオフ

## 注意事項

1. **実行時間**: 包括的な評価には時間がかかります（数時間〜数日）
2. **メモリ使用量**: 大規模な辞書サイズではメモリ使用量が増加します
3. **LPソルバー**: `highs`ソルバーが推奨されます（デフォルト）

## カスタム設定

### パラメータセットのカスタマイズ

`param_configs.json`ファイルを作成して、評価するパラメータセットを指定できます：

```json
[
  {
    "mu_x": 0.1,
    "sigma_x": 0.4,
    "mu_y": 0.15,
    "sigma_y": 0.9,
    "rho": 0.9,
    "z_range": [-2.0, 4.0]
  }
]
```

```bash
python benchmarks/benchmark.py --config configs/config_default.json --param-configs param_configs.json --output benchmarks/results/results.json
```

## トラブルシューティング

### LPソルバーが失敗する場合
- `pdf_tolerance`を緩和する（例: 0.01 → 0.05）
- 辞書サイズを増やす（`dict_J`、`dict_L`を増やす）
- `K`を増やす

### メモリ不足の場合
- グリッド解像度を下げる（`z_npoints`を減らす）
- 辞書サイズを減らす
- パラメータセット数を減らす

