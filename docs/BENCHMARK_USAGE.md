# benchmark.py 使い方ガイド

## 概要

`benchmark.py`は、GMMフィッティング手法（EM法、LP法、Hybrid法）の包括的な性能評価を行うスクリプトです。

## 基本的な使い方

### コマンドライン引数

```bash
python benchmarks/benchmark.py [OPTIONS]
```

### 必須オプション

- **`--config CONFIG`**: 設定ファイルのパス（必須）
  - 例: `configs/config_default.json`
  - 設定ファイルには`method`（`"em"`, `"lp"`, `"hybrid"`）を指定

### オプション

- **`--output OUTPUT`**: 結果を保存するJSONファイルのパス
  - デフォルト: `benchmark_results.json`
  - 例: `benchmarks/results/benchmark_em.json`

- **`--plot`**: 可視化プロットを生成
  - プロットファイルは`--output`で指定したファイル名の`.json`を`.png`に置き換えた名前で保存されます

- **`--vary-params`**: 複数の基本パラメータセットを評価
  - デフォルトで15種類のパラメータセットが評価されます
  - 異なる相関係数、平均オフセット、分散比をカバー

- **`--param-configs PARAM_CONFIGS`**: カスタムパラメータセットのJSONファイルパス
  - 評価するパラメータセットを指定できます

## 実行例

### 1. 基本的な実行（単一パラメータセット）

```bash
# EM法のベンチマーク
python benchmarks/benchmark.py \
    --config configs/config_default.json \
    --output benchmarks/results/benchmark_em.json

# LP法のベンチマーク
python benchmarks/benchmark.py \
    --config configs/config_lp.json \
    --output benchmarks/results/benchmark_lp.json

# Hybrid法のベンチマーク
python benchmarks/benchmark.py \
    --config configs/config_hybrid.json \
    --output benchmarks/results/benchmark_hybrid.json
```

### 2. プロット付き実行

```bash
python benchmarks/benchmark.py \
    --config configs/config_lp.json \
    --output benchmarks/results/benchmark_lp.json \
    --plot
```

### 3. 複数パラメータセットでの評価

```bash
python benchmarks/benchmark.py \
    --config configs/config_default.json \
    --output benchmarks/results/benchmark_varied.json \
    --vary-params \
    --plot
```

### 4. カスタムパラメータセットでの評価

まず、`param_configs.json`ファイルを作成：

```json
[
  {
    "mu_x": 0.1,
    "sigma_x": 0.4,
    "mu_y": 0.15,
    "sigma_y": 0.9,
    "rho": 0.9,
    "z_range": [-2.0, 4.0]
  },
  {
    "mu_x": 0.0,
    "sigma_x": 1.0,
    "mu_y": 0.0,
    "sigma_y": 1.0,
    "rho": 0.5,
    "z_range": [-4.0, 4.0]
  }
]
```

実行：

```bash
python benchmarks/benchmark.py \
    --config configs/config_default.json \
    --param-configs param_configs.json \
    --output benchmarks/results/results.json
```

## 評価される項目

### パラメータの組み合わせ

ベンチマークは以下のパラメータの組み合わせで実行されます：

- **K値（成分数）**: `[3, 4, 5, 10, 15, 20]`（全手法で統一）

- **グリッド解像度**: 
  - EM法・Hybrid法: `[8, 16, 32, 64, 128, 256, 512]`
  - LP法: `[32, 64, 128]`（最適化された設定）

- **L値（LP法のシグマレベル数）**: 
  - LP法: `[10]`（最適化された設定）
  - EM法・Hybrid法: `[10]`（使用されない）

- **モード**:
  - EM法: PDF-only、Moment-matching
  - LP法: PDF、Raw Moments（Momentsモードは実行速度の都合によりスキップ）
  - Hybrid法: Moment-matching有無

### 評価指標

各実行で以下の指標が測定されます：

1. **実行時間**: アルゴリズムの実行時間（秒）
2. **PDF誤差**:
   - L∞誤差: `max|f_true - f_hat|`
   - L2誤差: `sqrt(∫(f_true - f_hat)² dz)`
3. **CDF誤差**:
   - L∞誤差: `max|F_true - F_hat|`
4. **右裾誤差**:
   - 重み付きL1誤差（90パーセンタイル以降）
5. **モーメント誤差**（相対誤差）:
   - 平均: `|mean_hat - mean_true| / |mean_true| × 100%`
   - 標準偏差: `|std_hat - std_true| / |std_true| × 100%`
   - 歪度: `|skew_hat - skew_true| / |skew_true| × 100%`
   - 尖度: `|kurt_hat - kurt_true| / |kurt_true| × 100%`

## 出力

### コンソール出力

実行中に以下の情報が表示されます：

```
================================================================================
Comprehensive Performance Benchmark
================================================================================

Parameter Set 1/15
  mu_x=0.0, sigma_x=0.8
  mu_y=0.0, sigma_y=1.6, rho=0.9
  z_range=[-6.0, 8.0]

Running benchmarks...
  K values: [3, 4, 5, 10, 15, 20]
  Grid resolutions: [8, 16, 32, 64, 128, 256, 512]
  EM modes: PDF-only and Moment-matching

  Grid resolution: 64 points
    [1/40] K=4, L=10... ✓ pdf (0.1517s) ✓ moments (0.1561s)
    ...
```

実行完了後、統計サマリーが表示されます：

```
================================================================================
Summary Statistics (Overall)
================================================================================

EM Method (PDF-only mode):
  Average execution time: 0.2886s
  Average PDF L∞ error: 0.001292
  Average mean error (rel): 0.0001%
  Average std error (rel): 0.0001%
  Average skewness error (rel): 0.1691%
  Average kurtosis error (rel): 1.2522%

EM Method (Moment-matching mode):
  Average execution time: 0.2895s
  Average PDF L∞ error: 0.001548
  Average mean error (rel): 0.0011%
  Average std error (rel): 0.0065%
  Average skewness error (rel): 0.0442%
  Average kurtosis error (rel): 0.1811%
```

### JSON出力

結果はJSON形式で保存されます。各結果には以下の情報が含まれます：

```json
{
  "method": "em",
  "K": 10,
  "z_npoints": 128,
  "objective_mode": "pdf",
  "execution_time": 0.2886,
  "pdf_error_linf": 0.001292,
  "pdf_error_l2": 0.001363,
  "cdf_error_linf": 0.000123,
  "tail_l1_error": 0.000045,
  "mean_error_rel": 0.0001,
  "std_error_rel": 0.0001,
  "skewness_error_rel": 0.1691,
  "kurtosis_error_rel": 1.2522,
  ...
}
```

### プロット出力（`--plot`オプション使用時）

以下のプロットが生成されます：

- 実行時間 vs K
- PDF L∞誤差 vs K
- PDF L2誤差 vs K
- CDF L∞誤差 vs K
- 標準偏差誤差 vs K
- 歪度誤差 vs K
- 尖度誤差 vs K
- 実行時間 vs グリッド解像度
- 実行時間 vs 精度トレードオフ

## クイックモード

設定ファイルに`"quick_benchmark": true`を追加すると、評価パラメータが削減され、実行時間が短縮されます：

```json
{
  "method": "lp",
  "quick_benchmark": true,
  ...
}
```

クイックモードでは：
- K値: `[5, 10]`（通常は`[3, 4, 5, 10, 15, 20]`）
- グリッド解像度: `[32, 64, 128]`（通常はEM法・Hybrid法で`[8, 16, 32, 64, 128, 256, 512]`、LP法で`[32, 64, 128]`）
- L値: `[10]`（通常も`[10]`）

## 一括実行スクリプト

### 包括的ベンチマーク

```bash
./benchmarks/run_comprehensive_benchmark.sh
```

EM法、LP法、Hybrid法のすべてを評価します。

### クイックベンチマーク

```bash
./benchmarks/run_quick_benchmark.sh
```

クイックモードで各手法を評価します。

## トラブルシューティング

### LPソルバーが失敗する場合

**エラー**: `LP solve failed: The problem is infeasible`

**解決策**:
1. `pdf_tolerance`を緩和する（例: `0.01` → `0.05`）
2. 辞書サイズを増やす（`dict_J`、`dict_L`を増やす）
3. `K`を増やす
4. `L`を増やす

### メモリ不足の場合

**解決策**:
1. グリッド解像度を下げる（`z_npoints`を減らす）
2. 辞書サイズを減らす
3. パラメータセット数を減らす（`--vary-params`を使わない）
4. クイックモードを使用（`quick_benchmark: true`）

### 実行時間が長すぎる場合

**解決策**:
1. クイックモードを使用（`quick_benchmark: true`）
2. `--vary-params`を使わない（単一パラメータセットのみ）
3. K値の範囲を減らす（設定ファイルで調整）
4. グリッド解像度を減らす

## 設定ファイルの例

### EM法の設定

```json
{
  "method": "em",
  "K": 10,
  "use_moment_matching": false,
  "qp_mode": "hard",
  "max_iter": 100,
  "tol": 1e-6
}
```

### LP法の設定

```json
{
  "method": "lp",
  "K": 10,
  "L": 10,
  "objective_mode": "raw_moments",
  "lp_params": {
    "solver": "highs",
    "pdf_tolerance": 0.01,
    "lambda_raw": [1.0, 1.0, 1.0, 1.0],
    "objective_form": "A"
  }
}
```

### Hybrid法の設定

```json
{
  "method": "hybrid",
  "K": 10,
  "L": 10,
  "lp_params": {
    "dict_J": 20,
    "dict_L": 10,
    "objective_mode": "raw_moments",
    "tail_focus": "right",
    "tail_alpha": 2.0
  },
  "use_moment_matching": false
}
```

## 参考

- `docs/BENCHMARK_GUIDE.md`: 詳細なベンチマークガイド
- `docs/CONFIG_GUIDE.md`: 設定ファイルの詳細ガイド
- `benchmarks/run_comprehensive_benchmark.sh`: 一括実行スクリプト

