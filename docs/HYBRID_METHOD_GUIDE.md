# Hybrid法実行ガイド

## 概要

Hybrid法は、LP（線形計画法）→EM（期待値最大化法）→QP（二次計画法）の3ステップで構成される統合的なGMMフィッティング手法です。

### ワークフロー

1. **LPステップ**: Raw MomentsモードでLPを解き、初期成分を選択
2. **EMステップ**: LPで選択された成分を初期値としてEMアルゴリズムで最適化
3. **QPステップ**（オプション）: モーメントマッチングが必要な場合、QPでモーメントを強制

## 実行方法

### 1. 基本的な実行

```bash
python main.py --config configs/config_hybrid.json
```

### 2. 設定ファイルの構成

Hybrid法を使用するには、設定ファイルで以下を指定します：

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
  "use_moment_matching": false,
  "init": "custom"
}
```

## パラメータ説明

### 必須パラメータ

- **`method`**: `"hybrid"` を指定
- **`K`**: 最終的なGMM成分数
- **`L`**: 辞書の標準偏差レベル数（`dict_L`のデフォルト値として使用）

### `lp_params`内のパラメータ

#### 辞書パラメータ

- **`dict_J`** (int, デフォルト: `4 * K`)
  - 辞書の平均値（μ）の数
  - 大きいほど多くの候補から選択可能だが、計算時間が増加

- **`dict_L`** (int, デフォルト: `L`)
  - 辞書の標準偏差（σ）のレベル数
  - 各平均値に対して`dict_L`個の標準偏差が生成される

- **`mu_mode`** (str, デフォルト: `"quantile"`)
  - 平均値の生成方法: `"quantile"` または `"uniform"`

- **`tail_focus`** (str, デフォルト: `"none"`)
  - 裾に焦点を当てた辞書生成: `"none"`, `"right"`, `"left"`, `"both"`
  - 右裾に焦点を当てる場合は `"right"` を推奨

- **`tail_alpha`** (float, デフォルト: `1.0`)
  - 裾焦点の強度（1.0-10.0）
  - 大きいほど裾に集中

#### LPソルバーパラメータ

- **`objective_mode`** (str, 必須: `"raw_moments"`)
  - Hybrid法では`"raw_moments"`のみサポート

- **`solver`** (str, デフォルト: `"highs"`)
  - LPソルバー: `"highs"`, `"scipy"` など

- **`pdf_tolerance`** (float | null, デフォルト: `null`)
  - PDF誤差の上限（Objective Form Aの場合）
  - `null`の場合はPDF誤差の制約なし（生モーメント誤差のみ最小化）
  - 指定されている場合、その値が上限として使用される
  - 実行不可能な場合、自動的に緩和される（最大3回）

- **`lambda_pdf`** (float, デフォルト: `1.0`)
  - PDF誤差項の重み（Objective Form Bの場合）

- **`lambda_raw`** (list[float], デフォルト: `[1.0, 1.0, 1.0, 1.0]`)
  - 生モーメント誤差の重み `[λ_1, λ_2, λ_3, λ_4]`

- **`objective_form`** (`"A"` | `"B"`, デフォルト: `"A"`)
  - Objective Form A: PDF誤差を制約条件として扱う
  - Objective Form B: PDF誤差とモーメント誤差の重み付き和を最小化

- **`sigma_min_scale`** (float, デフォルト: `0.1`)
  - ガウス基底の標準偏差の最小スケール

- **`sigma_max_scale`** (float, デフォルト: `3.0`)
  - ガウス基底の標準偏差の最大スケール

### EMパラメータ

- **`max_iter`** (int, デフォルト: `100`)
  - EMアルゴリズムの最大反復回数

- **`tol`** (float, デフォルト: `1e-6`)
  - EMアルゴリズムの収束判定閾値

- **`reg_var`** (float, デフォルト: `1e-6`)
  - 分散の正則化パラメータ（最小値）

- **`n_init`** (int, デフォルト: `1`)
  - 初期化の試行回数（Hybrid法では`1`で十分）

- **`seed`** (int, デフォルト: `0`)
  - 乱数シード

- **`init`** (str, 必須: `"custom"`)
  - Hybrid法では`"custom"`を指定（LPの結果を使用）

### モーメントマッチングパラメータ（オプション）

- **`use_moment_matching`** (bool, デフォルト: `false`)
  - `true`にするとQPステップでモーメントを強制

- **`qp_mode`** (str, デフォルト: `"hard"`)
  - QPモード: `"hard"`（厳密）または `"soft"`（緩和）

- **`soft_lambda`** (float, デフォルト: `1e4`)
  - ソフト制約の重み（`qp_mode="soft"`の場合）

## 実行例

### 例1: 基本的なHybrid法（モーメントマッチングなし）

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
    "tail_alpha": 2.0,
    "pdf_tolerance": 0.01,
    "lambda_raw": [1.0, 1.0, 1.0, 1.0],
    "objective_form": "A"
  },
  "use_moment_matching": false,
  "init": "custom"
}
```

### 例2: モーメントマッチング付きHybrid法

```json
{
  "method": "hybrid",
  "K": 15,
  "L": 10,
  "lp_params": {
    "dict_J": 30,
    "dict_L": 10,
    "objective_mode": "raw_moments",
    "tail_focus": "right",
    "tail_alpha": 2.0,
    "pdf_tolerance": 0.02,
    "lambda_raw": [0.1, 0.1, 1.0, 10.0],
    "objective_form": "A"
  },
  "use_moment_matching": true,
  "qp_mode": "hard",
  "init": "custom"
}
```

### 例3: 大規模辞書での実行

```json
{
  "method": "hybrid",
  "K": 20,
  "L": 15,
  "lp_params": {
    "dict_J": 50,
    "dict_L": 15,
    "objective_mode": "raw_moments",
    "tail_focus": "both",
    "tail_alpha": 3.0,
    "pdf_tolerance": 0.01,
    "lambda_raw": [1.0, 1.0, 1.0, 1.0],
    "objective_form": "A"
  },
  "use_moment_matching": false,
  "init": "custom"
}
```

## 出力

実行すると以下の情報が表示されます：

1. **実行時間の内訳**:
   - LP runtime: LPステップの実行時間
   - EM runtime: EMステップの実行時間
   - QP runtime: QPステップの実行時間（使用時）

2. **辞書情報**:
   - Dictionary size: `dict_J × dict_L = 総基底数`

3. **PDF比較プロット**: 真のPDFと近似PDFの比較

4. **統計情報**: 平均、標準偏差、歪度、尖度の比較

5. **GMMパラメータ**: 最終的な成分の重み、平均、標準偏差

## トラブルシューティング

### LPステップで実行不可能エラー

**問題**: `LP solve failed: The problem is infeasible`

**解決策**:
1. `pdf_tolerance`を大きくする
2. `dict_J`や`dict_L`を増やす
3. `lambda_raw`の値を調整する

### EMステップで収束しない

**問題**: EMが最大反復回数に達する

**解決策**:
1. `max_iter`を増やす
2. `tol`を緩和する
3. LPステップの`pdf_tolerance`を緩和してより良い初期値を得る

### モーメント誤差が大きい

**問題**: モーメントマッチングを使用しても誤差が大きい

**解決策**:
1. `use_moment_matching: true`に設定
2. `qp_mode: "hard"`を使用
3. `K`を増やす
4. LPステップの`lambda_raw`で高次モーメントを重視

## ベンチマークでの使用

```bash
python benchmark.py --config configs/config_hybrid.json --output benchmark_hybrid.json
```

ベンチマークでは、`use_moment_matching`が`true`と`false`の両方が評価されます。

## 参考

- `main.py`: Hybrid法の実行ロジック（LP→EM→QP）
- `lp_method.py`: LPソルバー（`solve_lp_pdf_rawmoments_linf`）
- `em_method.py`: EMアルゴリズム（`fit_gmm1d_to_pdf_weighted_em`）
- `benchmark_hybrid.py`: Hybrid法のベンチマーク関数

