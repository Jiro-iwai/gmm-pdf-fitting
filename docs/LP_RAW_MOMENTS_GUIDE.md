# LP法 Raw Momentsモード実行ガイド

## 概要

LP法のRaw Momentsモードは、生モーメント（M1-M4）の誤差を最小化しながらPDF誤差を制約条件として扱う、完全に線形な最適化問題を解きます。

## 実行方法

### 1. 基本的な実行

```bash
python main.py --config configs/config_lp_rawmoments.json
```

### 2. 設定ファイルの構成

Raw Momentsモードを使用するには、設定ファイルで以下を指定します：

```json
{
  "method": "lp",
  "objective_mode": "raw_moments",
  "lp_params": {
    "solver": "highs",
    "pdf_tolerance": null,
    "lambda_pdf": 1.0,
    "lambda_raw": [1.0, 1.0, 1.0, 1.0],
    "objective_form": "A"
  }
}
```

## パラメータ説明

### 必須パラメータ

- **`method`**: `"lp"` を指定
- **`objective_mode`**: `"raw_moments"` を指定

### `lp_params`内のパラメータ

#### 基本パラメータ

- **`solver`** (str, デフォルト: `"highs"`)
  - LPソルバー。`"highs"`, `"scipy"` などが使用可能

- **`sigma_min_scale`** (float, デフォルト: `0.1`)
  - ガウス基底の標準偏差の最小スケール

- **`sigma_max_scale`** (float, デフォルト: `3.0`)
  - ガウス基底の標準偏差の最大スケール

#### Raw Momentsモード専用パラメータ

- **`pdf_tolerance`** (float | null, デフォルト: `null`)
  - **Objective Form Aの場合**: PDF誤差の上限値（制約条件）
  - `null`の場合は上限なし
  - 実行不可能な場合は自動的に緩和される（最大3回）

- **`lambda_pdf`** (float, デフォルト: `1.0`)
  - **Objective Form Bの場合**: PDF誤差項の重み

- **`lambda_raw`** (list[float], デフォルト: `[1.0, 1.0, 1.0, 1.0]`)
  - 生モーメント誤差の重み `[λ_1, λ_2, λ_3, λ_4]`
  - それぞれM1（平均）、M2、M3、M4に対応

- **`objective_form`** (`"A"` | `"B"`, デフォルト: `"A"`)
  - **Form A**: `minimize Σ λ_n * t_n` subject to `t_pdf <= pdf_tolerance` (if specified)
    - PDF誤差を制約条件として扱い、生モーメント誤差を最小化
    - `pdf_tolerance`が指定されている場合、その値が上限として使用される
    - `pdf_tolerance`が`null`の場合、PDF誤差の制約は適用されない（生モーメント誤差のみ最小化）
  - **Form B**: `minimize λ_pdf * t_pdf + Σ λ_n * t_n`
    - PDF誤差と生モーメント誤差の重み付き和を最小化
    - `pdf_tolerance`は不要（`null`でOK）

## Objective Formの選択

### Form A（推奨）

PDF誤差を一定以下に保ちながら、生モーメント誤差を最小化します。

```json
{
  "objective_form": "A",
  "pdf_tolerance": 0.01,
  "lambda_raw": [1.0, 1.0, 1.0, 1.0]
}
```

**特徴**:
- `pdf_tolerance`が指定されている場合、PDF誤差がその値以下であることが保証される
- `pdf_tolerance`が`null`の場合、PDF誤差の制約は適用されない
- 実行不可能な場合、`pdf_tolerance`が自動的に緩和される（最大3回）
- PDF精度を優先したい場合に適している

### Form B

PDF誤差と生モーメント誤差のバランスを調整します。

```json
{
  "objective_form": "B",
  "lambda_pdf": 1.0,
  "lambda_raw": [1.0, 1.0, 1.0, 1.0]
}
```

**特徴**:
- PDF誤差とモーメント誤差の重みを自由に調整できる
- `lambda_pdf`を大きくするとPDF精度が向上
- `lambda_raw`を大きくするとモーメント精度が向上

## 実行例

### 例1: Form A（PDF誤差を0.01以下に制約）

```json
{
  "method": "lp",
  "objective_mode": "raw_moments",
  "K": 10,
  "L": 10,
  "lp_params": {
    "solver": "highs",
    "objective_form": "A",
    "pdf_tolerance": 0.01,
    "lambda_raw": [1.0, 1.0, 1.0, 1.0]
  }
}
```

### 例2: Form B（PDFとモーメントのバランス）

```json
{
  "method": "lp",
  "objective_mode": "raw_moments",
  "K": 10,
  "L": 10,
  "lp_params": {
    "solver": "highs",
    "objective_form": "B",
    "lambda_pdf": 10.0,
    "lambda_raw": [1.0, 1.0, 1.0, 1.0]
  }
}
```

### 例3: 高次モーメントを重視

```json
{
  "method": "lp",
  "objective_mode": "raw_moments",
  "K": 15,
  "L": 15,
  "lp_params": {
    "solver": "highs",
    "objective_form": "A",
    "pdf_tolerance": 0.02,
    "lambda_raw": [0.1, 0.1, 1.0, 10.0]
  }
}
```

## 出力

実行すると以下の情報が表示されます：

1. **PDF比較プロット**: 真のPDFと近似PDFの比較
2. **統計情報**: 平均、標準偏差、歪度、尖度の比較
3. **診断情報**: 
   - PDF誤差 (`t_pdf`)
   - 生モーメント誤差 (`t_raw`)
   - 使用された基底関数の数
   - 非ゼロ重みの数

## トラブルシューティング

### 実行不可能エラー

**問題**: `LP solve failed: The problem is infeasible`

**解決策**:
1. `pdf_tolerance`を大きくする（Form Aの場合）
2. `L`（辞書サイズ）を増やす
3. `K`（成分数）を増やす
4. `lambda_pdf`を小さくする（Form Bの場合）

### モーメント誤差が大きい

**問題**: モーメント誤差が期待より大きい

**解決策**:
1. `lambda_raw`の対応する要素を大きくする
2. `K`や`L`を増やす
3. `objective_form`を`"A"`に変更し、`pdf_tolerance`を緩和する

## ベンチマークでの使用

ベンチマークスクリプトでも使用可能です：

```bash
python benchmark.py --config configs/config_lp_rawmoments.json --output benchmark_rawmoments.json
```

ベンチマークでは自動的に`pdf`、`moments`、`raw_moments`の3つのモードが評価されます。

## 参考

- `lp_method.py`: `solve_lp_pdf_rawmoments_linf`関数の実装
- `main.py`: LP法の実行ロジック
- `benchmark.py`: 性能評価スクリプト

