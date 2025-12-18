# 設定ファイルの書き方ガイド

このドキュメントでは、設定ファイル（JSON形式）の書き方と各パラメータの説明を提供します。

## 基本構造

設定ファイルはJSON形式です。**コメントは使用できません**。設定ファイルは`configs/`ディレクトリに配置します。

## 必須パラメータ

### 1. 2変量正規分布のパラメータ

元となる2変量正規分布 `(X, Y)` を定義し、その最大値 `Z = max(X, Y)` のPDFを計算するために使用されます。

```json
{
  "mu_x": 0.1,        // Xの平均値
  "sigma_x": 0.4,     // Xの標準偏差
  "mu_y": 0.15,       // Yの平均値
  "sigma_y": 0.9,     // Yの標準偏差
  "rho": 0.9          // XとYの相関係数（-1から1の間）
}
```

### 2. グリッド設定

PDFを計算するグリッドの範囲と点数を指定します。

```json
{
  "z_range": [-4, 4],    // グリッドの範囲 [最小値, 最大値]
  "z_npoints": 128       // グリッドの点数
}
```

### 3. GMMの成分数

```json
{
  "K": 4  // GMMの成分数（1以上の整数）
}
```

## 手法別の設定

### EM法（method="em"）

```json
{
  "method": "em",
  "max_iter": 20000,
  "tol": 1e-5,
  "reg_var": 1e-06,
  "n_init": 4,
  "seed": 1,
  "init": "quantile",
  "use_moment_matching": true,
  "qp_mode": "hard",
  "soft_lambda": 1e4
}
```

**パラメータ説明:**
- `max_iter`: 最大反復回数
- `tol`: 収束判定の許容誤差
- `reg_var`: 分散の正則化パラメータ
- `n_init`: 初期値の試行回数
- `seed`: 乱数シード
- `init`: 初期化方法（"quantile" または "wqmi"）
- `use_moment_matching`: モーメントマッチングを使用するか
- `qp_mode`: QPモード（"hard" または "soft"）
- `soft_lambda`: ソフト制約のペナルティ係数

### LP法（method="lp"）

#### 基本設定

```json
{
  "method": "lp",
  "K": 5,
  "L": 3
}
```

- `K`: セグメント数（fit_gmm_lp_simpleで使用）
- `L`: シグマレベル数（fit_gmm_lp_simpleで使用）

#### PDF誤差最小化モード（デフォルト）

```json
{
  "method": "lp",
  "objective_mode": "pdf",
  "lp_params": {
    "solver": "highs",
    "sigma_min_scale": 0.1,
    "sigma_max_scale": 3.0
  }
}
```

**パラメータ説明:**
- `objective_mode`: "pdf"（デフォルト）または "moments"
- `solver`: LPソルバー（"highs"を推奨）
- `sigma_min_scale`: 最小シグマスケール（真の標準偏差に対する比率）
- `sigma_max_scale`: 最大シグマスケール（真の標準偏差に対する比率）

#### モーメント誤差最小化モード（新機能）

```json
{
  "method": "lp",
  "objective_mode": "moments",
  "lp_params": {
    "solver": "highs",
    "sigma_min_scale": 0.5,
    "sigma_max_scale": 2.0,
    "lambda_mean": 1.0,
    "lambda_variance": 1.0,
    "lambda_skewness": 1.0,
    "lambda_kurtosis": 1.0,
    "pdf_tolerance": 1e-4
  }
}
```

**パラメータ説明:**
- `objective_mode`: "moments" を指定
- `lambda_mean`: 平均誤差の重み（デフォルト: 1.0）
- `lambda_variance`: 分散誤差の重み（デフォルト: 1.0）
- `lambda_skewness`: 歪度誤差の重み（デフォルト: 1.0）
- `lambda_kurtosis`: 尖度誤差の重み（デフォルト: 1.0）
- `pdf_tolerance`: PDF誤差の上限（制約として使用、デフォルト: 1e-6、推奨: 0.01〜0.05）
- `max_moment_iter`: モーメント制約の反復回数上限（デフォルト: 5、推奨: 5〜20）
- `moment_tolerance`: モーメント制約の収束判定許容誤差（デフォルト: 1e-6）

## 完全な設定例

### 例1: EM法の基本設定

```json
{
  "mu_x": 0.1,
  "sigma_x": 0.4,
  "mu_y": 0.15,
  "sigma_y": 0.9,
  "rho": 0.9,
  "z_range": [-4, 4],
  "z_npoints": 128,
  "K": 4,
  "method": "em",
  "max_iter": 20000,
  "tol": 1e-5,
  "reg_var": 1e-06,
  "n_init": 4,
  "seed": 1,
  "init": "quantile",
  "use_moment_matching": true,
  "qp_mode": "hard",
  "soft_lambda": 1e4,
  "output_path": "pdf_comparison",
  "show_grid_points": true,
  "max_grid_points_display": 200
}
```

### 例2: LP法（PDF誤差最小化モード）

```json
{
  "mu_x": 0.1,
  "sigma_x": 0.4,
  "mu_y": 0.15,
  "sigma_y": 0.9,
  "rho": 0.9,
  "z_range": [-4, 4],
  "z_npoints": 128,
  "K": 10,
  "L": 5,
  "method": "lp",
  "objective_mode": "pdf",
  "lp_params": {
    "solver": "highs",
    "sigma_min_scale": 0.1,
    "sigma_max_scale": 3.0
  },
  "output_path": "pdf_comparison_lp",
  "show_grid_points": true,
  "max_grid_points_display": 200
}
```

### 例3: LP法（モーメント誤差最小化モード）

```json
{
  "mu_x": 0.1,
  "sigma_x": 0.4,
  "mu_y": 0.15,
  "sigma_y": 0.9,
  "rho": 0.9,
  "z_range": [-4, 4],
  "z_npoints": 128,
  "K": 5,
  "L": 3,
  "method": "lp",
  "objective_mode": "moments",
  "lp_params": {
    "solver": "highs",
    "sigma_min_scale": 0.5,
    "sigma_max_scale": 2.0,
    "lambda_mean": 1.0,
    "lambda_variance": 1.0,
    "lambda_skewness": 1.0,
    "lambda_kurtosis": 1.0,
    "pdf_tolerance": 1e-4
  },
  "output_path": "pdf_comparison_moments",
  "show_grid_points": true,
  "max_grid_points_display": 200
}
```

### 例4: 平均・分散を重視する設定

```json
{
  "mu_x": 0.1,
  "sigma_x": 0.4,
  "mu_y": 0.15,
  "sigma_y": 0.9,
  "rho": 0.9,
  "z_range": [-4, 4],
  "z_npoints": 128,
  "K": 5,
  "L": 3,
  "method": "lp",
  "objective_mode": "moments",
  "lp_params": {
    "solver": "highs",
    "sigma_min_scale": 0.5,
    "sigma_max_scale": 2.0,
    "lambda_mean": 10.0,
    "lambda_variance": 10.0,
    "lambda_skewness": 0.1,
    "lambda_kurtosis": 0.1,
    "pdf_tolerance": 1e-4
  },
  "output_path": "pdf_comparison_mean_var",
  "show_grid_points": true,
  "max_grid_points_display": 200
}
```

## 出力設定

```json
{
  "output_path": "pdf_comparison",
  "show_grid_points": true,
  "max_grid_points_display": 200
}
```

- `output_path`: 出力ファイルのパス（拡張子なし、PNG形式で保存）
- `show_grid_points`: グリッド点をプロットに表示するか
- `max_grid_points_display`: プロットに表示する最大グリッド点数

## 実行方法

```bash
# デフォルトの設定ファイルを使用
python main.py --config configs/config_default.json

# カスタム設定ファイルを指定
python main.py --config config_moments_example.json
```

## 注意事項

1. **JSON形式**: コメントは使用できません
2. **デフォルト値**: `objective_mode` を省略した場合、デフォルトは `"pdf"` です
3. **パラメータの互換性**: 
   - `method="em"` の場合、EM法のパラメータが必要
   - `method="lp"` の場合、LP法のパラメータが必要
   - `objective_mode` は `method="lp"` の場合のみ有効
4. **モーメントモードの推奨設定**:
   - `pdf_tolerance`: 1e-4 から 1e-3 程度が推奨（厳しすぎると解が見つからない場合があります）
   - 各 `lambda_*` パラメータでモーメントの重要度を調整可能

