# 設定ファイルの書き方ガイド

このドキュメントでは、設定ファイル（JSON形式）の書き方と各パラメータの説明を提供します。

## 基本構造

設定ファイルはJSON形式です。コメントは使用できませんが、このガイドでは説明のため `//` でコメントを表記しています。設定ファイルは`configs/`ディレクトリに配置します。

## パラメータ一覧

### 1. 2変量正規分布のパラメータ（必須）

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

### 2. グリッド設定（必須）

PDFを計算するグリッドの範囲と点数を指定します。

```json
{
  "z_range": [-4, 4],    // グリッドの範囲 [最小値, 最大値]
  "z_npoints": 128       // グリッドの点数
}
```

### 3. GMMの成分数（必須）

```json
{
  "K": 4  // GMMの成分数（1以上の整数）
}
```

### 4. 手法選択（オプション）

```json
{
  "method": "lp"  // "em" または "lp"（デフォルト: "em"）
}
```

### 5. EM法のパラメータ（method="em" の場合）

```json
{
  "max_iter": 20000,           // 最大反復回数
  "tol": 1e-5,                 // 収束判定の許容誤差
  "reg_var": 1e-06,            // 分散の正則化パラメータ
  "n_init": 4,                 // 初期値の試行回数
  "seed": 1,                   // 乱数シード
  "init": "quantile",          // 初期化方法: "quantile" または "wqmi"
  "use_moment_matching": true, // モーメントマッチングを使用するか
  "qp_mode": "hard",           // QPモード: "hard" または "soft"
  "soft_lambda": 1e4           // ソフト制約のペナルティ係数
}
```

### 6. LP法のパラメータ（method="lp" の場合）

#### 6.1 基本パラメータ

```json
{
  "L": 5  // fit_gmm_lp_simple で使用するシグマレベル数
}
```

#### 6.2 LPソルバーパラメータ

```json
{
  "lp_params": {
    "solver": "highs",          // LPソルバー: "highs"（推奨）
    "sigma_min_scale": 0.1,     // 最小シグマスケール（真の標準偏差に対する比率）
    "sigma_max_scale": 3.0,     // 最大シグマスケール（真の標準偏差に対する比率）
    
    // モーメント誤差最小化モード（objective_mode="moments" の場合のみ使用）
    "lambda_mean": 1.0,         // 平均誤差の重み（デフォルト: 1.0）
    "lambda_variance": 1.0,     // 分散誤差の重み（デフォルト: 1.0）
    "lambda_skewness": 1.0,     // 歪度誤差の重み（デフォルト: 1.0）
    "lambda_kurtosis": 1.0,     // 尖度誤差の重み（デフォルト: 1.0）
    "pdf_tolerance": 0.04,      // PDF誤差の上限（制約、デフォルト: 1e-6、推奨: 0.01〜0.05）
    "max_moment_iter": 10,      // モーメント制約の反復回数上限（デフォルト: 5）
    "moment_tolerance": 1e-6    // モーメント制約の収束判定許容誤差（デフォルト: 1e-6）
  }
}
```

#### 6.5 目的関数モード（LP法のみ）

```json
{
  "objective_mode": "pdf"  // "pdf" または "moments"（デフォルト: "pdf"）
}
```

- **`"pdf"`**: PDF誤差のみを最小化（デフォルトモード）
- **`"moments"`**: モーメント（平均、分散、歪度、尖度）の相対誤差を最小化（PDF誤差は制約）

### 7. 出力設定（オプション）

```json
{
  "output_path": "pdf_comparison_lp",  // 出力ファイルのパス（拡張子なし）
  "show_grid_points": true,            // グリッド点をプロットに表示するか
  "max_grid_points_display": 200      // プロットに表示する最大グリッド点数
}
```

## 設定例

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


## 注意事項

1. **JSON形式**: コメントは使用できません。このガイドの `//` は説明のためのものです。
2. **デフォルト値**: 多くのパラメータにはデフォルト値がありますが、必須パラメータは明示的に指定してください。
3. **モードの選択**: 
   - `method="em"` の場合、EM法のパラメータが必要です
   - `method="lp"` の場合、LP法のパラメータが必要です
   - `objective_mode` は `method="lp"` の場合のみ有効です
4. **LP法の使い方**:
   - `K` と `L` を指定し、`lp_params` に `solver`, `sigma_min_scale`, `sigma_max_scale` を含める
   - `objective_mode="moments"` の場合、`lp_params` にモーメント関連のパラメータ（`lambda_mean`, `lambda_variance`, `lambda_skewness`, `lambda_kurtosis`, `pdf_tolerance`, `max_moment_iter`, `moment_tolerance`）を追加

