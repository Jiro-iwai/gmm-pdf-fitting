# gmm-pdf-fitting

ガウス混合モデル（GMM）による確率密度関数（PDF）の近似ライブラリ

このプロジェクトは、2変量正規分布の最大値の確率密度関数（PDF）を、ガウス混合モデル（GMM）で近似する実装です。3つのフィッティング方式を提供します：

1. **重み付きEMアルゴリズム** (`method: "em"`): 対数尤度最大化による従来のEM方式
2. **LPアルゴリズム** (`method: "lp"`): L∞誤差最小化による線形計画法方式
3. **Hybrid法** (`method: "hybrid"`): LP→EM→QPの統合手法

## 概要

- **2変量正規分布の最大値PDF**: `Z = max(X, Y)` の確率密度関数を計算
- **3つのフィッティング方式**: EM方式、LP方式、Hybrid方式から選択可能
- **EM方式の特徴**:
  - 重み付きEMアルゴリズム: PDFグリッドに1次元GMMをフィット
  - 4つの初期化方式: quantile, random, QMI, WQMIから選択可能
  - モーメント一致QP投影: オプションで1〜4次モーメント（平均、分散、歪度、尖度）を一致させるQP投影を適用
- **LP方式の特徴**:
  - PDF誤差のL∞最小化（`objective_mode="pdf"`）
  - 生モーメント誤差最小化モード（`objective_mode="raw_moments"`）：完全線形の生モーメント（M1-M4）マッチング
  - 辞書ベースの基底関数生成
  - **注意**: `objective_mode="moments"`は実行速度の都合によりベンチマークでスキップされています
- **Hybrid方式の特徴**:
  - LP法で大規模辞書から初期値を取得
  - EM法で微調整
  - オプションでQP射影によるモーメント一致
- **可視化**: 真のPDF、GMM近似、各コンポーネントをプロット表示
- **数値安定性**: `logsumexp`を使用した対数空間での計算（EM方式）
- **見やすい出力**: セクション構造化された標準出力で結果を表示

## 必要な環境

- Python 3.8以上
- uv（パッケージマネージャー）

## セットアップ

### 1. 仮想環境の作成と依存パッケージのインストール

**方法1: Makefileを使用（推奨）**

```bash
# 依存パッケージをインストール（仮想環境も自動的に作成されます）
make install
```

**方法2: 手動でインストール**

```bash
# 仮想環境を作成
uv venv

# 仮想環境を有効化
source .venv/bin/activate

# 依存パッケージをインストール
uv pip install -r requirements.txt
```

または、uvを使って直接実行する場合：

```bash
# 仮想環境を作成（初回のみ）
uv venv

# 依存パッケージをインストール（初回のみ）
uv pip install -r requirements.txt
```

## 実行方法

### 方法1: JSON設定ファイルを使用（推奨）

```bash
# 仮想環境を有効化
source .venv/bin/activate

# デフォルトの設定ファイルを使用して実行
python main.py --config configs/config_default.json

# または、カスタム設定ファイルを指定
python main.py --config configs/config_lp.json
```

### 方法2: uv runを使用して実行

```bash
# デフォルトの設定ファイルを使用
uv run python main.py --config configs/config_default.json

# カスタム設定ファイルを指定（configs/ディレクトリから）
uv run python main.py --config configs/config_lp.json
```

### 方法3: 実行例スクリプトを使用

```bash
# PDF誤差最小化モードの例
python examples/example_pdf_mode.py

# モーメント誤差最小化モードの例
python examples/example_moments_mode.py
```

## プロジェクト構造

```
ssta/
├── src/
│   └── gmm_fitting/     # GMMフィッティングパッケージ
│       ├── __init__.py
│       ├── em_method.py      # EM法の実装
│       ├── lp_method.py      # LP法の実装
│       └── gmm_utils.py      # GMMユーティリティ関数
├── configs/              # 設定ファイル（JSON）
│   ├── config_example.json
│   ├── config_lp.json
│   └── config_moments_example.json
├── examples/            # 実行例スクリプト
│   ├── example_pdf_mode.py
│   └── example_moments_mode.py
├── benchmarks/          # ベンチマークスクリプト
│   ├── benchmark.py
│   ├── benchmark_hybrid.py
│   └── results/         # ベンチマーク結果
├── docs/                # ドキュメント
│   ├── CONFIG_GUIDE.md
│   ├── config_examples.md
│   ├── lp_method.md
│   └── ...
├── outputs/             # 生成された出力ファイル（PNG等）
├── tests/               # テストファイル
├── webapp/              # Webアプリケーション
│   ├── api.py
│   ├── models.py
│   └── frontend/
├── main.py              # メイン実行スクリプト
├── requirements.txt     # 依存パッケージ
├── Makefile             # ビルドスクリプト
└── pytest.ini          # pytest設定
```

## 設定ファイル（JSON）

パラメータはJSONファイルで指定できます。設定ファイルの例は`configs/`ディレクトリにあります。デフォルトは`configs/config_default.json`です：

### EM方式の設定例

```json
{
  "mu_x": 0.0,
  "sigma_x": 0.8,
  "mu_y": 0.0,
  "sigma_y": 1.6,
  "rho": 0.9,
  "z_range": [-6.0, 8.0],
  "z_npoints": 2500,
  "K": 3,
  "method": "em",
  "max_iter": 400,
  "tol": 1e-10,
  "reg_var": 1e-6,
  "n_init": 8,
  "seed": 1,
  "init": "quantile",
  "use_moment_matching": false,
  "qp_mode": "hard",
  "soft_lambda": 1e4,
  "output_path": "pdf_comparison"
}
```

### LP方式の設定例

#### PDF誤差最小化モード（デフォルト）

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

#### 生モーメント誤差最小化モード（推奨）

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
  "L": 10,
  "method": "lp",
  "objective_mode": "raw_moments",
  "lp_params": {
    "solver": "highs",
    "sigma_min_scale": 0.1,
    "sigma_max_scale": 3.0,
    "lambda_raw": [1.0, 1.0, 1.0, 1.0],
    "objective_form": "A",
    "pdf_tolerance": 0.01
  },
  "output_path": "pdf_comparison_lp",
  "show_grid_points": true,
  "max_grid_points_display": 200
}
```

**注意**: `objective_mode="moments"`は実行速度の都合によりベンチマークでスキップされています。代わりに`objective_mode="raw_moments"`を使用してください。

### パラメータ説明

#### 2変量正規分布のパラメータ

これらのパラメータは、元となる2変量正規分布 `(X, Y)` を定義し、その最大値 `Z = max(X, Y)` のPDFを計算するために使用されます。

- **`mu_x`** (float): Xの平均値
  - デフォルト: `0.0`
  - Xの分布の中心位置を決定します
  - 例: `0.0`, `1.5`, `-0.5`

- **`sigma_x`** (float): Xの標準偏差
  - デフォルト: `0.8`
  - Xの分布の広がりを決定します（必ず正の値）
  - 値が大きいほど分布が広がります
  - 例: `0.4`, `0.8`, `1.6`

- **`mu_y`** (float): Yの平均値
  - デフォルト: `0.0`
  - Yの分布の中心位置を決定します
  - 例: `0.0`, `-0.5`, `1.0`

- **`sigma_y`** (float): Yの標準偏差
  - デフォルト: `1.6`
  - Yの分布の広がりを決定します（必ず正の値）
  - 値が大きいほど分布が広がります
  - 例: `0.9`, `1.6`, `2.0`

- **`rho`** (float): XとYの相関係数
  - デフォルト: `0.9`
  - 範囲: `-1.0 ≤ rho ≤ 1.0`
  - `rho = 1`: 完全な正の相関（XとYが同じ方向に動く）
  - `rho = 0`: 無相関（XとYは独立）
  - `rho = -1`: 完全な負の相関（XとYが逆方向に動く）
  - 高い相関（例: `0.9`）では、最大値PDFが非ガウス的になりやすい
  - 例: `0.0`, `0.5`, `0.9`, `-0.3`

#### PDF計算のパラメータ

これらのパラメータは、PDFを計算するグリッドの設定を制御します。

- **`z_range`** (list): PDFを計算する範囲 `[z_min, z_max]`
  - デフォルト: `[-6.0, 8.0]`
  - PDFが十分に小さい値になる範囲を指定します
  - 範囲が狭すぎるとPDFの裾野が切り取られます
  - 範囲が広すぎると計算時間が増加します
  - 推奨: 平均±4〜6標準偏差程度
  - 例: `[-6, 6]`, `[-8, 8]`, `[-5, 10]`

- **`z_npoints`** (int): グリッド点数（目標点数）
  - デフォルト: `2500`
  - PDFを評価するグリッド点の数を指定します
  - **影響**:
    - 値が大きいほど精度が向上しますが、計算時間も増加します
    - EMアルゴリズムの計算時間はほぼ `z_npoints` に比例します
  - **推奨値**:
    - 低精度・高速: `256` 〜 `512`
    - 標準: `1000` 〜 `2500`
    - 高精度: `5000` 〜 `10000`
  - 例: `512`, `128`, `2500`, `5000`

#### GMMフィッティングのパラメータ

##### 共通パラメータ

- **`method`** (string): フィッティング方式の選択
  - デフォルト: `"em"`
  - **`"em"`**: 重み付きEMアルゴリズム（対数尤度最大化）
  - **`"lp"`**: LPアルゴリズム（L∞誤差最小化）
  - **`"hybrid"`**: Hybrid法（LP→EM→QPの統合手法）
  - 例: `"em"`, `"lp"`, `"hybrid"`

- **`K`** (int): GMMの成分数
  - デフォルト: `3`
  - ガウス分布の混合数を指定します
  - **影響**:
    - 値が小さい（例: `2`）: シンプルなモデルですが、複雑な分布を表現できない可能性があります
    - 値が大きい（例: `6`）: より複雑な分布を表現できますが、過学習のリスクがあります
  - 推奨: `3` 〜 `5`
  - 例: `2`, `3`, `5`, `6`

##### EM方式のパラメータ（`method: "em"`の場合）

これらのパラメータは、EMアルゴリズムによるGMMフィッティングの動作を制御します。

- **`K`** (int): GMMの成分数
  - デフォルト: `3`
  - ガウス分布の混合数を指定します
  - **影響**:
    - 値が小さい（例: `2`）: シンプルなモデルですが、複雑な分布を表現できない可能性があります
    - 値が大きい（例: `6`）: より複雑な分布を表現できますが、過学習のリスクがあります
  - 推奨: `3` 〜 `5`
  - 例: `2`, `3`, `5`, `6`

- **`max_iter`** (int): EMアルゴリズムの最大反復回数
  - デフォルト: `400`
  - 収束しない場合の最大反復回数を指定します
  - **影響**:
    - 値が小さい: 収束前に終了する可能性があります
    - 値が大きい: 収束まで時間がかかる可能性がありますが、より良い結果が得られる可能性があります
  - 推奨: `500` 〜 `10000`
  - 例: `400`, `1000`, `10000`

- **`tol`** (float): 収束判定の許容誤差
  - デフォルト: `1e-10`
  - 前回と現在の対数尤度の差がこの値より小さくなったら収束と判定します
  - **影響**:
    - 値が小さい（例: `1e-10`）: より厳密な収束条件ですが、反復回数が増加します
    - 値が大きい（例: `1e-4`）: 早く収束しますが、精度が低下する可能性があります
  - 推奨: `1e-6` 〜 `1e-10`
  - 例: `1e-4`, `1e-6`, `1e-7`, `1e-10`

- **`reg_var`** (float): 分散の正則化項
  - デフォルト: `1e-6`
  - 分散の最小値を設定し、0や非常に小さな値になることを防ぎます
  - **役割**:
    - 数値安定性の確保（0除算やオーバーフローの防止）
    - 特異性の回避（分散が0になることの防止）
  - **影響**:
    - 値が小さすぎる（例: `1e-10`）: 数値誤差の影響を受けやすい
    - 値が大きすぎる（例: `1e-4`）: 分散の推定が不正確になる可能性があります
  - 推奨: `1e-6` 〜 `1e-8`
  - 例: `1e-6`, `1e-7`, `1e-8`

- **`n_init`** (int): 初期化の試行回数
  - デフォルト: `8`
  - 異なる初期値からEMアルゴリズムを実行し、最良の結果を選択します
  - **影響**:
    - 値が小さい（例: `3`）: 計算時間が短縮されますが、局所最適解に陥る可能性があります
    - 値が大きい（例: `20`）: より良い結果が得られる可能性がありますが、計算時間が増加します
  - 推奨: `5` 〜 `10`
  - 例: `5`, `8`, `10`, `20`

- **`seed`** (int): 乱数シード
  - デフォルト: `1`
  - 乱数生成のシード値を指定します
  - 同じシード値を使用すると、同じ結果が得られます（再現性の確保）
  - 異なるシード値で試行することで、結果の安定性を確認できます
  - 例: `1`, `42`, `123`

- **`init`** (string): 初期化方法
  - デフォルト: `"quantile"`
  - **`"quantile"`**: 分位点ベースの初期化（シンプル）
    - PDFの分位点で平均を初期化し、混合重みと分散は均等に設定
    - 安定した結果が得られやすい
  - **`"random"`**: ランダム初期化
    - グリッド点からランダムに平均を選択
    - 複数の初期化を試行することで局所最適を回避
  - **`"qmi"`**: Quantile-based Moment Initialization（分位点ビン局所モーメント初期化）
    - PDFをK個の分位点で分割し、各ビンで局所モーメント（質量、平均、分散）を計算
    - PDFの形状により適した初期値が得られる
    - `initial_guess_spec.md`の方法1に基づく
  - **`"wqmi"`**: Winner-decomposed QMI（勝者分解 + 分位点ビン局所モーメント初期化）
    - MAX(X,Y)のPDFをXが勝つ部分（g_X）とYが勝つ部分（g_Y）に分解
    - 各寄与に対してQMIを適用し、より少ないKでkink/歪みを捉えやすい
    - `initial_guess_spec.md`の方法2に基づく
    - MAX(X,Y)の構造を初期値に反映するため、特に有効
  - 例: `"quantile"`, `"random"`, `"qmi"`, `"wqmi"`

- **`use_moment_matching`** (boolean): モーメント一致QP投影を使用するか
  - デフォルト: `false`
  - **`true`**: EM学習の最後に1回だけ、混合重み（π）をQPで更新して、真のPDFと1〜4次中心モーメント（平均、分散、歪度、尖度）を一致させる
    - モーメントの一致により統計量の誤差が小さくなる可能性があります
    - ただし、形状が若干崩れる可能性があります（モーメント一致と形状一致はトレードオフ）
    - ⚠️ **重要**: `K < 5` の場合、自由度不足により尖度などの高次モーメントで大きな誤差が発生する可能性があります。正確なモーメント一致には `K ≥ 5` を推奨します。
  - **`false`**: 通常のEM学習のみ（モーメント一致の制約なし）
  - **注意**: K≥5を推奨（等式制約が5本あるため）
  - 例: `true`, `false`

- **`qp_mode`** (string): QP投影のモード（`use_moment_matching=true`の場合のみ有効）
  - デフォルト: `"hard"`
  - **`"hard"`**: ハード制約QPを試行し、失敗時はソフト制約にフォールバック
    - 成功時はモーメントを厳密に一致させます
  - **`"soft"`**: ソフト制約QPを直接使用
    - 常に解けますが、モーメントは近似的に一致します
  - 例: `"hard"`, `"soft"`

- **`soft_lambda`** (float): ソフト制約の罰則係数（`use_moment_matching=true`の場合のみ有効）
  - デフォルト: `1e4`
  - 値が大きいほどモーメント一致が厳密になりますが、最適化が難しくなります
  - 推奨: `1e3` 〜 `1e6`
  - 例: `1e3`, `1e4`, `1e5`

##### LP方式のパラメータ（`method: "lp"`の場合）

これらのパラメータは、LPアルゴリズムによるGMMフィッティングの動作を制御します。

- **`K`** (int): セグメント数（平均位置の候補数）
  - デフォルト: `10`
  - 値が大きいほど多様な平均位置を試せますが、計算時間が増加します
  - 推奨: `5` 〜 `20`

- **`L`** (int): シグマレベル数（各セグメントあたりの標準偏差候補数）
  - デフォルト: `5`
  - 値が大きいほど多様な標準偏差を試せますが、計算時間が増加します
  - 推奨: `3` 〜 `10`

- **`objective_mode`** (string): 目的関数モード
  - デフォルト: `"pdf"`
  - **`"pdf"`**: PDF誤差のみを最小化（L∞ノルム）
  - **`"raw_moments"`**: 生モーメント誤差（M1-M4）を最小化（PDF誤差を制約として使用、完全線形LP）
  - **`"moments"`**: モーメント誤差を最小化（PDF誤差を制約として使用、反復LP）
    - **注意**: 実行速度の都合によりベンチマークでスキップされています
  - 例: `"pdf"`, `"raw_moments"`, `"moments"`

- **`lp_params`** (dict): LPソルバーパラメータ
  - **`solver`** (string): LPソルバー
    - デフォルト: `"highs"`
    - 利用可能なソルバー: `"highs"`, `"interior-point"`, `"revised simplex"`
    - 推奨: `"highs"`（高速で安定）
  - **`sigma_min_scale`** (float): 最小標準偏差のスケール（真の標準偏差に対する比率）
    - デフォルト: `0.1`
    - 例: `0.05`, `0.1`, `0.2`
  - **`sigma_max_scale`** (float): 最大標準偏差のスケール（真の標準偏差に対する比率）
    - デフォルト: `3.0`
    - 例: `2.0`, `3.0`, `5.0`
  
  **`objective_mode="raw_moments"`の場合の追加パラメータ:**
  - **`lambda_raw`** (list[float]): 生モーメント誤差の重み [λ_1, λ_2, λ_3, λ_4]
    - デフォルト: `[1.0, 1.0, 1.0, 1.0]`
    - M1（平均）、M2、M3、M4の誤差に対する重み
    - 例: `[1.0, 1.0, 1.0, 1.0]`, `[2.0, 1.0, 1.0, 1.0]`
  - **`objective_form`** (string): 目的関数の形式
    - デフォルト: `"A"`
    - **`"A"`**: PDF誤差を制約として使用し、生モーメント誤差を最小化
    - **`"B"`**: PDF誤差と生モーメント誤差の重み付き和を最小化
  - **`pdf_tolerance`** (float): PDF誤差の上限（`objective_form="A"`の場合の制約として使用）
    - デフォルト: `None`
    - 値が大きいほど制約が緩くなり、モーメント誤差を小さくできます
    - 推奨: `0.01` 〜 `0.05`

  **`objective_mode="moments"`の場合の追加パラメータ:**
  - **注意**: このモードは実行速度の都合によりベンチマークでスキップされています
  - **`lambda_mean`** (float): 平均誤差の重み
  - **`lambda_variance`** (float): 分散誤差の重み
  - **`lambda_skewness`** (float): 歪度誤差の重み
  - **`lambda_kurtosis`** (float): 尖度誤差の重み
  - **`pdf_tolerance`** (float): PDF誤差の上限
  - **`max_moment_iter`** (int): モーメント制約の反復回数上限
  - **`moment_tolerance`** (float): モーメント制約の収束判定許容誤差

##### Hybrid方式のパラメータ（`method: "hybrid"`の場合）

Hybrid法は、LP→EM→QPの3ステップで構成される統合的なGMMフィッティング手法です。

- **`K`** (int): 最終的なGMM成分数
  - デフォルト: `3`
  - LPステップで選択された上位K成分がEMステップの初期値として使用されます

- **`L`** (int): 辞書の標準偏差レベル数（`dict_L`のデフォルト値として使用）
  - デフォルト: `10`

- **`lp_params`** (dict): LPステップのパラメータ
  - **`dict_J`** (int, デフォルト: `4 * K`): 辞書の平均値（μ）の数
  - **`dict_L`** (int, デフォルト: `L`): 辞書の標準偏差（σ）のレベル数
  - **`mu_mode`** (str, デフォルト: `"quantile"`): 平均値の生成方法（`"quantile"` または `"uniform"`）
  - **`tail_focus`** (str, デフォルト: `"none"`): 裾に焦点を当てた辞書生成（`"none"`, `"right"`, `"left"`, `"both"`）
  - **`tail_alpha`** (float, デフォルト: `1.0`): 裾焦点の強度
  - **`objective_mode`** (str, デフォルト: `"raw_moments"`): LPの目的関数モード（Hybrid法では`"raw_moments"`のみサポート）
  - **`pdf_tolerance`** (float): PDF誤差の上限
  - **`lambda_raw`** (list[float]): 生モーメント誤差の重み [λ_1, λ_2, λ_3, λ_4]
  - **`objective_form`** (str): 目的関数の形式（`"A"` または `"B"`）
  - その他のLPパラメータ（`solver`, `sigma_min_scale`, `sigma_max_scale`など）も使用可能

- **`init`** (string): EMステップの初期化方法
  - デフォルト: `"custom"`（Hybrid法ではLPの結果を初期値として使用）
  - Hybrid法では通常`"custom"`を使用します

- **`use_moment_matching`** (boolean): QPステップ（モーメント一致）を使用するか
  - デフォルト: `false`
  - `true`の場合、EMステップ後にQP投影でモーメントを一致させます

- **`qp_mode`** (string): QP投影のモード（`use_moment_matching=true`の場合のみ有効）
  - デフォルト: `"hard"`
  - **`"hard"`**: ハード制約QPを試行し、失敗時はソフト制約にフォールバック
  - **`"soft"`**: ソフト制約QPを直接使用

- **`soft_lambda`** (float): ソフト制約の罰則係数（`use_moment_matching=true`の場合のみ有効）
  - デフォルト: `1e4`

詳細は `docs/HYBRID_METHOD_GUIDE.md` を参照してください。

#### プロット出力のパラメータ

- **`output_path`** (string): 出力PNGファイルのパス（拡張子なし）
  - デフォルト: `"pdf_comparison"`
  - 出力ファイル名を指定します（拡張子 `.png` は自動的に追加されます）
  - `{output_path}.png` として1つのPNGファイルが生成されます
  - ファイルには線形スケールと対数スケールの2つのサブプロットが含まれます
  - 例: `"pdf_comparison"`, `"result"`, `"output/my_result"`

- **`show_grid_points`** (boolean): グリッド点をプロットに表示するか
  - デフォルト: `true`
  - `true`: グリッド点を青い点として表示します
  - `false`: グリッド点を表示しません
  - グリッド点が多い場合は自動的に間引いて表示されます

- **`max_grid_points_display`** (int): プロットに表示するグリッド点の最大数
  - デフォルト: `200`
  - グリッド点数がこの値を超える場合、自動的に間引いて表示されます
  - 値が大きすぎるとプロットが重くなります
  - 例: `100`, `200`, `500`

設定ファイルが存在しない場合、デフォルト値が使用されます。

## 実行例

コードを実行すると、以下のような出力が得られます：

```
Configuration file: configs/config_default.json
Parameters: mu_x=0.1, sigma_x=0.4, mu_y=0.15, sigma_y=0.9, rho=0.9
Using uniform grid with 32 points

======================================================================
EM ALGORITHM RESULTS
======================================================================
Best weighted log-likelihood: -0.9647953395
Iterations: 12 / 20000
Convergence: Yes

----------------------------------------------------------------------
EXECUTION TIME
----------------------------------------------------------------------
EM algorithm:          0.003823 seconds
QP projection:          0.001388 seconds
Total (EM + QP):        0.003824 seconds

----------------------------------------------------------------------
MOMENT MATCHING QP PROJECTION
----------------------------------------------------------------------
Method: SOFT (moments approximately matched)
Constraint error: 9.427060e-04
Moment errors:
  M0: +0.000000e+00
  M1: -4.340517e-04
  M2: +7.852583e-04
  M3: +1.749318e-04
  M4: -2.303518e-04

======================================================================
PDF STATISTICS COMPARISON
======================================================================
Statistic                 True PDF     GMM Approx PDF        Rel Error (%)
-----------------------------------------------------------------------
Mean                      0.352219           0.351801             -0.1187%
Std Dev                   0.683920           0.676609             -1.0689%
Skewness                  0.735281           0.759378              3.2773%
Kurtosis                  0.459016           0.471499              2.7195%

======================================================================
GMM PARAMETERS
======================================================================
Number of components: 4

Component details:
  Component 1: π=0.13630585, μ=-0.12322876, σ=0.30780651
  Component 2: π=0.41672955, μ=0.04488131, σ=0.41310724
  Component 3: π=0.17334335, μ=0.37301551, σ=0.54426462
  Component 4: π=0.27362125, μ=1.04246131, σ=0.66725729

======================================================================
PLOT OUTPUT
======================================================================
Plot saved: pdf_comparison.png
======================================================================
```

これは、2変量正規分布の最大値PDFを4成分のGMMで近似し、モーメント一致QP投影を適用した結果です。

**出力の説明**:
- **Configuration file**: 使用した設定ファイル名
- **Parameters**: 2変量正規分布のパラメータ
- **Using uniform grid with N points**: 使用した均等グリッドの点数
- **EM ALGORITHM RESULTS**: EMアルゴリズムの結果
  - **Best weighted log-likelihood**: 最良の対数尤度
  - **Iterations**: 実際の反復回数 / 最大反復回数
  - **Convergence**: 収束状況（Yes/No）
- **EXECUTION TIME**: 実行時間のサマリー
  - **EM algorithm**: EMアルゴリズムの実行時間（秒）
  - **QP projection**: QP投影の実行時間（秒、モーメント一致が有効な場合のみ）
  - **Total (EM + QP)**: 合計実行時間（秒、モーメント一致が有効な場合）またはEMのみの時間（無効な場合）
- **MOMENT MATCHING QP PROJECTION**: モーメント一致が有効な場合に表示
  - **Method**: `HARD`（ハード制約成功、モーメント厳密一致）または`SOFT`（ソフト制約使用、モーメント近似一致）
  - **Constraint error**: モーメント制約の誤差
  - **Moment errors**: 各モーメント（M0〜M4）の誤差（符号付き）
- **PDF STATISTICS COMPARISON**: 真のPDFとGMM近似PDFの統計量比較（相対誤差を%で表示）
- **GMM PARAMETERS**: 推定されたGMMの各成分のパラメータ
  - **Number of components**: 成分数
  - **Component details**: 各成分の混合重み（π）、平均（μ）、標準偏差（σ）
- **PLOT OUTPUT**: プロットファイルの保存情報

## プロット出力

### コマンドライン実行時のプロット

`main.py`で実行すると、1つのPNGファイル `{output_path}.png` が生成されます。このファイルには以下の2つのサブプロットが含まれます：

- **上部**: 縦軸が線形スケールのプロット
- **下部**: 縦軸が対数スケールのプロット

各プロットには以下が表示されます：
- **青い実線**: 真のPDF（2変量正規分布の最大値）
- **赤い破線**: GMM近似PDF（全コンポーネントの合計）
- **点線（色分け）**: 各GMMコンポーネント（個別の正規分布）
  - 各コンポーネントは `π_k * N(z; μ_k, σ²_k)` として表示されます
  - ラベルには混合重み（π）が表示されます
  - 色はコンポーネントごとに異なります
- **青い点**: グリッド点（`show_grid_points=true`の場合）
  - PDFが評価されたグリッド点を表示します
- パラメータ情報と対数尤度は全体のタイトルに表示されます

### Webアプリケーションのプロット

Webアプリケーションでは、Plotly.jsによるインタラクティブなプロットが表示されます：

- **動的なスケール切り替え**: 線形/対数スケールをUIから切り替え可能
- **カスタマイズ可能な設定**:
  - X軸、Y軸の範囲指定（Min/Max入力フィールド）
  - True PDFとGMM近似の色設定
  - 線のスタイル（実線、破線、点線など）と太さ
  - グリッドポイントの表示/非表示とサイズ
- **インタラクティブ機能**: ズーム、パン、ホバーで値を確認
- **設定の永続化**: プロット設定はCompute実行後も保持されます

## 性能評価（ベンチマーク）

包括的な性能評価を実施するには、`benchmark.py`スクリプトを使用します。

### 基本的な使用方法

```bash
# EM法のベンチマーク
python benchmarks/benchmark.py --config configs/config_default.json --output benchmarks/results/benchmark_em.json

# LP法のベンチマーク（プロットも生成）
python benchmarks/benchmark.py --config configs/config_lp.json --output benchmarks/results/benchmark_lp.json --plot
```

### 評価項目

ベンチマークスクリプトは以下の項目を評価します：

1. **実行時間**: 各手法の実行時間を測定
2. **PDF近似精度**: 
   - L∞誤差（最大絶対誤差）
   - L2誤差（二乗平均誤差）
3. **モーメント精度**: 
   - 平均、分散、歪度、尖度の誤差（絶対値と相対値）
4. **スケーラビリティ**: 
   - K（成分数）に対する性能
   - グリッド解像度に対する性能
   - L（辞書サイズ）に対する性能（LP法のみ）

### 出力

- **JSONファイル**: 詳細なベンチマーク結果（`--output`で指定）
- **プロット**: 6つのサブプロットを含む可視化（`--plot`オプション使用時）
  - 実行時間 vs K
  - PDF L∞誤差 vs K
  - 尖度誤差 vs K
  - 実行時間 vs グリッド解像度
  - PDF L2誤差 vs グリッド解像度
  - 実行時間 vs 精度のトレードオフ

### ベンチマーク設定

ベンチマークスクリプトは、設定ファイルに基づいて以下のパラメータ範囲で評価を実行します：

- **K値**: [3, 4, 5, 10, 15, 20]（全手法で統一）
- **グリッド解像度**: 
  - EM法・Hybrid法: [8, 16, 32, 64, 128, 256, 512]ポイント
  - LP法: [32, 64, 128]ポイント（最適化された設定）
- **L値**（LP法のみ）: [10]（最適化された設定）

## 依存パッケージ

### コアライブラリ（`requirements.txt`）

- `numpy>=2.0.0`: 数値計算ライブラリ
- `scipy>=1.0.0`: 科学計算ライブラリ（`scipy.special.ndtr`, `scipy.special.logsumexp`を使用）
- `matplotlib>=3.5.0`: プロット作成ライブラリ

### Webアプリケーション（`webapp/requirements.txt`）

- `fastapi>=0.100.0`: Web APIフレームワーク
- `uvicorn>=0.23.0`: ASGIサーバー
- `pydantic>=2.0.0`: データバリデーション
- その他の依存パッケージは`webapp/requirements.txt`を参照してください

### フロントエンド（`webapp/frontend/package.json`）

- `react>=18.2.0`: Reactフレームワーク
- `@mui/material>=5.14.0`: Material-UIコンポーネントライブラリ
- `plotly.js>=2.35.3`: インタラクティブプロットライブラリ
- `react-plotly.js>=2.6.0`: React用Plotlyラッパー
- その他の依存パッケージは`webapp/frontend/package.json`を参照してください

## ファイル構成

このプロジェクトのファイル構成：

- **`main.py`**: メイン実行スクリプト
  - 設定ファイルの読み込み
  - PDF計算とGMMフィッティングの実行
  - 結果の表示とプロット生成
  
- **`src/gmm_fitting/em_method.py`**: EM方式の実装
  - PDF計算関数（`max_pdf_bivariate_normal`, `normalize_pdf_on_grid`）
  - GMMフィッティング関数（`fit_gmm1d_to_pdf_weighted_em`）
  - 初期化関数（`_init_gmm_qmi`, `_init_gmm_wqmi`）
  - モーメント一致QP投影関数（`_project_moments_qp`）
  - 統計量計算関数（`compute_pdf_statistics`）
  - プロット関数（`plot_pdf_comparison`）
  - 出力フォーマット関数（各種`print_*`関数）
  
- **`src/gmm_fitting/lp_method.py`**: LP方式の実装
  - 辞書生成関数（`build_gaussian_dictionary_simple`）
  - 基底行列計算関数（`compute_basis_matrices`）
  - CDF計算関数（`pdf_to_cdf_trapz`）
  - LP解法関数（`solve_lp_pdf_linf`, `solve_lp_pdf_rawmoments_linf`）
  - メイン関数（`fit_gmm_lp_simple`）

- **`src/gmm_fitting/gmm_utils.py`**: GMMユーティリティ関数
  - モーメント計算関数（`compute_pdf_raw_moments`, `compute_gmm_moments_from_weights`）
  - 誤差計算関数（`compute_errors`）

- **`examples/`**: 実行例スクリプト
  - `example_pdf_mode.py`: PDF誤差最小化モードの例
  - `example_moments_mode.py`: モーメント誤差最小化モードの例

- **`benchmarks/`**: ベンチマークスクリプト
  - `benchmark.py`: 包括的な性能評価スクリプト
  - `benchmark_hybrid.py`: Hybrid法のベンチマーク実装

- **`configs/`**: 設定ファイルの例
  - `config_example.json`: 基本的な設定例
  - `config_lp.json`: LP法の設定例
  - `config_hybrid.json`: Hybrid法の設定例
  - `config_moments_example.json`: モーメントモードの設定例

- **`docs/`**: ドキュメント
  - `CONFIG_GUIDE.md`: 設定ファイルの詳細ガイド
  - `config_examples.md`: 設定例の説明
  - `lp_method.md`: LP法の実装仕様書
  - `moment_em.md`: モーメントマッチングの説明
  - `initial_guess_spec.md`: 初期化方法の仕様

- **`tests/`**: 単体テスト
  - `test_pdf_calculation.py`: PDF計算のテスト
  - `test_gmm_fitting.py`: GMMフィッティングのテスト
  - `test_lp_method.py`: LP方式のテスト
  - `test_moments.py`: モーメント計算のテスト
  - `test_statistics.py`: 統計量計算のテスト
  - `test_config.py`: 設定読み込みのテスト
  - `test_output_formatting.py`: 出力フォーマットのテスト

## コードの構成

- `max_pdf_bivariate_normal()`: 2変量正規分布の最大値PDFを計算
- `max_pdf_bivariate_normal_decomposed()`: MAX(X,Y)のPDFをg_Xとg_Yに分解（WQMI用）
- `normalize_pdf_on_grid()`: PDFを正規化（積分=1）
- `_init_gmm_qmi()`: QMI方式によるGMM初期化（分位点ビン局所モーメント）
- `_init_gmm_wqmi()`: WQMI方式によるGMM初期化（勝者分解 + QMI）
- `fit_gmm1d_to_pdf_weighted_em()`: 重み付きEMアルゴリズムでGMMをフィット（オプションでモーメント一致QP投影）
- `_compute_central_moments()`: 重み付きグリッドから中心モーメント（1〜4次）を計算
- `_central_to_raw_moments()`: 中心モーメントをrawモーメント（0〜4次）に変換
- `_compute_component_raw_moments()`: GMM成分のrawモーメントを計算
- `_project_moments_qp()`: QP投影で混合重みをモーメント一致させる
- `gmm1d_pdf()`: 推定されたGMMパラメータでPDFを評価
- `compute_pdf_statistics()`: PDFから統計量（平均、標準偏差、歪度、尖度）を計算
- `plot_pdf_comparison()`: PDF比較プロットをPNGファイルとして出力（線形・対数スケール、各コンポーネントも表示）

## テスト

このプロジェクトにはpytestを使用した単体テストが含まれています。

### テストの実行

```bash
# すべてのテストを実行
make test

# 詳細な出力で実行
make test-verbose

# カバレッジレポート付きで実行
make test-cov

# HTMLカバレッジレポートを生成
make test-cov-html

# 高速モードで実行（カバレッジなし）
make test-fast

# 特定のテストファイルを実行
make test-specific FILE=tests/test_pdf_calculation.py
```

### その他のMakefileコマンド

```bash
# 生成ファイルのクリーンアップ
make clean

# ヘルプメッセージの表示
make help
```

詳細は`docs/TESTING.md`を参照してください。

## Webアプリケーション（FastAPI + React）

GUIでパラメータを指定し、結果を可視化できるWebアプリケーションを提供しています。

### 主な機能

- **3つのフィッティング方式**: EM、LP、Hybrid方式を選択可能
- **インタラクティブなプロット**: Plotly.jsによる動的なプロット表示
  - 線形/対数スケールの切り替え
  - プロット範囲のカスタマイズ（X軸、Y軸）
  - 色、線のスタイル、線の太さの調整
  - グリッドポイントの表示/非表示
- **統計比較テーブル**: 相対誤差に応じた色分け表示（誤差が小さい=青、普通=黄、大きい=赤）
- **ダークモード**: ライト/ダークモードの切り替え（設定は自動保存）
- **設定のエクスポート/インポート**: 現在のパラメータをJSONファイルとして保存・読み込み可能
- **リアルタイム計算**: パラメータ変更後すぐに結果を確認

### セットアップ

詳細は `webapp/README.md` を参照してください。

**クイックスタート（Makefile使用）:**

```bash
# 1. 依存パッケージのインストール（初回のみ）
make webapp-install

# 2. Webアプリの起動
make webapp-start

# 3. ブラウザで http://localhost:3000 にアクセス

# 4. Webアプリの停止
make webapp-stop

# その他のコマンド
make webapp-status    # ステータス確認
make webapp-logs      # ログ表示
make webapp-backend   # バックエンドのみ起動
make webapp-frontend  # フロントエンドのみ起動
make webapp-clean     # ログ・PIDファイルのクリーンアップ
```
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>
read_file

**手動起動:**

```bash
# 1. バックエンド依存パッケージのインストール
pip install -r webapp/requirements.txt

# 2. フロントエンド依存パッケージのインストール
cd webapp/frontend
npm install

# 3. バックエンドサーバーの起動（別ターミナル）
python -m webapp.api

# 4. フロントエンド開発サーバーの起動（別ターミナル）
cd webapp/frontend
npm run dev
```

詳細は `webapp/README.md` を参照してください。

## 注意事項

- コードは`np.trapezoid`を使用しており、非推奨警告は表示されません

## ライセンス

このプロジェクトのライセンス情報については、LICENSEファイルを参照してください。

## コントリビューション

プルリクエストやイシューの報告を歓迎します。詳細はCONTRIBUTING.mdを参照してください。

