# 設計書：MAX 後 PDF 近似の改良実装（LP 生モーメント + テール重視辞書 + Hybrid：LP→EM→QP + 数値安定化 + 評価拡張）

## 1. 目的

添付論文 **2401.03588v1** が述べる「非ガウス PDF をガウス核（Gaussian kernel）の線形結合で表し、最適化で係数（重み）を求める」枠組みに沿って、現状実装（EM / LP）を次の方向で改良する。

1. **LP のモーメント一致を“完全に線形”にする**（反復線形化を廃止）
2. **辞書（Gaussian basis）の μ 配置を“分位点（quantile）＋右裾（right tail）重視”**にする
3. **Hybrid 法（LPで成分選択→EMで微調整→QPでモーメント保証）**を追加する
4. **相関係数 ρ→±1 近傍の数値安定性**を改善する
5. **評価指標（CDF誤差・分位点誤差・右裾重み誤差）**を追加し、レポート更新を容易にする

---

## 2. 参照（2401.03588v1 の要点）

### 2.1 LP（L∞最小化）の基本形

論文では、RBF（ここではガウス核）を用いた近似
[
\hat f(x_i)=\sum_{j=1}^{m} w_j ,\phi_j(x_i)
]
に対し、残差 (r_i=\hat f(x_i)-y_i) の **無限ノルム**
[
|Mx-y|_\infty = \max_i |r_i|
]
を最小化する **minimax（Chebyshev）型**を LP に落とす形を提示している（式 (35)〜(40)）。

本実装も同様に、グリッド点 (z_i) 上で

* 目的：(\min t_{\rm pdf})
* 制約：(|\Phi w - f|\le t_{\rm pdf})

を LP で解く。

---

## 3. 用語定義（この設計書内で使用）

* **PDF**（probability density function）：確率密度関数 (f(z))
* **CDF**（cumulative distribution function）：累積分布関数
  [
  F(z)=\int_{-\infty}^{z} f(u),du
  ]
* **分位点（quantile）**：(p\in(0,1)) に対し
  [
  q_p = F^{-1}(p)
  ]
* **生モーメント（raw moment）**：
  [
  M_n=\mathbb{E}[Z^n]=\int_{-\infty}^{\infty} z^n f(z),dz
  ]
* **中心モーメント（central moment）**：(\mu=\mathbb{E}[Z]) として
  [
  \mu_n=\mathbb{E}[(Z-\mu)^n]
  ]
* **右裾（right tail）**：分布の「大きい (z) 側」。SSTA では遅延の大きい側が重要になりやすい。
* **辞書（dictionary）**：ガウス核 (\phi_j(z)=\mathcal{N}(z;\mu_j,\sigma_j^2)) の集合（候補基底の集合）
* **GMM**：ガウス混合モデル
  [
  \hat f(z)=\sum_{k=1}^{K}\pi_k,\mathcal N(z;\mu_k,\sigma_k^2),\quad
  \pi_k\ge 0,\quad \sum_{k=1}^{K}\pi_k=1
  ]

---

## 4. 変更対象ファイル

* `gmm_utils.py`（存在しない場合は新規作成。存在する場合は機能追加）
* `lp_method.py`
* `em_method.py`
* `main.py`
* `method_recommendations.md`（評価表・推奨の更新）

---

## 5. 実装方針（全体構成）

### 5.1 新しい LP 目的（objective_mode="raw_moments"）

従来の「歪度・尖度（中心化して標準化）を LP に入れる」方式は非線形で、反復線形化が必要になり不安定になりやすい。
そこで **生モーメント (M_1,M_2,M_3,M_4)** を使い、LP 制約を **完全に線形**にする。

### 5.2 新しい辞書生成（μ のテール重視分位点）

既存の `build_gaussian_dictionary`（quantile）がある場合は拡張し、分位点レベル (p_j) を **右裾寄せ**できるようにする。

### 5.3 新しい Hybrid 法（method="hybrid"）

1. 大きい辞書で LP を解く（PDF＋生モーメント）
2. 重み上位 (K) 個を選んで GMM 初期値にする
3. EM で微調整
4. 必要なら QP（既存の moment matching）で生モーメント一致を保証

---

## 6. 詳細設計

## 6.1 `gmm_utils.py`（新規 or 追記）

### 6.1.1 必須関数（不足していれば実装）

#### (A) `compute_pdf_raw_moments`

**目的**：グリッド上の PDF から生モーメント (M_0..M_4) を数値積分で得る。

* (M_0=1) を保証する（入力 PDF が正規化されていない場合は正規化してから計算）

**シグネチャ**

```python
def compute_pdf_raw_moments(z: np.ndarray, f: np.ndarray, max_order: int = 4) -> np.ndarray:
    """
    Returns moments M[0..max_order], where M[n] = ∫ z^n f(z) dz.
    """
```

**計算**
[
M_n \approx \int z^n f(z),dz
]
は台形則（`np.trapezoid`）で計算。

#### (B) `compute_component_raw_moments`

**目的**：各ガウス成分 (\mathcal{N}(\mu,\sigma^2)) の生モーメントを閉形式で返す。
ガウスの生モーメント（0〜4次）は以下：
[
\begin{aligned}
M_0 &= 1\
M_1 &= \mu\
M_2 &= \mu^2+\sigma^2\
M_3 &= \mu^3+3\mu\sigma^2\
M_4 &= \mu^4+6\mu^2\sigma^2+3\sigma^4
\end{aligned}
]

**シグネチャ（推奨：ベクトル対応）**

```python
def compute_component_raw_moments(mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    """
    Returns A shape (5, K):
      A[0,k]=1
      A[1,k]=mu[k]
      A[2,k]=mu[k]^2 + var[k]
      A[3,k]=mu[k]^3 + 3*mu[k]*var[k]
      A[4,k]=mu[k]^4 + 6*mu[k]^2*var[k] + 3*var[k]^2
    """
```

#### (C) `compute_linf_errors_pdf_cdf_quantiles`（評価用：新規推奨）

**目的**：PDF/CDFの L∞誤差と分位点誤差をまとめて返す。

**シグネチャ**

```python
def compute_errors(
    z: np.ndarray,
    f_true: np.ndarray,
    f_hat: np.ndarray,
    quantile_ps: list[float] = [0.9, 0.99, 0.999],
) -> dict:
    """
    Returns:
      linf_pdf, linf_cdf, quantile_errors{p: abs(q_true - q_hat)}, q_true, q_hat
    """
```

---

## 6.2 `lp_method.py` の改良

## 6.2.1 新 LP ソルバ：`solve_lp_pdf_rawmoments_linf`

### 目的

* PDF の L∞誤差を抑えつつ
* 生モーメント (M_1..M_4) をターゲットに近づける
* すべて線形制約・線形目的で解く（反復線形化を廃止）

### 変数

辞書サイズを (m)、グリッド点数を (N) とする。未知変数は
[
x=
\begin{bmatrix}
w_1,\dots,w_m,\ t_{\rm pdf},\ t_1,\ t_2,\ t_3,\ t_4
\end{bmatrix}^T
]

* (w_j\ge 0)：辞書成分の重み
* (t_{\rm pdf}\ge 0)：PDF の L∞誤差上界
* (t_n\ge 0)：生モーメント誤差の上界（(n=1..4)）

### 制約

#### (A) PDF の L∞ 制約

[
\left|\sum_{j=1}^{m} w_j,\Phi_{ij}-f_i\right|\le t_{\rm pdf}\quad (i=1..N)
]

#### (B) 重みの制約

[
\sum_{j=1}^{m} w_j = 1,\quad w_j\ge 0
]

#### (C) 生モーメント制約（線形）

辞書成分 (j) の生モーメントを (M_{n,j})（`compute_component_raw_moments` で計算）とし、
[
\left|\sum_{j=1}^{m} w_j,M_{n,j}-M_n^{\rm (target)}\right|\le t_n
\quad (n=1..4)
]

#### (D) オプション：`pdf_tolerance`

[
t_{\rm pdf}\le \tau_{\rm pdf}
]
（`linprog` の bounds で表現）

### 目的関数（線形）

以下どちらかを実装（両方実装して config で切替推奨）

* **方式A（推奨：PDF を必ず抑える）**
  [
  \min \ \lambda_1 t_1+\lambda_2 t_2+\lambda_3 t_3+\lambda_4 t_4
  \quad \text{s.t. } t_{\rm pdf}\le \tau_{\rm pdf}
  ]
* **方式B（単一目的）**
  [
  \min \ \lambda_{\rm pdf} t_{\rm pdf}+\sum_{n=1}^{4}\lambda_n t_n
  ]

### シグネチャ（必須）

```python
def solve_lp_pdf_rawmoments_linf(
    Phi_pdf: np.ndarray,      # shape (N, m)
    mus: np.ndarray,          # shape (m,)
    sigmas: np.ndarray,       # shape (m,)
    z: np.ndarray,            # shape (N,) (target raw moments算出に使う)
    f: np.ndarray,            # shape (N,)
    pdf_tolerance: float | None,
    lambda_pdf: float,
    lambda_raw: tuple[float,float,float,float],  # (λ1..λ4)
    solver: str = "highs",
) -> dict:
    """
    Returns:
      w, t_pdf, t_raw(4,), objective, status, message, diagnostics
    """
```

### diagnostics に必ず入れる項目

* `n_dict = m`
* `t_pdf`
* `raw_target = (M1..M4)`
* `raw_mix = (M1..M4)`
* `raw_abs_err = |raw_mix - raw_target|`
* `n_nonzero = #(w_j > eps)`
* `selected_indices`（後段の Hybrid で上位Kを取るため、`argsort(w)`結果も保持可）

---

## 6.2.2 辞書生成の拡張（μ のテール重視分位点）

既存 `build_gaussian_dictionary` を拡張し、以下を追加する。

### 追加パラメータ

* `tail_focus: str`：`"none" | "right" | "left" | "both"`
* `tail_alpha: float`：(\alpha \ge 1)。大きいほど tail に寄せる。
* `quantile_levels: Optional[np.ndarray]`：ユーザが分位点レベル (p_j) を直接指定したい場合

### 分位点レベルの生成（例：right tail）

基準の一様レベル (u_j) を
[
u_j=\frac{j+0.5}{J},\quad j=0,1,\dots,J-1
]
とし、右裾重視は
[
p_j=1-(1-u_j)^{\alpha}
]
（(\alpha>1) で (p_j) が 1 に寄る）

### μ 候補の計算

ターゲット CDF (F(z))（台形則で計算）に対し
[
\mu_j = F^{-1}(p_j)
]
は補間（`np.interp(p_j, F, z)`）で求める。

### 実装上の注意

* `F` は単調増加になっていること（`pdf_to_cdf_trapz` 内で単調化・0..1のクリップ推奨）
* (p_j) は `(eps, 1-eps)` にクリップ（例：`eps=1e-6`）

---

## 6.2.3 `fit_gmm_lp_simple` の拡張（互換性維持）

### 追加する `objective_mode`

* 既存：`"pdf"`, `"moments"`
* 追加：`"raw_moments"`

### 追加する辞書モード

* 既存：`build_gaussian_dictionary_simple(K,L,...)`
* 追加：`build_gaussian_dictionary(J,L, mu_mode="quantile", tail_focus=..., tail_alpha=...)`

### 推奨：パラメータ名の整理（混乱回避）

* `K`：最終GMMの成分数（EM/HYBRIDで使用）
* `J_dict`：辞書の μ 候補数
* `L_dict`：辞書の σ 候補数
* 辞書サイズ：`m = J_dict * L_dict`

**後方互換**：既存 config しかない場合は

* `J_dict = K`（旧仕様の K を辞書μ数として扱う）
* `L_dict = L`

---

## 6.3 `em_method.py` の改良

## 6.3.1 `max_pdf_bivariate_normal` の数値安定化（ρ→±1）

現状：
[
\sigma_{Y|X=z} = \sigma_Y\sqrt{1-\rho^2}
]
が 0 に近づくと 0/0 が発生しうる。

### 仕様

* `rho_clip = min(max(rho, -1+eps_rho), 1-eps_rho)` を内部で使う（例：`eps_rho=1e-12`）
* もしくは `cond_std = max(sigma * sqrt(max(1-rho^2, eps)), std_floor)` を使う
* `std_floor` は `1e-15` 程度（または `SIGMA_FLOOR`）

---

## 6.3.2 EM 初期化に `"custom"` を追加（Hybrid 用）

### 目的

LP の上位 (K) 成分（(\pi,\mu,\sigma^2)）を EM 初期値として投入できるようにする。

### 仕様

`fit_gmm1d_to_pdf_weighted_em` の `init` に `"custom"` を追加し、`init_params` に以下を要求する：

* `pi_init: np.ndarray shape (K,)`
* `mu_init: np.ndarray shape (K,)`
* `var_init: np.ndarray shape (K,)`

### 検証ルール

* `pi_init >= 0`、`sum(pi_init) > 0`、正規化して `sum=1`
* `var_init >= reg_var`（下限クリップ）
* shape 不一致なら `ValueError`

### 反復初期化（n_init > 1 の扱い）

* `trial==0`：そのまま
* `trial>0`：`mu_init` と `var_init` に小さな摂動（例：1%）を加えたものも試して良い（任意）

---

## 6.4 `main.py` の改良

## 6.4.1 method="hybrid" を追加

### config 例（推奨）

```json
{
  "method": "hybrid",
  "K": 5,
  "L": 10,
  "lp_params": {
    "dict_J": 40,
    "dict_L": 10,
    "mu_mode": "quantile",
    "tail_focus": "right",
    "tail_alpha": 2.0,
    "solver": "highs",
    "objective_mode": "raw_moments",
    "pdf_tolerance": 1e-4,
    "lambda_pdf": 1.0,
    "lambda_raw": [1.0, 1.0, 1.0, 1.0]
  },
  "max_iter": 200,
  "tol": 1e-9,
  "reg_var": 1e-6,
  "init": "custom",
  "use_moment_matching": true,
  "qp_mode": "hard"
}
```

### Hybrid の処理手順（必須）

1. 真の `f_true` 作成（既存）
2. LP 用辞書を生成（`dict_J, dict_L`）
3. `objective_mode="raw_moments"` なら `solve_lp_pdf_rawmoments_linf`
4. LP 解の重み `w_all` から上位 (K) 成分を選ぶ
   [
   \text{idx}=\text{argsort}(w_{\rm all})[::-1][:K]
   ]
5. EM を `init="custom"` で実行（`pi_init, mu_init, var_init` を投入）
6. `use_moment_matching=True` なら既存 QP を適用（既存仕様を流用）
7. 評価指標を計算して出力（次節）

---

## 6.4.2 評価指標の追加（標準出力＋レポート用）

最低限追加する（`gmm_utils.compute_errors` を呼ぶ想定）

* PDF L∞誤差
  [
  \max_i |f_{\rm true}(z_i)-f_{\rm hat}(z_i)|
  ]
* CDF L∞誤差
  [
  \max_i |F_{\rm true}(z_i)-F_{\rm hat}(z_i)|
  ]
* 分位点誤差（例：p=0.9,0.99,0.999）
  [
  |q_p^{\rm true}-q_p^{\rm hat}|
  ]
* 右裾重み付き L1 誤差（例：p0=0.9）
  [
  \int_{q_{p0}}^{\infty} |f_{\rm true}(z)-f_{\rm hat}(z)|,dz
  ]
  （グリッド上で台形則）

---

## 7. 受け入れ条件（Acceptance Criteria）

1. `objective_mode="raw_moments"` の LP が **反復なし**で解ける（status=0）
2. `method="hybrid"` が end-to-end で動く（LP→EM→QP）
3. ρ=0.99 等の高相関でも NaN/Inf が出ない
4. 既存の `method="em"` と `method="lp"` が壊れない（後方互換）

---

## 8. テスト計画

### 8.1 単体テスト（推奨）

* `compute_component_raw_moments`：(\mu=0,\sigma^2=1) で
  (M_1=0), (M_2=1), (M_3=0), (M_4=3)
* `compute_pdf_raw_moments`：標準正規 PDF を広い範囲で与え、同上に近い値になる
* `build_gaussian_dictionary`：`tail_alpha>1` で `mu_candidates` が右側に寄る（`np.diff(mu)` の分布で確認）

### 8.2 結合テスト（推奨）

* 代表パラメータセットを固定して回す：
  ((\mu_x,\sigma_x,\mu_y,\sigma_y,\rho)) を 5〜10ケース（ρ=0,0.9,0.99 を含む）
* 既存 `method_recommendations.md` の表に新方式（LP raw_moments / hybrid）を追記できる形式で JSON or CSV を吐く（任意）

---

## 9. 実装メモ（落とし穴）

* LP の `A_ub` は大きくなりがち：必要なら `scipy.sparse.csr_matrix` を使用する
* `pdf_tolerance` は小さすぎると infeasible：
  方式Aの場合、失敗時に `pdf_tolerance *= 10` を最大数回まで緩める fallback を入れてよい（無限ループ禁止）
* LP で得た `w_all` の上位K選択後、必ず `pi_init /= sum(pi_init)` で正規化
* EM の `var_init` は `reg_var` 以上にクリップ

---
