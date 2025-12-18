# AI向け実装仕様書：線形計画法（LP）による GMM フィッティング

この仕様書は、**真の 1次元 pdf** (f(z))（MAX 後 pdf を含む）を **ガウス混合（GMM）**で近似するために、

1. **重み（混合比）だけ**を LP で求める
2. **辞書ベースの基底関数生成**でK*L個の候補から最適な重みを選択する
   …という方式を、他のAIが Python で実装できるレベルに落としたものです。

**注意**: この仕様書は古い実装について記載していますが、現在の実装では以下の2つのモードのみが利用可能です：
- `objective_mode="pdf"`: PDF誤差のみを最小化（L∞ノルム）
- `objective_mode="moments"`: モーメント誤差を最小化（PDF誤差を制約として使用）

詳細な実装については、`lp_method.py`の`fit_gmm_lp_simple`関数を参照してください。

---

## 0. 用語定義（未定義語を避けるため明示）

* **グリッド**：単調増加の配列 (z_0<z_1<\cdots<z_{N-1})。
* **真の pdf**：グリッド上の値 (f_i \approx f(z_i))（非負）。
* **ガウス pdf（正規分布 pdf）**：
  [
  \mathcal N(z;\mu,\sigma^2)=\frac{1}{\sqrt{2\pi}\sigma}\exp!\left(-\frac{(z-\mu)^2}{2\sigma^2}\right)
  ]
* **ガウス基底（Gaussian basis）**：候補のガウス pdf (\phi_j(z)=\mathcal N(z;\mu_j,\sigma_j^2))。
* **辞書（dictionary）**：基底の集合 ({\phi_j}_{j=1}^{m})（(m) は候補数）。
* **混合（mixture）**：
  [
  \hat f(z)=\sum_{j\in S} w_j \phi_j(z)
  ]
  ここで (S) は選ばれた基底の添字集合。
  混合比（重み） (w_j) は
  [
  w_j\ge 0,\qquad \sum_{j\in S} w_j = 1
  ]
* **誤差（残差）**：グリッド点での差
  [
  r_i^{(pdf)}=\hat f(z_i)-f(z_i)
  ]
* **無限ノルム（(\ell_\infty)）**：
  [
  |r|_\infty = \max_i |r_i|
  ]

---

## 1. 目的（最終成果物）

与えられた真の pdf (f(z)) に対し、**K*L個の基底関数**から最適な重みを選択して近似
[
\hat f(z)=\sum_{j=1}^{m} w_j \phi_j(z)
]
を返す。ここで
[
w_j\ge 0,\qquad \sum_{j=1}^{m} w_j=1
]
が成り立つこと。

本方式では、((\mu_j,\sigma_j)) は辞書から生成し、(\pi_j) は LP で決める。

---

## 2. I/F（実装すべき関数）

### 2.1 必須：パイプライン

```python
def fit_gmm_lp_simple(
    z: np.ndarray,             # (N,)
    f: np.ndarray,             # (N,)
    K: int,                    # number of segments
    L: int,                    # number of sigma levels per segment
    lp_params: dict,           # LP settings (solver, sigma scales, objective mode params)
    objective_mode: str = "pdf"  # "pdf" or "moments"
) -> dict:
    """
    Returns:
      {
        "weights": np.ndarray (K*L,),   # w_j (may be sparse)
        "mus":     np.ndarray (K*L,),
        "sigmas":  np.ndarray (K*L,),
        "lp_objective": float,
        "diagnostics": {...}
      }
    """
```

### 2.2 必須：辞書生成

```python
def build_gaussian_dictionary_simple(
    z: np.ndarray,             # (N,)
    f: np.ndarray,             # (N,)
    K: int,                    # number of segments
    L: int,                    # number of sigma levels per segment
    sigma_min_scale: float,    # minimum sigma scale
    sigma_max_scale: float     # maximum sigma scale
) -> dict:
    """
    Returns:
      {
        "mus": np.ndarray (K*L,),    # mu_j
        "sigmas": np.ndarray (K*L,)  # sigma_j
      }
    """
```

### 2.3 必須：基底行列計算（PDF）

```python
def compute_basis_matrices(
    z: np.ndarray,             # (N,)
    mus: np.ndarray,           # (m,)
    sigmas: np.ndarray         # (m,)
) -> dict:
    """
    Returns:
      {
        "Phi_pdf": np.ndarray (N,m),   # Phi_pdf[i,j] = N(z_i; mu_j, sigma_j^2)
        "Phi_cdf": np.ndarray (N,m)    # Phi_cdf[i,j] = Φ((z_i-mu_j)/sigma_j) (for reference)
      }
    """
```

### 2.4 必須：subset 上での LP 解法（PDF誤差最小化）

```python
def solve_lp_pdf_linf(
    Phi_pdf_sub: np.ndarray,   # (N,s)
    f: np.ndarray,              # (N,)
    solver: str = "highs"       # LP solver
) -> dict:
    """
    Minimizes: t_pdf
    Subject to:
        -t_pdf <= sum(w_j * Phi_pdf[:,j]) - f <= t_pdf
        w_j >= 0, sum(w_j) = 1
        t_pdf >= 0
    
    Returns:
      {
        "w": np.ndarray (s,),
        "t_pdf": float,
        "objective": float,
        "status": int,
        "message": str
      }
    """
```

### 2.5 モーメント誤差最小化モード（オプション）

```python
def solve_lp_pdf_moments_linf(
    Phi_pdf_sub: np.ndarray,   # (N,s)
    mus: np.ndarray,           # (s,)
    sigmas: np.ndarray,        # (s,)
    f: np.ndarray,             # (N,)
    target_mean: float,
    target_variance: float,
    target_skewness: float,
    target_kurtosis: float,
    lambda_mean: float = 1.0,
    lambda_variance: float = 1.0,
    lambda_skewness: float = 1.0,
    lambda_kurtosis: float = 1.0,
    solver: str = "highs",
    pdf_tolerance: float = 1e-6,
    max_moment_iter: int = 5,
    moment_tolerance: float = 1e-6
) -> dict:
    """
    Minimizes: λ_mean * t_mean + λ_variance * t_var + λ_skewness * t_skew + λ_kurtosis * t_kurt
    Subject to:
        -t_pdf <= sum(w_j * Phi_pdf[:,j]) - f <= t_pdf  (PDF constraint)
        t_pdf <= pdf_tolerance (hard constraint)
        -t_mean * |mean_target| <= mean_mixture - mean_target <= t_mean * |mean_target|
        -t_var * |var_target| <= var_mixture - var_target <= t_var * |var_target|
        -t_skew * |skew_target| <= skew_mixture - skew_target <= t_skew * |skew_target|
        -t_kurt * |kurt_target| <= kurt_mixture - kurt_target <= t_kurt * |kurt_target|
        w_j >= 0, sum(w_j) = 1
        t_mean >= 0, t_var >= 0, t_skew >= 0, t_kurt >= 0
    
    Note: Variance, skewness, and kurtosis constraints are nonlinear and are linearized iteratively.
    
    Returns:
      {
        "w": np.ndarray (s,),
        "t_pdf": float,
        "t_mean": float,
        "t_var": float,
        "t_skew": float,
        "t_kurt": float,
        "objective": float,
        "status": int,
        "message": str,
        "moment_errors": dict
      }
    """
```

---

## 3. 前処理（pdfの正規化）

グリッド上の pdf (f) を正規化して、積分が1になるようにする。

---

## 4. 辞書生成（K*L個の基底）

* `K`個のセグメントに分割（平均位置の候補）
* 各セグメントに対して`L`個のシグマレベルを生成
* 合計`K*L`個の基底関数を生成

---

## 5. LP定式化

### 5.1 PDF誤差最小化モード（`objective_mode="pdf"`）

変数: (w_1,\ldots,w_s, t_{pdf})

目的関数:
[
\min\ t_{pdf}
]

制約:
[
-t_{pdf}\ \le\ \sum_{j=1}^{s} w_j \Phi_{pdf}[i,j] - f_i\ \le\ t_{pdf},\quad \forall i
]

[
\sum_{j=1}^{s} w_j = 1,\qquad w_j\ge 0,\qquad t_{pdf}\ge 0
]

### 5.2 モーメント誤差最小化モード（`objective_mode="moments"`）

変数: (w_1,\ldots,w_s, t_{pdf}, t_{mean}, t_{var}, t_{skew}, t_{kurt})

目的関数:
[
\min\ \lambda_{mean} t_{mean} + \lambda_{var} t_{var} + \lambda_{skew} t_{skew} + \lambda_{kurt} t_{kurt}
]

制約:
[
-t_{pdf}\ \le\ \sum_{j=1}^{s} w_j \Phi_{pdf}[i,j] - f_i\ \le\ t_{pdf},\quad \forall i
]

[
t_{pdf} \le \text{pdf\_tolerance}
]

[
-t_{mean} |\mu_{target}| \le \mu_{mixture} - \mu_{target} \le t_{mean} |\mu_{target}|
]

[
-t_{var} |\sigma^2_{target}| \le \sigma^2_{mixture} - \sigma^2_{target} \le t_{var} |\sigma^2_{target}|
]

[
-t_{skew} |\text{skew}_{target}| \le \text{skew}_{mixture} - \text{skew}_{target} \le t_{skew} |\text{skew}_{target}|
]

[
-t_{kurt} |\text{kurt}_{target}| \le \text{kurt}_{mixture} - \text{kurt}_{target} \le t_{kurt} |\text{kurt}_{target}|
]

[
\sum_{j=1}^{s} w_j = 1,\qquad w_j\ge 0,\qquad t_{pdf}\ge 0, t_{mean}\ge 0, t_{var}\ge 0, t_{skew}\ge 0, t_{kurt}\ge 0
]

**注意**: 分散、歪度、尖度の制約は非線形のため、反復的な線形化（Sequential Linear Programming）を使用します。

---

## 6. 実装の詳細

### 6.1 辞書生成

* `build_gaussian_dictionary_simple`: K個のセグメントとL個のシグマレベルからK*L個の基底を生成

### 6.2 LP解法

* `solve_lp_pdf_linf`: PDF誤差のみを最小化
* `solve_lp_pdf_moments_linf`: モーメント誤差を最小化（PDF誤差を制約として使用）

### 6.3 モーメント制約の線形化

分散、歪度、尖度は重みに対して非線形のため、以下の反復プロセスを使用：

1. 現在の重みからモーメントを計算
2. モーメント制約を現在の重み周りで線形化
3. LPを解く
4. 重みが収束するまで1-3を反復

---

## 7. 実装の注意点

* PDF誤差最小化モードでは、CDFは考慮されません
* モーメント誤差最小化モードでは、PDF誤差を制約として使用し、モーメント誤差を最小化します
* 非線形なモーメント制約（分散、歪度、尖度）は反復的な線形化で処理されます
* `max_moment_iter`と`moment_tolerance`で反復の収束を制御します
