# 設計仕様書：重み付きEM + 最終1回だけ「モーメント一致Q P投影」（方式A・最終投影版）

## 目的

与えられた **真の 1次元 pdf** (f(z))（グリッド上）を、**K成分の 1D GMM**（Gaussian Mixture Model：正規分布の混合）
[
\hat f(z)=\sum_{k=1}^{K}\pi_k,\mathcal N(z;\mu_k,\sigma_k^2)
]
で近似する。

* 通常の **重み付き EM** で ((\tilde\pi_k,\mu_k,\sigma_k^2)) を学習
* **学習の最後に1回だけ**、(\pi) を **小規模QP**で更新して、真の pdf と **平均・分散・歪度・尖度（= 1〜4次中心モーメント）を一致**させる
  （不可能なら「最小二乗で可能な限り一致」に落とす）

本仕様書は「他のAIに実装させる」前提で、Python 実装向けに書く。

---

## 用語定義

* **GMM**：(\sum_k \pi_k \mathcal N(\mu_k,\sigma_k^2)) の形の確率密度。
* **重み付き EM**：グリッド点 (z_i) に対し、重み (w_i\propto f(z_i)\Delta z) を付けた EM。
* **中心モーメント**：
  [
  \mu^*=E[Z],\quad v^*=\mathrm{Var}(Z),\quad
  \mu_3^*=E[(Z-\mu^*)^3],\quad
  \mu_4^*=E[(Z-\mu^*)^4]
  ]
* **歪度**（skewness）：
  [
  \gamma_1^*=\frac{\mu_3^*}{(v^*)^{3/2}}
  ]
* **尖度**（excess kurtosis）：
  [
  \gamma_2^*=\frac{\mu_4^*}{(v^*)^{2}}-3
  ]
  ※制約には歪度・尖度ではなく (\mu_3^*,\mu_4^*) を使う（数値的に安定）。

---

## 入力

* `z: np.ndarray shape (N,)`
  昇順（strictly increasing）のグリッド。
* `f: np.ndarray shape (N,)`
  真の pdf 値（負の場合は 0 にクリップしてよい）。
* `K: int`
  成分数。**モーメント一致（0〜4次 raw）を厳密に狙うなら推奨 (K\ge5)**。
* `em_max_iter: int`（例 300）
* `em_tol: float`（例 1e-10）
* `var_floor: float`（例 1e-12 〜 1e-6 * var_true）
* `qp_mode: str`（`"hard"` or `"soft_fallback"`）
* `soft_lambda: float`（フォールバック時の罰則係数、例 1e3〜1e6）

---

## 出力

* `pi: np.ndarray (K,)`
* `mu: np.ndarray (K,)`
* `var: np.ndarray (K,)`
* 任意：学習ログ（EMの対数尤度、QP成功/失敗、モーメント誤差）

---

## 前処理：真の pdf の正規化と重み

1. `f = np.maximum(f, 0)`
2. `area = trapz(f,z)`（台形則）
3. `f = f/area`
4. EM の重み（グリッド積分の重み）を
   [
   w_i \propto f(z_i)\Delta z_i,\quad \sum_i w_i=1
   ]
   として作る。
   実装は `dz = diff(z); dz = append(dz, dz[-1])` などでOK。

---

## ステップ1：真の pdf のターゲット中心モーメント（1〜4次）

重み (w_i) を使って
[
\mu^*=\sum_i w_i z_i
]
[
v^*=\sum_i w_i (z_i-\mu^*)^2
]
[
\mu_3^*=\sum_i w_i (z_i-\mu^*)^3
]
[
\mu_4^*=\sum_i w_i (z_i-\mu^*)^4
]

---

## ステップ2：ターゲット raw モーメント (M_n^*)（0〜4次）

制約は raw モーメントで書く（(\pi) への線形制約にするため）。

[
M_0^*=1
]
[
M_1^*=\mu^*
]
[
M_2^*=v^*+(\mu^*)^2
]
[
M_3^*=\mu_3^*+3\mu^*v^*+(\mu^*)^3
]
[
M_4^*=\mu_4^*+4\mu^*\mu_3^*+6(\mu^*)^2v^*+(\mu^*)^4
]

---

## ステップ3：重み付き EM による GMM 学習（通常）

### 3.1 初期値

初期化法は任意（例：分位点ビン局所モーメント初期化 QMI）。
最低限：

* (\pi_k=1/K)
* (\mu_k) は `z` の分位点
* (\sigma_k^2) は真の分散の分割程度、かつ `var_floor` 以上

### 3.2 E-step（重み付き責務）

各点 (z_i) と成分 (k) について
[
r_{ik}=
\frac{\pi_k \mathcal N(z_i;\mu_k,\sigma_k^2)}
{\sum_{j=1}^K \pi_j \mathcal N(z_i;\mu_j,\sigma_j^2)}
]
数値安定のため `logsumexp` 推奨。

### 3.3 M-step（重み付き更新）

[
N_k=\sum_i w_i r_{ik}
]
[
\tilde\pi_k = \frac{N_k}{\sum_j N_j} \quad (\sum_j N_j=1\ なので通常 \tilde\pi_k=N_k)
]
[
\mu_k=\frac{1}{N_k}\sum_i w_i r_{ik} z_i
]
[
\sigma_k^2=\frac{1}{N_k}\sum_i w_i r_{ik}(z_i-\mu_k)^2
]
[
\sigma_k^2\leftarrow \max(\sigma_k^2,\texttt{var_floor})
]
収束判定：重み付き対数尤度
[
\mathcal L=\sum_i w_i \log\left(\sum_k \pi_k \mathcal N(z_i;\mu_k,\sigma_k^2)\right)
]
の差分が `em_tol` 未満。

### 3.4 EM 結果

EM の出力を

* (\tilde\pi_k)（EMが求めた混合比）
* (\mu_k,\sigma_k^2)

として保持。

---

## ステップ4：最終1回だけの QP 投影（方式A）

### 4.1 成分 raw モーメント（0〜4次）

正規分布 (\mathcal N(\mu_k,\sigma_k^2)) の raw モーメント：
[
m_{0k}=1
]
[
m_{1k}=\mu_k
]
[
m_{2k}=\mu_k^2+\sigma_k^2
]
[
m_{3k}=\mu_k^3+3\mu_k\sigma_k^2
]
[
m_{4k}=\mu_k^4+6\mu_k^2\sigma_k^2+3\sigma_k^4
]

行列 (A\in\mathbb R^{5\times K}) を
[
A_{n,k}=m_{nk}\quad (n=0,1,2,3,4)
]
ベクトル (b\in\mathbb R^{5}) を
[
b_n=M_n^*
]
とする。

### 4.2 ハード制約QP（可能なら厳密一致）

**変数**：(\pi\in\mathbb R^{K})

**目的**：EM重みからなるべく動かさない
[
\min_{\pi}\ \frac12|\pi-\tilde\pi|_2^2
]

**制約**：
[
A\pi=b
]
[
\pi_k\ge 0\quad(\forall k)
]

成功したら、(\pi) を最終出力に採用し、(\mu,\sigma^2) は EM の値をそのまま使う。

#### 実装指針

* `scipy.optimize.minimize(method="SLSQP")` で可

  * objective: `0.5*np.sum((pi - pi_em)**2)`
  * eq constraints: `A@pi - b == 0`（5本）
  * bounds: `[(0,None)]*K`
* 解が得られない場合を想定し、例外捕捉してフォールバックへ。

### 4.3 フォールバック（常に解けるソフト制約）

ハードが失敗した場合、次の凸問題を解く：

[
\min_{\pi\ge0,\ \sum\pi=1}
\frac12|\pi-\tilde\pi|_2^2
+\frac{\lambda}{2}|A\pi-b|_2^2
]

* (\lambda=\texttt{soft_lambda})
* 制約は (\pi\ge0,\ \sum\pi=1) のみ（これなら常に可解）

実装：

* 同じく SLSQP で

  * objective: `0.5||pi - pi_em||^2 + 0.5*lam*||A@pi-b||^2`
  * eq constraint: `sum(pi)=1`
  * bounds: `pi>=0`

---

## 出力整形

* 最終 `pi` を `pi/pi.sum()` で正規化（数値誤差対策）
* `mu` 昇順で成分を並べ替え（任意だが推奨）
* その場合、`pi,var` も同じ順序で並べ替え

---

## 設計上の注意（重要）

1. **K>=5 推奨**
   等式制約が 5 本（0〜4次 raw）あるため。K<5 だと一致が困難になりやすい。
2. **非負制約があるので、ハードQPは必ず可解ではない**
   そのためフォールバック必須。
3. **var_floor は必須**
   (\sigma_k^2\to 0) の退化を防ぐ（EMでありがち）。
4. **モーメント一致は形状を崩すことがある**
   これは仕様（モーメント一致と形状一致はトレードオフ）。
   ただし「最後に1回だけ投影」は、その崩れを最小化する意図。

---

## 受け入れ基準（Acceptance Criteria）

* EM後の (\tilde\pi) から最終 (\pi) が得られる（ハード成功またはフォールバック成功）
* `pi>=0`、`abs(pi.sum()-1)<1e-12`
* `var>=var_floor`
* ハード成功時：
  [
  |A\pi-b|_\infty < 10^{-8}
  ]
* フォールバック時：
  [
  |A\pi-b|_2
  ]
  が `soft_lambda` を増やすほど減少する（単調性を確認）

---

## 推奨ファイル構成（AI実装タスク向け）

* `moments.py`

  * pdf正規化、重み生成、中心/生モーメント変換
* `em_gmm1d.py`

  * 重み付きEM本体（logsumexp使用）
* `moment_projection.py`

  * A行列生成、ハードQP、フォールバックQP
* `fit_pipeline.py`

  * 全体をつなぐ `fit_gmm_em_then_project(...)`
