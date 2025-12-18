# 仕様書1：方法1（QMI）分位点ビン局所モーメント初期化

## 目的

MAX 後の真の 1次元 pdf (f_Z(z)) が **グリッド**で与えられたとき、任意の成分数 (K) の 1次元 GMM（Gaussian Mixture Model：正規分布の混合）の **EM 初期値** ((\pi_k,\mu_k,\sigma_k^2)) を決定的に生成する。

GMM は
[
\hat f_Z(z)=\sum_{k=1}^{K}\pi_k,\mathcal N(z;\mu_k,\sigma_k^2)
]
で表す。ここで

* (\pi_k)：混合比（重み）。(\pi_k\ge 0), (\sum_k \pi_k=1)
* (\mu_k)：平均
* (\sigma_k^2)：分散（正）

## 入力

* `z: np.ndarray shape (N,)`
  単調増加（strictly increasing）な実数グリッド (z_0<z_1<\cdots<z_{N-1})
* `f: np.ndarray shape (N,)`
  各点の pdf 値 (f_i \approx f_Z(z_i))。非負を仮定（負があれば 0 にクリップしてよい）
* `K: int`
  成分数 (K\ge 1)
* `sigma_floor: float`（任意）
  分散の下限 (\sigma_{\min}^2)（例：`1e-12` あるいは `1e-6 * varZ`）
* `mass_floor: float`（任意）
  ビン質量が極小のときの回避用（例：`1e-15`）

## 出力

* `pi: np.ndarray shape (K,)`
* `mu: np.ndarray shape (K,)`
* `var: np.ndarray shape (K,)`
  すべて `float64` 推奨。
* 追加で返してよい：`edges: np.ndarray shape (K+1,)`（分位点境界）

## 数値積分（定義）

ここでの (\int_a^b g(z),dz) は **台形則**で評価する。
グリッド配列 `z` と値 `g` が与えられたとき
[
\int g(z),dz \approx \sum_{i=0}^{N-2}\frac{g_i+g_{i+1}}{2}(z_{i+1}-z_i)
]
として計算する（NumPy の `np.trapz(g,z)` で可）。

## アルゴリズム

### Step 0: pdf 正規化

1. `f = np.maximum(f, 0)`
2. `area = trapz(f,z)`
3. `f = f / area`（`area<=0` なら例外）

### Step 1: CDF を作る

CDF（累積分布関数） (F(z)) は
[
F(z)=\int_{-\infty}^{z} f_Z(t),dt
]
グリッド上では `cdf[i]` を
[
\mathrm{cdf}[0]=0,\quad
\mathrm{cdf}[i]=\sum_{j=0}^{i-1}\frac{f_j+f_{j+1}}{2}(z_{j+1}-z_j)
]
で作る。最後に `cdf[-1]` が 1 に近い（数値誤差は許容）。

### Step 2: 分位点境界（quantile edges）を求める

分位点境界 (q_k) は
[
F(q_k)=\frac{k}{K}\quad (k=0,\dots,K)
]

* `targets = np.linspace(0,1,K+1)`
* `edges = interp(targets, cdf, z)`（線形補間）

  * `edges[0]=z[0]`, `edges[-1]=z[-1]` にクリップ可

### Step 3: 各ビン (I_k=[q_{k-1},q_k]) の局所モーメント

ビン (k) の質量
[
\pi_k = \int_{q_{k-1}}^{q_k} f_Z(z),dz
]
平均
[
\mu_k=\frac{1}{\pi_k}\int_{q_{k-1}}^{q_k} z f_Z(z),dz
]
分散
[
\sigma_k^2=\frac{1}{\pi_k}\int_{q_{k-1}}^{q_k} (z-\mu_k)^2 f_Z(z),dz
]

実装上：

* 境界 `edges[k-1], edges[k]` に対応する区間だけを使って台形則で積分する
  （簡易には、境界に挟まれるインデックス範囲を `searchsorted` で取り、必要なら境界点を補間して区間端点に追加する）
* `pi_k < mass_floor` の場合：

  * 対応する成分を一時的に `pi_k=0` として記録し、後で隣接ビンと統合するか、
  * `mu_k` をビン中央、`var_k` を小さめ（`sigma_floor`）で置く（推奨：統合）

### Step 4: 正規化と安定化

* `pi = np.maximum(pi, 0)`
* `pi = pi / pi.sum()`（0除算なら例外）
* `var = np.maximum(var, sigma_floor)`
* 返す前に `mu` でソートしてもよい（`argsort(mu)`）

## 要求仕様（必須）

* 返り値は必ず

  * (\pi_k\ge 0)
  * (\sum_k \pi_k = 1)（許容誤差 `1e-12`）
  * (\sigma_k^2>0)
* 入力 `z` が昇順でない場合は例外
* `K > N/2` のように分割が細かすぎる場合は警告または例外（推奨：例外）

## テスト

1. `f` が単峰の正規 pdf のとき、出力 GMM が概ねその形に沿う（分散が極端に小さくならない）
2. `K=1` のとき、
   [
   \mu_1\approx \int z f(z)dz,\quad \sigma_1^2\approx \int (z-\mu_1)^2f(z)dz
   ]
3. `sum(pi)==1` が保たれる

---

# 仕様書2：方法2（WQMI）勝者分解 + 分位点ビン局所モーメント初期化

## 目的

2入力 MAX
[
Z=\max(X,Y)
]
の MAX 後 pdf を、**勝者（Xが勝つ／Yが勝つ）**の寄与に分解した 2 つの非負関数
[
g_X(z),\ g_Y(z)\quad (\ge 0)
]
を用いて、任意の (K) の GMM 初期値 ((\pi_k,\mu_k,\sigma_k^2)) を作る。

（WQMI は QMI の拡張で、MAX の構造を初期値に埋め込む。）

## 入力（2パターン対応させる）

### 入力パターンA（推奨）：`gx, gy` が既に与えられる

* `z: np.ndarray (N,)` 昇順
* `gx: np.ndarray (N,)` (g_X(z_i))
* `gy: np.ndarray (N,)` (g_Y(z_i))
* `K: int` 成分数 (K\ge 2)
* `sigma_floor, mass_floor`（仕様書1と同様）

このとき真の pdf は
[
f_Z(z)=g_X(z)+g_Y(z)
]
である。

### 入力パターンB：2変量正規（bivariate normal）から `gx, gy` を計算する

* `mux, varx, muy, vary, rho`（相関係数）
  (X,Y) が2変量正規である場合、次で定義：

  * (f_X(z)=\mathcal N(z;\mu_X,\sigma_X^2))
  * (f_Y(z)=\mathcal N(z;\mu_Y,\sigma_Y^2))
  * (Y|X=z) の条件付き正規、(X|Y=z) の条件付き正規

[
g_X(z)=f_X(z),P(Y\le z\mid X=z)
]
[
g_Y(z)=f_Y(z),P(X\le z\mid Y=z)
]
で計算する（(\Phi) を用いる）。

（**注**：このパターンBの式は、実装の依存が増えるので、可能なら `gx,gy` を先に生成してパターンAで初期化すること。）

## 出力

* `pi, mu, var`（shape (K,)）
* 任意：`KX, KY`（X側とY側に割り当てた成分数）、各側の `edges`

## アルゴリズム

### Step 0: 正規化

`gx, gy` を非負にし、合計質量で正規化する：
[
total = \int (g_X(z)+g_Y(z))dz
]
[
g_X \leftarrow g_X/total,\quad g_Y \leftarrow g_Y/total
]
すると
[
\int (g_X+g_Y)dz = 1
]

### Step 1: 勝率（混合比の大枠）

[
p_X=\int g_X(z)dz,\quad p_Y=\int g_Y(z)dz=1-p_X
]

### Step 2: 成分数配分

[
K_X=\max(1,\min(K-1,\mathrm{round}(Kp_X))),\quad K_Y=K-K_X
]

* 必ず (K_X\ge 1), (K_Y\ge 1)

### Step 3: 各側を QMI で初期化

QMI（仕様書1）をそのまま使う。ただし pdf として使うのは：

* X側：
  [
  h_X(z)=\frac{g_X(z)}{p_X}
  ]
  に対して QMI を (K_X) 成分で適用し、((\tilde\pi^{(X)}_j,\mu^{(X)}_j,\sigma^{2(X)}_j)) を得る
  ただしこの (\tilde\pi) は (\sum \tilde\pi = 1) なので、真の重みは
  [
  \pi^{(X)}_j = p_X,\tilde\pi^{(X)}_j
  ]

* Y側も同様に
  [
  h_Y(z)=\frac{g_Y(z)}{p_Y}
  ]
  にQMIを (K_Y) 成分で適用し、(\pi^{(Y)}_j=p_Y\tilde\pi^{(Y)}_j)

### Step 4: 結合と最終正規化

X側 (K_X) 個とY側 (K_Y) 個を結合して K 個にする。
最後に

* `pi = pi / pi.sum()`（数値誤差対策）
* `var = max(var, sigma_floor)`
* `mu` で成分を昇順ソート（任意だが推奨）

## 要求仕様（必須）

* (\pi_k\ge 0), (\sum_k \pi_k=1), (\sigma_k^2>0)
* `K>=2` を要求（勝者が2つあるので）
* `gx,gy` が全ゼロ等で `total<=0` の場合は例外

## テスト

* `gx+gy` を `f` として QMI した結果（方法1）と比べ、WQMI の方が少ない (K) で kink/歪みを拾うことを確認
* `KX+KY==K`、`sum(pi)==1`
