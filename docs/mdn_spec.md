# 設計書：MDN（Mixture Density Network）による EM 初期値のアモータイズド推定

## 0. 目的

MAX 演算後の 1 次元 pdf (f(z))（グリッド上で与えられる）から、**K 成分 GMM（Gaussian Mixture Model）の初期値**
[
{\pi_k,\mu_k,\sigma_k^2}_{k=1}^{K}
]
を **機械学習モデル（MDN）で直接推定**し、EM 法（Expectation-Maximization）の

* 多スタート回数 (n_{\text{init}})
* 反復回数 (I)

を減らして高速化・安定化する。

---

## 1. 用語定義

* **pdf**（probability density function）：確率密度関数 (f(z))
* **GMM**：
  [
  \hat f(z)=\sum_{k=1}^{K}\pi_k,\mathcal N(z;\mu_k,\sigma_k^2),\quad
  \pi_k\ge 0,\quad \sum_{k=1}^{K}\pi_k=1,\quad \sigma_k>0
  ]
* **EM 法**：GMM の最尤推定で使う反復アルゴリズム
* **MDN（Mixture Density Network）**：ニューラルネットが「混合分布（ここでは GMM）のパラメータ」を直接出力する枠組み
* **MLP**（Multi-Layer Perceptron）：全結合層（Linear）を重ねたニューラルネット
* **softmax**：任意実数ベクトル (\alpha) を確率ベクトル (\pi) に変換
  [
  \pi_k=\frac{e^{\alpha_k}}{\sum_{j=1}^{K} e^{\alpha_j}}
  ]
* **softplus**：任意実数 (\beta) を正の値に変換
  [
  \mathrm{softplus}(\beta)=\log(1+e^{\beta})
  ]
* **log-sum-exp**：
  [
  \log\sum_k e^{a_k}=m+\log\sum_k e^{a_k-m},\quad m=\max_k a_k
  ]
  （数値安定化のために使う）

---

## 2. 入出力仕様

### 2.1 入力（推論時）

1 サンプルあたり：

* グリッド点：({z_i}_{i=1}^{N})（例：(N=64)）
* 真（またはターゲット）の pdf サンプル：({f_i}_{i=1}^{N}), (f_i=f(z_i))

**入力テンソルの推奨形**（固定グリッドを前提）：

* (x \in \mathbb{R}^{N})： (x_i = \tilde f_i)

ここで (\tilde f) は必ず正規化する：
[
\tilde f_i=\frac{f_i}{\int f(z),dz}\approx \frac{f_i}{\sum_i f_i w_i}
]
(w_i) は積分重み（等間隔なら (w_i=\Delta z)、台形則なら両端が (\Delta z/2)）。

> 備考：グリッド (z) がサンプルごとに変わる場合は、**固定グリッドへ補間**してから入力する（§6）。

### 2.2 出力（推論時）

ネットワークは以下の **非制約パラメータ**を出力する：

* (\alpha_k)（混合重みの logits）
* (\mu_k)（平均）
* (\beta_k)（分散パラメータの logits）

出力ベクトル：
[
o=[\alpha_1,\ldots,\alpha_K,\ \mu_1,\ldots,\mu_K,\ \beta_1,\ldots,\beta_K]\in\mathbb{R}^{3K}
]

これを GMM パラメータに変換する：
[
\pi_k=\frac{e^{\alpha_k}}{\sum_{j=1}^{K} e^{\alpha_j}}
]
[
\sigma_k=\log(1+e^{\beta_k})+\sigma_{\min},\quad \sigma_{\min}>0
]
[
\sigma_k^2 = (\sigma_k)^2
]

**成分の並び固定（重要）**：ラベル入替の不安定さを抑えるために
[
(\mu_1,\ldots,\mu_K)\ \text{を昇順にソートし、同じ順で}\ \pi_k,\sigma_k\ \text{も並べ替える。}
]

---

## 3. (\hat f_\theta(z_i)) の計算（損失と推論で共通）

ネット出力 ({\pi_k,\mu_k,\sigma_k}) が決まれば、関数型は固定で

[
\hat f_\theta(z)=\sum_{k=1}^{K}\pi_k\frac{1}{\sqrt{2\pi}\sigma_k}
\exp\left(-\frac{(z-\mu_k)^2}{2\sigma_k^2}\right)
]

グリッド点評価は代入するだけ：
[
\hat f_\theta(z_i)=\sum_{k=1}^{K}\pi_k\frac{1}{\sqrt{2\pi}\sigma_k}
\exp\left(-\frac{(z_i-\mu_k)^2}{2\sigma_k^2}\right)
]

数値安定のため、実装では (\log \hat f_\theta(z_i)) を log-sum-exp で計算する：
[
\log \hat f_\theta(z_i)=\log\sum_{k=1}^{K}\exp\Bigl(\log\pi_k + \log\mathcal N(z_i;\mu_k,\sigma_k^2)\Bigr)
]
[
\log\mathcal N(z_i;\mu_k,\sigma_k^2)=
-\frac{1}{2}\log(2\pi)-\log\sigma_k-\frac{(z_i-\mu_k)^2}{2\sigma_k^2}
]

---

## 4. モデル（2層 MLP）仕様

### 4.1 構造

入力 (x\in\mathbb{R}^{N}) に対し、隠れ層 2 層の MLP：

[
\begin{aligned}
h_1 &= \mathrm{ReLU}(W_1 x + b_1),\quad W_1\in\mathbb{R}^{H\times N}\
h_2 &= \mathrm{ReLU}(W_2 h_1 + b_2),\quad W_2\in\mathbb{R}^{H\times H}\
o   &= W_3 h_2 + b_3,\quad W_3\in\mathbb{R}^{(3K)\times H}
\end{aligned}
]

推奨ハイパラ（まず固定で良い）：

* (N=64)（あなたの条件）
* (K=5)（あなたの条件）
* (H=128)
* 活性化：ReLU

### 4.2 正規化（推奨）

入力 pdf はスケールが揃っていても、数値安定のために

* pdf 正規化（積分 1）
* optional：(\log(\tilde f_i+\varepsilon)) へ変換（(\varepsilon=10^{-12})）

どちらかを採用する。デフォルトは **pdf 正規化のみ**。

---

## 5. 学習データ生成（教師あり：pdf → “良い初期値”）

### 5.1 データ生成の基本方針

学習サンプル (s) を大量に作る：

* 入力：MAX 後の真 pdf（グリッド上） (f^{(s)}(z_i))
* 教師：不要（本設計は **教師なしに近い**：損失は pdf 近似のクロスエントロピー）

  * ただし、学習時に必要な「真の pdf」自体は生成する

MAX 後 pdf の生成は、論文（2401.03588v1）で議論される **2 変数正規（相関あり）の MAX の厳密式**（既存コード `max_pdf_bivariate_normal`）を利用する想定。

### 5.2 サンプリングする入力パラメータ例

1 サンプルごとに以下をランダム生成（例）：

* 平均：(\mu_X,\mu_Y \sim \mathrm{Uniform}(\mu_{\min},\mu_{\max}))
* 標準偏差：(\sigma_X,\sigma_Y \sim \mathrm{LogUniform}(\sigma_{\min},\sigma_{\max}))
  （正の量なので対数一様が扱いやすい）
* 相関：(\rho \sim \mathrm{Uniform}(\rho_{\min},\rho_{\max}))

推奨レンジ（まず固定）：

* (\mu_{\min}=-3,\ \mu_{\max}=3)
* (\sigma_{\min}=0.3,\ \sigma_{\max}=2.0)
* (\rho_{\min}=-0.99,\ \rho_{\max}=0.99)

**ρ の数値安定化**：(\rho) 近傍で分母が小さくなる場合があるため、生成側で
[
1-\rho^2 \leftarrow \max(1-\rho^2,\ \varepsilon_\rho),\quad \varepsilon_\rho=10^{-12}
]
などの床（floor）を入れる（既存実装方針と一致させる）。

### 5.3 グリッド設計

固定グリッドを推奨：

* (N=64)
* 範囲 ([z_{\min},z_{\max}]) はサンプルごとに決めず、**標準化して固定**するのが最も簡単。

推奨（固定範囲）：
[
z_i = z_{\min} + i\Delta z,\quad \Delta z = \frac{z_{\max}-z_{\min}}{N-1}
]
[
z_{\min}=-8,\quad z_{\max}=8
]

（この範囲で取りこぼしが出るなら (\pm 10) に拡張）

---

## 6. 損失関数（学習目的）

### 6.1 主損失：クロスエントロピー（CE）

**定義**：
[
\mathcal L_{\mathrm{CE}}(\theta)
================================

-\int f(z),\log \hat f_\theta(z),dz
]

離散化（台形則）：
[
\mathcal L_{\mathrm{CE}}(\theta)
\approx
-\sum_{i=1}^{N} \tilde f(z_i),\log(\hat f_\theta(z_i)+\varepsilon),w_i
]

* (\tilde f)：正規化済み pdf
* (w_i)：積分重み（等間隔なら (w_i=\Delta z)、台形則なら端点 (\Delta z/2)）
* (\varepsilon=10^{-12})（(\log 0) 回避）

> 備考：これは (\mathrm{KL}(f| \hat f_\theta)) と定数差なので、「真 pdf に近い混合 pdf」を学習できる。

### 6.2 オプション損失：モーメントペナルティ（任意）

生モーメント：
[
M_n(f)=\int z^n f(z),dz,\quad n=1,2,3,4
]
同様に (\hat f_\theta) の (M_n(\hat f_\theta)) を計算し
[
\mathcal L_{\mathrm{mom}}(\theta)=\sum_{n=1}^{4}\left(M_n(\hat f_\theta)-M_n(f)\right)^2
]
総損失：
[
\mathcal L(\theta)=\mathcal L_{\mathrm{CE}}(\theta)+\lambda_{\mathrm{mom}}\mathcal L_{\mathrm{mom}}(\theta)
]
推奨：最初は (\lambda_{\mathrm{mom}}=0)（CEのみ）で学習し、必要なら追加。

---

## 7. 学習手順

### 7.1 仕様

* フレームワーク：PyTorch
* 最適化：Adam
* 学習率：(10^{-3})（固定）
* バッチサイズ：256（メモリに応じて）
* エポック：20（固定。足りなければ増やす）

### 7.2 学習ループ（擬似コード）

```python
for batch in loader:
    z, f_true = batch  # z: (B,N), f_true: (B,N)
    f_true = normalize_pdf(z, f_true)  # ∫f=1
    x = f_true  # (B,N)

    alpha, mu, beta = net(x)  # alpha:(B,K), mu:(B,K), beta:(B,K)
    pi = softmax(alpha, dim=-1)
    sigma = softplus(beta) + sigma_min

    # sort by mu
    mu, idx = sort(mu, dim=-1)
    pi = gather(pi, idx)
    sigma = gather(sigma, idx)

    log_fhat = log_gmm_pdf(z, pi, mu, sigma)  # (B,N)  log-sum-exp
    loss_ce = -sum_i f_true_i * log_fhat_i * w_i
    loss = loss_ce + lambda_mom * loss_mom  # optional

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

---

## 8. 推論（EM 初期値として使う）

### 8.1 推論 API（必須）

```python
def mdn_predict_init(
    z: np.ndarray,          # (N,)
    f: np.ndarray,          # (N,)
    K: int,
    model_path: str,
) -> dict:
    """
    Returns:
      {
        "pi":  np.ndarray shape (K,),
        "mu":  np.ndarray shape (K,),
        "var": np.ndarray shape (K,)
      }
    """
```

### 8.2 返り値の要件

* (\pi_k\ge 0)
* (\sum_k \pi_k=1)
* (\mathrm{var}_k=\sigma_k^2 \ge \text{VAR_FLOOR})
* (\mu_k) は昇順（ソート済み）

### 8.3 既存 EM への接続（あなたの現行設計と整合）

既存の `init="custom"` 経由で EM に渡す：

```python
init_params = {"pi": pi_init, "mu": mu_init, "var": var_init}
fit_gmm1d_to_pdf_weighted_em(..., init="custom", init_params=init_params)
```

推奨運用：

* 原則：`n_init = 1`
* `max_iter`（EM反復上限）を小さく（例：10〜30）しても良い
* もし EM の最終評価が悪い場合のみ fallback（§9）

---

## 9. フォールバック（安全策）

MDN は分布外入力で外す可能性があるため、推論時に次の条件で fallback を入れる。

### 9.1 失敗判定

* (\exists i) で (\hat f(z_i)) が NaN/Inf
* (\min_i \hat f(z_i) < 0)（理論上は起きないが数値バグ検出）
* 初期負対数尤度（CE）が閾値より悪い
  [
  \mathcal L_{\mathrm{CE}} > \mathcal L_{\mathrm{CE,th}}
  ]

### 9.2 fallback 先（推奨順）

1. 重み付き k-means++ 初期化（wkmeanspp）
2. 既存 WQMI 初期化
3. 既存 random 多スタート

---

## 10. 評価指標（学習の良否判定）

1 サンプルに対し、MDN 初期値→（必要なら EM）後で以下を測る。

### 10.1 分布近似の誤差

* PDF の (L_\infty)：
  [
  \max_i |f(z_i)-\hat f(z_i)|
  ]
* CDF の (L_\infty)：
  [
  \max_i |F(z_i)-\hat F(z_i)|
  ]
* 分位点誤差（例 (p=0.9,0.99,0.999)）：
  [
  |q_p-\hat q_p|
  ]

### 10.2 計算量の改善

* EM 反復回数 (I)（収束まで）
* 多スタート回数 (n_{\text{init}})
* 総実行時間（任意）

---

## 11. 実装ファイル構成（推奨）

```
ml_init/
  dataset.py        # MAX pdf生成 + 正規化 + fixed-grid化
  model.py          # MLP-MDN本体 + gmm pdf(log-sum-exp)
  train.py          # 学習ループ + checkpoint保存
  infer.py          # mdn_predict_init() 実装
  metrics.py        # pdf/cdf/quantile誤差
  export.py         # 保存形式（pt）と互換性管理
```

---

## 12. 受け入れ条件（Acceptance Criteria）

1. 推論 API が常に (\pi) 正規化・(\sigma>0) を満たし、NaN/Inf を出さない
2. あなたの条件（(N=64,K=5)）で、baseline（例：WQMI + n_init=8）に対して

   * 平均 EM 反復回数が減る、または
   * (n_{\text{init}}) を減らしても同等以上の精度（PDF/CDF/分位点）
3. fallback を入れた状態で、全テストケースが完走する

---

## 付録：固定グリッドでない場合（必須対応の方針）

サンプルごとに ({z_i}) が異なる場合は、**固定グリッド ({\bar z_j}_{j=1}^{N})** を用意し、
[
\bar f(\bar z_j) = \mathrm{interp}\bigl(\bar z_j;\ z,\ f(z)\bigr)
]
で補間してから入力する。補間後に (\int \bar f(z)dz = 1) となるよう再正規化する。

---
