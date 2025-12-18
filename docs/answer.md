ここでの方針は「実装側で迷わないように、すべて決め打ち」しています。

---

## 1. 生モーメント計算関数

### Q1-1: `compute_pdf_raw_moments` の返り値形式

* **形状**：`shape = (max_order + 1,)`
* **内容**：

  * `M[0] = 1.0`（数値誤差レベルで 1）
  * `M[n] = ∫ z^n f(z) dz`（正規化後の PDF に対する生モーメント）

**仕様：**

```python
def compute_pdf_raw_moments(
    z: np.ndarray,
    f: np.ndarray,
    max_order: int = 4
) -> np.ndarray:
    """
    z: shape (N,)
    f: shape (N,)  （正規化されていなくても良い）
    戻り値: shape (max_order+1,), M[n] = E[Z^n] （M[0] = 1）
    """
```

内部で `normalize_pdf_on_grid(z, f)` を必ず呼び、正規化後の PDF を使って積分してください。

### Q1-2: PDF正規化のタイミング

* **方針**：`compute_pdf_raw_moments` の中で毎回正規化します。
* すでに正規化済みでも、再度 `normalize_pdf_on_grid` で正規化して構いません。
* 判定用の閾値は不要です（`normalize_pdf_on_grid` 内で `area <= 0` だけ例外にします）。

---

## 2. 辞書生成の拡張（テール重視分位点）

既存の `build_gaussian_dictionary` にパラメータを追加する形でお願いします。

### Q2-1: 左裾（left）と両裾（both）の実装

基準として、`J` 個の μ 候補を作るとき

[
u_j = \frac{j+0.5}{J},\quad j=0,\dots,J-1
]

とします。

* `tail_focus="none"`
  [
  p_j = u_j
  ]
* `tail_focus="right"`
  [
  p_j = 1 - (1 - u_j)^{\alpha}
  ]
* `tail_focus="left"`
  [
  p_j = u_j^{\alpha}
  ]
* `tail_focus="both"`（左右のテール重視・中央値 0.5 対称）
  [
  p_j = 0.5 + \operatorname{sign}(u_j - 0.5)\cdot |u_j - 0.5|^{\alpha}
  ]

ここで (\alpha = \text{tail_alpha}) です。

その後、`p_j` を `eps <= p_j <= 1-eps` にクリップし（`eps = 1e-6` 程度）、
`F = pdf_to_cdf_trapz(z, f)` から

```python
F_monotone = np.maximum.accumulate(F)
F_monotone /= F_monotone[-1]
mu_candidates = np.interp(p, F_monotone, z)
```

で μ を決めてください。

### Q2-2: `tail_alpha` のデフォルト値と範囲

* デフォルト値：`tail_alpha = 1.0`
* 範囲：`tail_alpha >= 1.0` を強制してください。
  `tail_alpha < 1.0` が渡された場合は、内部で `tail_alpha = 1.0` に丸めて構いません。
* `tail_alpha = 1.0` のときは、上記の式から自動的に「通常の一様分位点」と同等になります。

### Q2-3: `quantile_levels` が指定された場合

* `quantile_levels` が **指定されている場合**は、`tail_focus` と `tail_alpha` は **無視** してください。
* そのまま `p = np.asarray(quantile_levels)` を `(eps, 1-eps)` にクリップして使用します。

### Q2-4: CDFの単調性保証

* `pdf_to_cdf_trapz` 自体は単調増加に近い値を返しますが、数値誤差を考慮して、
  分位点計算の直前に **必ず**

```python
F = pdf_to_cdf_trapz(z, f)
F = np.maximum.accumulate(F)
if F[-1] <= 0:
    raise ValueError("CDF integral is non-positive")
F /= F[-1]
```

を行ってください。
これで `F` は厳密に単調増加（かつ `F[0]=0, F[-1]=1`）になります。

---

## 3. LPソルバ `solve_lp_pdf_rawmoments_linf`

### Q3-1: 目的関数の方式選択

* パラメータ名：`objective_form: Literal["A", "B"]`
* デフォルト：`objective_form="A"`

意味は設計書どおりです。

* `"A"`：`pdf_tolerance` を上限制約にしつつ、`t_1..t_4` の線形結合を最小化
* `"B"`：`lambda_pdf * t_pdf + Σ lambda_n * t_n` を一括で最小化

### Q3-2: `lambda_raw` のデフォルト値

* デフォルト：`lambda_raw = (1.0, 1.0, 1.0, 1.0)`
* 自動スケーリングは **実装しません**。モーメントのスケール調整が必要になった場合は、利用側（実験コード）で `lambda_raw` を調整する方針とします。

### Q3-3: `pdf_tolerance` が `None` の場合

* `objective_form="A"` のとき

  * `pdf_tolerance is None` の場合は、**上限制約を付けません**。
    つまり `t_pdf` は下限 `0` のみで、目的関数は「`Σ lambda_n t_n` の最小化 + PDF L∞ 制約」のみになります。
* `objective_form="B"` のとき

  * `pdf_tolerance` は完全に無視します（`t_pdf` に上限を付けない）。

### Q3-4: LP変数の順序と制約行列の構築

変数ベクトル (x) の順序を **固定** します：

[
x = [w_1,\dots,w_m,\ t_{\rm pdf},\ t_1,\ t_2,\ t_3,\ t_4]^T
]

インデックス：

```python
idx_t_pdf = m
idx_t1    = m + 1
idx_t2    = m + 2
idx_t3    = m + 3
idx_t4    = m + 4
n_vars    = m + 5
```

#### (A) PDF L∞ 制約

[
|\Phi_i w - f_i| \le t_{\rm pdf}
]

は、各 (i=0..N-1) について

1. (\Phi_i w - t_{\rm pdf} \le f_i)
2. (-\Phi_i w - t_{\rm pdf} \le -f_i)

なので `A_ub` に 2N 行追加します。

* 行 i（0〜N-1）：

  * `A_ub[i, 0:m] = Phi[i, :]`
  * `A_ub[i, idx_t_pdf] = -1`
  * `b_ub[i] = f[i]`
* 行 N+i（N〜2N-1）：

  * `A_ub[N+i, 0:m] = -Phi[i, :]`
  * `A_ub[N+i, idx_t_pdf] = -1`
  * `b_ub[N+i] = -f[i]`

#### (B) 生モーメント制約

辞書の生モーメント行列 `A_raw` を

```python
A_raw = compute_component_raw_moments(mus, sigmas**2)  # shape (5, m)
```

として、`A_raw[1],...,A_raw[4]` を使います。
ターゲットの生モーメントを `M_target[1..4]` とします。

[
\left|\sum_j w_j M_{n,j} - M_n^{\rm target}\right| \le t_n
\quad (n=1..4)
]

* 各 n = 1..4 につき 2 行追加（合計 8 行）
* 行の作り方（例：n=1）：

  * 行 `row_plus`：

    * `A_ub[row_plus, 0:m] = A_raw[n, :]`
    * `A_ub[row_plus, idx_tn] = -1`  （t_n の列）
    * `b_ub[row_plus] = M_target[n]`
  * 行 `row_minus`：

    * `A_ub[row_minus, 0:m] = -A_raw[n, :]`
    * `A_ub[row_minus, idx_tn] = -1`
    * `b_ub[row_minus] = -M_target[n]`

#### (C) equality 制約

* 重みの総和：

[
\sum_j w_j = 1
]

* `A_eq` 1 行のみ：

  * `A_eq[0, 0:m] = 1`
  * それ以外 0
* `b_eq[0] = 1.0`

#### (D) bounds

* `w_j`：`(0, None)` （非負）
* `t_pdf`：

  * `objective_form="A"` かつ `pdf_tolerance is not None` のとき：`(0, pdf_tolerance)`
  * それ以外：`(0, None)`
* `t_1..t_4`：`(0, None)`

#### (E) A_ub のまとめ方

* PDF制約 + モーメント制約を **1つの `A_ub`** にまとめてください。
  （`linprog` の呼び出しは一回で済ませる）

### Q3-5: スパース行列の使用条件

* 条件：`N * m > 1e6` の場合は、`A_ub` と `A_eq` を `scipy.sparse.csr_matrix` で構築してください。
* それ以下は dense のままで構いません。

### Q3-6: `pdf_tolerance` の fallback 処理

* `objective_form="A"` かつ `pdf_tolerance` が指定されているときのみ fallback を実装。
* 試行回数：最大 3 回

擬似コード：

```python
taus = [pdf_tolerance, pdf_tolerance * 10, pdf_tolerance * 100]
for tau in taus:
    # bounds を tau に合わせて更新
    res = linprog(...)
    if res.success:
        break
if not res.success:
    raise RuntimeError("LP infeasible even after relaxing pdf_tolerance")
```

---

## 4. Hybrid法の実装

### Q4-1: `init="custom"` の実装詳細

`fit_gmm1d_to_pdf_weighted_em` の引数に

```python
init: Literal["wqmi", "random", "kmeans", "custom"] = "wqmi",
init_params: Optional[Dict[str, np.ndarray]] = None,
```

を追加してください。

`init == "custom"` の場合は

* `init_params` が `{"pi": ..., "mu": ..., "var": ...}` を必ず含むこと
* それぞれの shape は `(K,)`

### Q4-2: LPからEMへの初期値の変換

* 基本方針：**重みの大きい順** に上位 `K` 成分を選択します。

```python
idx = np.argsort(w_all)[::-1][:K]
pi_init  = w_all[idx]
mu_init  = mus_all[idx]
var_init = sigmas_all[idx]**2
```

* 重みが同じ場合も、`np.argsort` の仕様通りインデックス順で問題ありません。

### Q4-3: EM初期値の正規化とクリップ

* `pi_init`：`total = pi_init.sum()` を計算し、`total > 0` なら `pi_init /= total`

  * `total <= 0` の場合は `ValueError` を投げてください（Hybrid 呼び出し側で必ず対処すること）。
* `var_init`：`var_init = np.maximum(var_init, VAR_FLOOR)` を適用（`gmm_utils.VAR_FLOOR` を使用）。
* `mu_init` のクリップは不要です。

### Q4-4: QPでの生モーメント一致保証

* QP については **既存実装（中心モーメント／平均・分散・歪度・尖度ベース）をそのまま使用** してください。
* 生モーメント用の QP は **新規実装しません**。
* 真の PDF からの統計量（mean, var, skew, kurt）は、既存の `compute_pdf_statistics` を使って求め、そのまま QP に渡してください。

### Q4-5: `n_init > 1` の `init="custom"` の扱い

* `n_init == 1` のとき：`init_params` をそのまま使用。
* `n_init > 1` のとき：

  * `trial==0`：`init_params` をそのまま使用。
  * `trial>=1`：

    * `mu_trial = mu_init * (1 + 0.01 * randn_like(mu_init))`
    * `var_trial = var_init * (1 + 0.02 * randn_like(var_init))` をクリップ付きで使用
  * 摂動の割合（1%, 2%）は **固定値** として実装してください（設定項目にはしない）。

---

## 5. 評価指標の追加

### Q5-1: `compute_errors` 関数の仕様

返り値は `dict` で統一します：

```python
def compute_errors(
    z: np.ndarray,
    f_true: np.ndarray,
    f_hat: np.ndarray,
    quantile_ps: list[float] = [0.9, 0.99, 0.999],
    tail_weight_p0: float = 0.9,
) -> dict:
    """
    戻り値:
      {
        "linf_pdf": float,
        "linf_cdf": float,
        "quantiles_true": {p: q_true, ...},
        "quantiles_hat":  {p: q_hat, ...},
        "quantile_abs_errors": {p: abs(q_true - q_hat), ...},
        "tail_l1_error": float,
      }
    """
```

### Q5-2: 分位点の計算方法

* CDF は両方について

```python
F_true = pdf_to_cdf_trapz(z, f_true)
F_true = np.maximum.accumulate(F_true)
F_true /= F_true[-1]
# f_hat も同様
```

* 分位点は `np.interp` を使います：

```python
q_true = {p: np.interp(p, F_true, z) for p in quantile_ps}
q_hat  = {p: np.interp(p, F_hat,  z) for p in quantile_ps}
```

### Q5-3: 右裾重み付きL1誤差

* `tail_weight_p0` をパラメータとし、デフォルト `0.9`。
* `q_true_p0 = np.interp(tail_weight_p0, F_true, z)` を計算し、
  その値以上の領域で

```python
mask = z >= q_true_p0
tail_l1_error = np.trapezoid(np.abs(f_true[mask] - f_hat[mask]), z[mask])
```

を計算してください。

### Q5-4: 評価指標の出力形式

* `main.py` では、各手法の実行後に `compute_errors` を呼び、その結果を

  * 人間が読める形で `print`（標準出力）
  * 可能なら `metrics` dict をまとめて JSON でファイル保存（任意。既存の出力仕様に合わせて実装）

形式は任せますが、少なくとも標準出力で `linf_pdf`, `linf_cdf`, `quantile_abs_errors`, `tail_l1_error` を見えるようにしてください。

---

## 6. 数値安定化（ρ→±1）

### Q6-1: `eps_rho` の値

* 推奨値：`eps_rho = 1e-12` を `em_method.py` 内で定数として使用してください。

### Q6-2: `rho` のクリップの仕方

* 明示的に `rho` をクリップするのではなく、**計算時に**

```python
rho2 = rho * rho
delta = max(1.0 - rho2, eps_rho)
sy_given_x = sy * np.sqrt(delta)
sy_given_x = max(sy_given_x, SIGMA_FLOOR)  # gmm_utils.SIGMA_FLOOR
```

という形で処理してください。

### Q6-3: `std_floor` の値

* `std_floor` は `gmm_utils.SIGMA_FLOOR` をそのまま使ってください（`1e-12`）。
* 別の floor 定数は導入しません。

---

## 7. 後方互換性

### Q7-1: 既存configの扱い

* `method="lp"` の場合：

  * 既存の `K`, `L` パラメータの意味は **変更しません**。
  * 新しい辞書用パラメータ `dict_J`, `dict_L` が指定されていない場合：

    * `dict_J = K`
    * `dict_L = L`
* `method="hybrid"` の場合：

  * `lp_params.dict_J`, `lp_params.dict_L` がなければ

    * `dict_J = 4 * K`（経験則：辞書は最終成分数の数倍）
    * `dict_L = L`
      として内部で補う形にしてください。

### Q7-2: `objective_mode` の設計

* 既存の `objective_mode="pdf"`, `"moments"` はそのまま残します。
* 新しく `objective_mode="raw_moments"` を **追加** します。
* `"moments"` の挙動は変更しません（実装済みの中心モーメント近似用）。

### Q7-3: 既存関数の動作

* `objective_mode` が `"raw_moments"` **以外** の場合は、既存のコードパスをそのまま通るようにしてください（if/elif 分岐で明示）。

---

## 8. テスト計画

### Q8-1: テストケース

代表的なテストセット（例）を想定しておいてください：

* 相関係数 ρ：

  * `rho in [0.0, 0.5, 0.9, 0.99, -0.9, -0.99]`
* 平均・分散の組：

  * `(mu_x, sigma_x, mu_y, sigma_y)` の代表例を 3〜5 パターン

    * 例: `(0, 1, 0, 1)`, `(0, 1, 2, 1)`, `(0, 1, 0, 2)` 等
* GMM成分数：

  * `K in [5, 10, 15]`
* 辞書テール重み：

  * `tail_alpha in [1.0, 2.0, 3.0]`

### Q8-2: 受け入れ条件

* `objective_mode="raw_moments"` で `linprog` の結果 `status == 0`（`res.success == True`）であること。
* テスト用に設定したパラメータセットで、LP が全部成功すること（fallback を含めて）。
  それでも infeasible の場合はテストを **失敗** と見なして構いません。

---

## 9. その他

### Q9-1: エラーハンドリング

* LP が 3 回の fallback 後も `success == False` のとき：

  * `RuntimeError("LP infeasible even after relaxing pdf_tolerance")` を投げてください。
* Hybrid 法で LP が失敗した場合：

  * そのまま例外を投げて構いません（暗黙に EM のみにフォールバックはしない）。

### Q9-2: ログ出力

* `diagnostics` dict に、可能であれば次の情報を入れてください：

  * `"lp_runtime_sec"`
  * `"em_runtime_sec"`
  * `"qp_runtime_sec"`（あれば）
  * `"n_nonzero"`, `"t_pdf"`, `"raw_abs_err"` など
* 実際にどこまで `print` するかは `main.py` 側の判断に任せます。

### Q9-3: ドキュメント更新

* `README.md` には少なくとも：

  * `objective_mode="raw_moments"` の説明
  * `method="hybrid"` の概要
  * 辞書の `tail_focus` / `tail_alpha` の挙動
  * 新しい評価指標（CDF誤差・分位点誤差・右裾L1誤差）
* `method_recommendations.md` には、新方式の結果を既存の表に追加する形で追記してください（フォーマットは現状に合わせる）。

---

以上です。この方針で実装してもらえれば、設計書と実装の齟齬はほぼ出ないはずです。
