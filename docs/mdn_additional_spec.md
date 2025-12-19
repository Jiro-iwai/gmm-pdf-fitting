# 1. データ生成と学習データセット

## 1-1. 学習サンプル数（想定）

* **合計 100,000 サンプル**を標準とします。

  * train: **80,000**
  * val: **10,000**
  * test: **10,000**

理由：モデルが小さい（2層MLP）ので学習が軽く、分布パラメータ空間（(\mu_x,\sigma_x,\mu_y,\sigma_y,\rho)）を広めにカバーするには 10k だと不足しやすい一方、100k は保存サイズも小さく現実的です（(100000\times 64) 程度）。

## 1-2. 分割比率

* **80 / 10 / 10** で固定します。

## 1-3. 生成方式（毎回 or 事前生成）

* **事前生成して保存**を基本方針にします（再現性のため）。
* 生成スクリプト `generate_dataset.py` を用意し、以下を固定：

  * 乱数 seed（例：train=0、val=1、test=2）
  * パラメータ範囲（(\mu,\sigma,\rho) のレンジ）
  * グリッド（(N=64, z_{\min}, z_{\max})）

保存形式は `.npz` 推奨：

* `z` : shape ((64,)) float32（固定グリッド）
* `f` : shape ((S,64)) float32（正規化済み pdf）
* `params` : shape ((S,5)) float32（(\mu_x,\sigma_x,\mu_y,\sigma_y,\rho) など）
* `meta` : JSON文字列（レンジ、seed、生成コードのバージョン等）

> 追加（任意）：学習時に **on-the-fly で少量の拡張**（ノイズなど）を入れたい場合でも、まずは保存データのみで学習して評価するのを優先してください。

---

# 2. グリッドサイズの扱い（入力が (N\neq 64) の場合）

## 2-1. 補間方法

* **線形補間（piecewise linear interpolation）で固定**します。

  * スプライン補間は **オーバーシュートで負の pdf が出うる**ため採用しません。

入力グリッド ({z^{\mathrm{in}}*j}*{j=1}^{N_{\mathrm{in}}})、入力 pdf ({f^{\mathrm{in}}*j}) を
モデル固定グリッド ({\bar z_i}*{i=1}^{64}) に写す：

[
\bar f_i = \mathrm{interp}!\left(\bar z_i;\ z^{\mathrm{in}},\ f^{\mathrm{in}}\right)
]

範囲外は 0 とします（外挿はしない）。

## 2-2. 補間後の正規化

* **`normalize_pdf_on_grid` を使用して問題ありません。**
* 追加で安全策：

  * (\bar f_i \leftarrow \max(\bar f_i, 0)) を適用（負値を潰す）
  * その後 `normalize_pdf_on_grid( z_fixed, f_resampled )`

---

# 3. (\sigma_{\min}) の値（softplus + sigma_min）

結論：**EM の `reg_var` と統一**します。ただし `reg_var` は「分散（variance）」で、(\sigma_{\min}) は「標準偏差（standard deviation）」なので、

[
\sigma_{\min} = \sqrt{\mathrm{reg_var}}
]

とします。

* 既存 EM のデフォルト：(\mathrm{reg_var}=10^{-6})
* よってデフォルト：(\sigma_{\min}=10^{-3})

MDN の出力変換は以下で固定：

[
\sigma_k = \log(1+e^{\beta_k}) + \sigma_{\min}
]
[
\sigma_k^2 = (\sigma_k)^2
]

この設計で、EM 側での分散下限（`reg_var`）と矛盾しません。

---

# 4. (\mu) のソート処理（同一 (\mu) が出た場合）

## 4-1. ソートキー（決定）

成分の並び固定は **次の優先順位**で決めます：

1. (\mu) 昇順
2. (\pi) 降順（同じ (\mu) なら重い成分を先）
3. (\sigma) 昇順（さらに同率なら狭い成分を先）

つまり、辞書式順序として

[
(\mu,\ -\pi,\ \sigma)
]

でソートします。

Numpy 実装例（lexsort のキー順に注意）：

* `idx = np.lexsort((sigma, -pi, mu))`
  （最後の `mu` が主キーになります）

---

# 5. モーメントペナルティ

## 5-1. (\lambda_{\mathrm{mom}}) のデフォルト

* **デフォルトは (\lambda_{\mathrm{mom}} = 0)** で問題ありません。
* まずは **CE（クロスエントロピー）単体**で学習→評価し、必要なら追加します。

## 5-2. モーメントの計算（流用の可否）

学習は PyTorch を想定しているため、既存の numpy 実装（`compute_pdf_statistics` 等）をそのまま学習損失に流用はできません（微分できないため）。

モーメントペナルティを入れる場合は、**torch 上で raw moment（生モーメント）**を使ってください。
（中心モーメントより実装が簡単で安定です）

グリッド上の生モーメント（台形則）：

[
M_n(f) \approx \sum_{i=1}^{64} z_i^n f(z_i) w_i,\quad n=1,2,3,4
]

ペナルティ：

[
\mathcal L_{\mathrm{mom}} = \sum_{n=1}^{4}\left(M_n(\hat f_\theta)-M_n(f)\right)^2
]

---

# 6. モデルの保存とバージョン管理

## 6-1. 保存形式

* **`.pt`（PyTorch `state_dict`）**で固定
* 併せて **`metadata.json`**を保存（同名で OK）

## 6-2. metadata の必須項目

互換性判定のため、最低限これを入れてください：

* `version`: 例 `"mdn_init_v1"`
* `N_model`: 64
* `K_model`: 5（※Kを変えたら別モデル）
* `z_min`, `z_max`
* `sigma_min` と、それを決めた `reg_var`
* `input_transform`: `"pdf"` か `"logpdf"`（今回は `"pdf"`）
* `param_ranges`: 学習時の (\mu,\sigma,\rho) 範囲
* `seed_info`: train/val/test の seed

## 6-3. 互換性の方針

* **K が違うモデルは互換なし**（ロード時に例外）
* **入力 N は任意でよい**（推論時に 64 に resample するため）

---

# 7. 既存コードとの統合（init の扱い・優先順位）

## 7-1. init の設計（決定）

* `fit_gmm1d_to_pdf_weighted_em` に **`init="mdn"` を新設**してください。
* ただし内部実装は、既存の `init="custom"` 経由で統合します：

手順：

1. `init=="mdn"` の場合 `mdn_predict_init(...)` を呼ぶ
2. `init_params={"pi":..., "mu":..., "var":...}` を生成
3. 既存の `init=="custom"` と同じコードパスで EM を開始

こうすると既存の `custom` 実装を最大限使い回せます。

## 7-2. MDN 初期化の優先順位（決定）

推論時の優先順位は以下で固定：

1. **MDN**
2. （MDN が失敗 or 閾値超過なら）**wkmeanspp**（後述：新規実装）
3. それもダメなら **wqmi**（既存）
4. 最後に **quantile**（既存）

---

# 8. フォールバックの閾値（最重要）

## 8-1. 指標：クロスエントロピー（CE）の定義

ターゲット pdf (f(z)) と候補 pdf (g(z)) に対して

[
\mathcal L_{\mathrm{CE}}(f,g) = -\int f(z)\log g(z),dz
]

離散化（台形則）：

[
\mathcal L_{\mathrm{CE}}(f,g) \approx -\sum_{i=1}^{64} f(z_i)\log(g(z_i)+\varepsilon),w_i
]

* (\varepsilon = 10^{-12})
* (f) と (g) はどちらも **`normalize_pdf_on_grid` で正規化**してから評価

## 8-2. 閾値の決め方（決定：相対比較）

絶対閾値は分布レンジで変動するため採用しません。
**同一入力に対して WQMI 初期化の CE と比較**します。

[
\mathcal L_{\mathrm{CE}}^{\mathrm{MDN}} > (1+\delta),\mathcal L_{\mathrm{CE}}^{\mathrm{WQMI}}
\Rightarrow \text{fallback}
]

* **(\delta = 0.2)**（= 20% 悪ければフォールバック）

加えて、以下は無条件 fallback：

* (\exists i) で (g(z_i)) が NaN / Inf
* (\min_i g(z_i) < 0)（理論上ないがバグ検知）
* (\sum_i g(z_i)w_i \le 0)（正規化不能）

## 8-3. wkmeanspp の実装有無

* **既存にはない前提で新規実装**してください。
* ただし「最小実装で急ぐ」場合は、当面 `wkmeanspp` を飛ばして `wqmi` へ落としてもよいです。
  ただし今回の優先仕様では **wkmeanspp を入れる**のが第一推奨です。

---

# 9. 推論時のデバイス（CPU/GPU）とバッチ推論

## 9-1. デバイス方針

* **デフォルトは CPU**（モデルが小さいので GPU の起動オーバーヘッドが勝ちやすい）
* 設定で変更可能にする：

  * `device="cpu" | "cuda" | "auto"`
  * `auto` は `cuda` があれば `cuda`、なければ `cpu`

## 9-2. バッチ推論

* 実運用（SSTAの1ケース処理）では **単一サンプル推論で十分**
* ただし評価用に `mdn_predict_init_batch(f_batch)` は用意して良い（任意）

---

# 10. 評価とベンチマーク

## 10-1. 統合先

* 学習は別スクリプト：`ml_init/train.py`
* 評価は 2 つ用意：

  1. `ml_init/eval.py`（モデル単体：CE、PDF/CDF誤差、分位点誤差）
  2. 既存の `main.py` / 既存ベンチに **`init="mdn"` を追加**して、EM込みの性能比較

## 10-2. 比較対象（baseline）

* baseline はご提案通りで OK：

  * **WQMI + n_init=8**（現状の基準）
* 追加で（推奨）：

  * quantile init（n_init=8）
  * wqmi init（n_init=1）も併記（「多スタートの寄与」を切り分けるため）

---

## まとめ（優先度の高い確定事項）

* **(1)** データ：総 **100k**、**80/10/10**、**事前生成して保存**
* **(3)** (\sigma_{\min}=\sqrt{\mathrm{reg_var}})（reg_var=1e-6 → sigma_min=1e-3）
* **(7)** `init="mdn"` を新設し、内部は `custom` 経由で統合。優先順位は **MDN→wkmeanspp→wqmi→quantile**
* **(8)** fallback 閾値：
  [
  \mathcal L_{\mathrm{CE}}^{\mathrm{MDN}} > 1.2,\mathcal L_{\mathrm{CE}}^{\mathrm{WQMI}}
  \Rightarrow \text{fallback}
  ]
  wkmeanspp は **新規実装**

この仕様で実装に進めてください。
