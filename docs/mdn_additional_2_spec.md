# 追加確認事項への回答（仕様確定）

## 1. wkmeanspp（重み付き k-means++）の実装詳細【最優先】

### 1-1. 重みの定義

**はい、重みは (f(z_i)\times w_i) でOK**です。

* (f(z_i))：グリッド上のpdf値（補間後・非負化後）
* (w_i)：積分重み（台形則を推奨）

  * 等間隔 (\Delta z) のとき
    [
    w_0=\frac{\Delta z}{2},\quad w_{N-1}=\frac{\Delta z}{2},\quad w_i=\Delta z\ (1\le i\le N-2)
    ]
* 重みベクトル：
  [
  \omega_i = \max(f(z_i),0),w_i
  ]
* k-means++ の確率に使うために正規化：
  [
  p_i=\frac{\omega_i}{\sum_j \omega_j}
  ]
  （※ (\sum_j\omega_j) が 0 の場合は異常系として fallback）

### 1-2. k-means++ 初期化は「重み付き距離」で行うか？

**初期化の選択確率を “重み×距離²” にする（=重み付き k-means++）で固定**します。

* 1個目の中心 (c_1) は (p_i) に比例してサンプル
* 2個目以降（t=2..K）：

  * 各点の最近傍中心への距離
    [
    D_i=\min_{1\le k<t} |z_i-c_k|
    ]
  * 次の中心選択確率
    [
    \Pr(z_i\ \text{を選ぶ})\propto p_i,D_i^2
    ]

> これで「確率密度が高い（重要な）領域」を優先しつつ、中心が分散して選ばれます。

### 1-3. 距離は何を使うか？

**1次元なので距離はユークリッド（絶対値）で固定**します。

[
\text{dist}(z_i,c)=|z_i-c|
]

「重み付きユークリッド距離」は使いません。重みは上の **確率**と **中心更新（平均）**に反映されるため、距離自体を重み付けすると二重に効いて挙動が歪みやすいです。

### 1-4. Lloyd 反復（クラスタ更新）の仕様

* 割当：
  [
  a(i)=\arg\min_k |z_i-c_k|
  ]
* 中心更新（重み付き平均）：
  [
  c_k=\frac{\sum_{i:a(i)=k} p_i z_i}{\sum_{i:a(i)=k} p_i}
  ]
* 反復停止条件：最大 `max_iter=20`、または (\max_k|c_k^{new}-c_k^{old}|<10^{-6})

### 1-5. GMM 初期値（wkmeanspp 出力）の作り方

クラスタ (k) の重み：
[
\pi_k=\sum_{i:a(i)=k} p_i
]
平均：
[
\mu_k=c_k
]
分散（重み付き分散 + 下限）：
[
\sigma_k^2=\frac{\sum_{i:a(i)=k} p_i (z_i-\mu_k)^2}{\pi_k}+\text{reg_var}
]
最終的に (\sum_k \pi_k = 1) になる（p_i を正規化しているため）。

### 1-6. 空クラスタ（(\pi_k=0)）の扱い

空クラスタが出たら、そのクラスタ中心は次で再初期化して続行：

* 既存中心集合への距離 (D_i) を用いて、
  (p_i D_i^2) が最大の点を選び新中心にする
  （最も“重要かつ未カバー”な点を割り当てる）

---

## 2. モデルアーキテクチャの詳細

### 2-1. バイアス

**使用します**（Linear の bias=True）。設計書の (b_1,b_2,b_3) の通り。

### 2-2. 重み初期化

（用語定義）

* **He / Kaiming 初期化**：ReLU 系で分散が保たれるように重みを初期化する方法
* **Xavier 初期化**：入力・出力の分散が釣り合うように初期化する方法

**仕様：**

* ReLU を使う層（W1, W2）：**He(Kaiming) uniform**
* 出力層（W3）：**Xavier uniform**
* bias：ゼロ初期化

### 2-3. 正則化（dropout等）

（用語定義）

* **Dropout**：学習時にランダムにユニットを無効化して過学習を抑える手法

**当面は使いません**（dropout=0）。モデル小＋データ 100k なので不要と判断。

---

## 3. 学習の詳細

* 学習率スケジューラ：**使わない**（固定 lr=1e-3）
* 勾配クリッピング：**使う（max_norm=1.0）**
  ※ log-likelihood 系の損失で稀に勾配が跳ねるのを抑制
* バッチ内のサンプル数差：固定グリッドなので **不要**

---

## 4. チェックポイントと評価

* **各epochで val CE を計算し、最良（最小）モデルを保存**
* 追加の評価指標（PDF L∞、CDF L∞、分位点誤差）は
  **val のサブセット（例：512〜1024件）で epoch ごとに計算（任意・推奨）**
  ※フル val(10k)で毎回やる必要はなし

---

## 5. データローダー

* `num_workers=0` をデフォルト（環境差で詰まりにくい）
* `shuffle=True` は train のみ、val/test は False

---

## 6. 推論時のエラーハンドリング

* `mdn_predict_init` がモデルを読めない／不一致／NaN 等の場合：

  * **例外（MDNInitError）を投げる**
  * 呼び出し側（init="mdn" パス）で捕捉して **fallback チェーンへ移行**

バージョン不一致メッセージは必ず以下を含める：

* expected: `version, N_model, K_model`
* got: `version, N_model, K_model`
* path: `model_path`

---

## 7. 既存コードとの統合（model_path の取得）【最優先】

### 7-1. init="mdn" の追加と model_path の受け渡し

**`fit_gmm1d_to_pdf_weighted_em` に `init="mdn"` を追加**し、`model_path` は以下の優先順で決めます。

**優先順位（決定）：**

1. `main.py` の config から明示指定（最優先）
2. 環境変数 `MDN_MODEL_PATH`
3. デフォルトパス：`./ml_init/checkpoints/mdn_init_v1_N64_K5.pt`

config 例（推奨）：

```json
{
  "init": "mdn",
  "mdn": {
    "model_path": "./ml_init/checkpoints/mdn_init_v1_N64_K5.pt",
    "device": "auto"
  }
}
```

### 7-2. 関数シグネチャ案（決定）

`fit_gmm1d_to_pdf_weighted_em` への追加引数（既存互換を壊さない）：

```python
def fit_gmm1d_to_pdf_weighted_em(
    ...,
    init: str = "wqmi",
    init_params: dict | None = None,
    mdn_model_path: str | None = None,
    mdn_device: str = "auto",
):
    ...
```

* `init=="mdn"` の場合のみ `mdn_model_path` / config / env / default を参照
* 取得できずファイルが存在しない場合は **MDNInitError → fallback**

---

## 8. 学習データ生成の詳細

* パラメータ範囲外：**クリップしない**（歪むので）。範囲内でサンプリングする（=事実上除外）
* (1-\rho^2) の床処理：**両方で実施**

  * 生成スクリプト：(|\rho|\le 0.99) でサンプル
  * `max_pdf_bivariate_normal` 内部：
    [
    1-\rho^2 \leftarrow \max(1-\rho^2,\varepsilon_\rho),\ \varepsilon_\rho=10^{-12}
    ]
    （防御的に二重に入れる）

---

# 実装開始条件（あなたの優先項目の確定）

* **wkmeanspp**：重み (\omega_i=f_i w_i)、選択確率 (\propto p_i D_i^2)、距離は (|z-c|)
* **model_path**：config → env → default の優先順で取得、見つからなければ MDN失敗扱いで fallback

この仕様で実装に進めてください。
