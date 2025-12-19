## 実装順序について

* 先に `wkmeanspp.py` / `metrics.py` を作る → 後段（train/eval/infer/fallback）が全部そこで使える
* `infer.py` を「最後」にするのも良い（checkpoint仕様・model/metadataの確定が先なので）

一点だけ微調整するなら（任意）：
**infer.py は em_method.py 統合の直前**に置くと、import/例外/metadata検証の整合確認がしやすいですが、今の順序でも問題ありません。

---

## 実装時の注意点への回答

### 1) 数値安定性

#### 1-1. log-sum-exp 前の NaN チェックは必要？

**不要（やらない）でOK**です。
理由：`max` を取る前に NaN が混入しているなら、その時点でモデル出力か入力が壊れているので、後段でまとめて検知して `MDNInitError` に落とす方が簡潔です。

**決定：**

* `log_gmm_pdf` 内で **NaN/Inf の検知は最後に1回だけ**行い、異常なら例外を投げる
  例：`if not torch.isfinite(log_fhat).all(): raise MDNInitError(...)`

#### 1-2. softplus のオーバーフロー対策は必要？

**基本は不要**です。PyTorch の `F.softplus` は数値的に安定な実装になっています。
**決定：** `torch.nn.functional.softplus` を使う（自前実装しない）。

---

### 2) エラーハンドリング

#### 2-1. mdn_predict_init のエラーメッセージは具体的に出す？

**はい、必ず具体的に出してください（確定）**。

`MDNInitError` の message には最低限：

* `model_path`
* 失敗種別（FileNotFound / JSON decode / version mismatch / K mismatch / N mismatch / non-finite output）
* version mismatch の場合：

  * expected: `version, N_model, K_model`
  * got: `version, N_model, K_model`

例（方針）：

* `MDNInitError(f"MDN init failed: version mismatch. expected=(v1,N64,K5) got=(v0,N64,K10) path=...")`

---

### 3) 既存コードとの互換性

#### 3-1. fit_gmm1d_to_pdf_weighted_em の引数追加で既存呼び出しに影響する？

**影響しない（確定）**です。

**決定：**

* 新規引数は末尾に追加し、デフォルト値を必ず付ける：

  * `mdn_model_path: str | None = None`
  * `mdn_device: str = "auto"`

既存の呼び出しコードは引数追加の影響を受けません。

---
