# 1. MDNInitError の定義場所【最優先：確定】

## 決定

* **新規ファイル `ml_init/infer.py` に定義**してください。
* 例外は **MDN 推論（モデルロード〜初期値生成）に閉じた責務**なので、EM本体（`em_method.py`）に混ぜません。

### 例外クラス仕様（決定）

`ml_init/infer.py` に以下を置く：

```python
class MDNInitError(RuntimeError):
    """Raised when MDN-based initialization fails (load/version/device/numerics)."""
    pass
```

### 使い方（決定）

* `mdn_predict_init(...)` は MDN関連の失敗を **必ず `MDNInitError` で投げる**
  （FileNotFound / version mismatch / NaN / shape mismatch など）
* `em_method.py` 側（init="mdn" の分岐）で `MDNInitError` を捕捉し、**fallback チェーンへ移行**する。

> 例外の import：`from ml_init.infer import MDNInitError, mdn_predict_init`

---

# 2. エポックごとの評価指標（val サブセット）の選び方【確定】

## 決定

* **val の先頭から固定で 1024 件**（不足なら全件）を使います。
* **毎エポック同じサブセット**です（学習の進行比較が安定するため）。

理由：ランダム抽出はノイズが増え、学習の比較がぶれます。固定サブセットが実務向きです。

---

# 3. チェックポイントの保存内容【確定】

## 決定

* **モデルは `state_dict` のみ保存**（`mdn_init_v1_N64_K5.pt`）。
* optimizer state は **保存しません**（再開学習が必要になったら拡張で対応）。
* epoch 数 / best_val_loss などのメタ情報は **`metadata.json` に含める**。

### metadata.json に必須で入れる（決定）

* `version, N_model, K_model, z_min, z_max, sigma_min, reg_var, input_transform`
* `train_args`（CLI引数一式）
* `best_epoch, best_val_ce`
* `created_at`（任意）

---

# 4. 学習スクリプト train.py CLI【確定】

提示いただいた形でOKです。

例：

```bash
python -m ml_init.train \
  --data_dir ./ml_init/data \
  --output_dir ./ml_init/checkpoints \
  --batch_size 256 \
  --lr 1e-3 \
  --epochs 20 \
  --lambda_mom 0.0
```

---

# 5. 評価スクリプト eval.py CLI【確定】

提示いただいた形でOKです。

例：

```bash
python -m ml_init.eval \
  --model_path ./ml_init/checkpoints/mdn_init_v1_N64_K5.pt \
  --data_path ./ml_init/data/test.npz \
  --output_path ./ml_init/eval_results.json
```

---

# 6. ファイル構造【最優先：確定】

## 決定

提示の構造で進めてOKです。
ただし **`metadata.json` は checkpoints 配下に置き、モデルと対に**してください。

確定構造：

```
ml_init/
  __init__.py
  dataset.py
  model.py
  train.py
  eval.py
  infer.py
  metrics.py
  export.py
  wkmeanspp.py
  data/
    train.npz
    val.npz
    test.npz
  checkpoints/
    mdn_init_v1_N64_K5.pt
    metadata.json
```

> 例外クラス MDNInitError は `infer.py` に置く（§1）。

---

# 7. 学習時ログ出力【確定】

提示のログ項目でOKです。

* epoch
* train_loss（CE）
* val_loss（CE）
* val_metrics（PDF L∞、CDF L∞、分位点誤差）※固定サブセット
* epoch_time_sec（学習時間）

---

# 8. 推論時デバイス指定【確定】

* `"auto"` は **`torch.cuda.is_available()`** で判定して問題ありません。

  * True → `"cuda"`
  * False → `"cpu"`

---

## 実装開始OKの最終結論

* **MDNInitError は `ml_init/infer.py` に定義**
* **ディレクトリ構造は提示どおりで確定（metadata.json は checkpoints 内）**

この仕様で実装に進めてください。
