# 1. generate_dataset.py（データ生成）の詳細

## 1-1. CLI 引数

提示の引数で **問題ありません**。確定します。

例：

```bash
python -m ml_init.generate_dataset \
  --output_dir ./ml_init/data \
  --n_train 80000 --n_val 10000 --n_test 10000 \
  --seed_train 0 --seed_val 1 --seed_test 2 \
  --z_min -8 --z_max 8 --n_points 64
```

## 1-2. パラメータ範囲（μ, σ, ρ）

* **スクリプト内ハードコードでOK**（まず固定して再現性優先）
* ただし将来変更しやすいように、`generate_dataset.py` 内に

  * `MU_RANGE = (-3.0, 3.0)`
  * `SIGMA_RANGE = (0.3, 2.0)`
  * `RHO_RANGE = (-0.99, 0.99)`
    を **定数として明示**してください（“魔法の数”として散らさない）。

---

# 2. 既存コードとの統合時の import パス / 配置場所【最優先：確定】

## 2-1. import 文

提示の import は **OK**。確定します。

```python
from ml_init.infer import MDNInitError, mdn_predict_init
```

## 2-2. `ml_init` の配置場所（どこに置くか）

**結論：`ml_init` は “src 配下のパッケージ” として配置してください。**

* リポジトリが `src/` レイアウトの場合は：

  * **`src/ml_init/`** に置く（推奨・確定）
  * そうすると `em_method.py` が `src/...` 配下にある前提でも import が安定します

### 確定レイアウト（推奨）

```
src/
  gmm_fitting/
    em_method.py
    ...
  ml_init/
    __init__.py
    infer.py
    ...
```

この形なら `PYTHONPATH` の調整無しで `from ml_init...` が通る構成にできます。

> もし現プロジェクトが `src/` を使っていない（ルート直下にモジュールがある）場合は、同じ階層に `ml_init/` を置く必要がありますが、今回の統合方針は **src 配下で統一**とします。

---

# 3. PyTorch の依存関係【最優先：確定】

## 3-1. requirements の分離方針

**結論：`requirements-ml.txt` を新設**してください。
理由：PyTorch は環境（CPU/GPU）やインストール手順がプロジェクトによって重くなりやすいので、コア依存から分離します。

* 既存 `requirements.txt`：現状維持（PyTorch を入れない）
* 新設 `requirements-ml.txt`：ML 周りのみ

## 3-2. PyTorch のバージョン指定

**固定方針：`torch>=2.0.0`**（上限は付けない）で確定します。

`requirements-ml.txt` 例：

```
torch>=2.0.0
```

> もし社内環境で特定バージョン固定が必要になった時だけ、`torch==2.x.y` に切り替えてください。まずは下限のみで十分です。

---

# 4. テストの有無

* **最小限のユニットテストは作成してください（推奨）**
* 置き場所：既存が `tests/` なら **そこに追加**（`tests/test_mdn_init.py` 等）
* 追加する最小テスト：

  * `mdn_predict_init` が shape と制約（πの和=1、var>=floor）を満たす
  * `wkmeanspp` が空クラスタ処理を含めて落ちない

---

# 5. ドキュメント

* **README.md に短い追記（必須）**：

  * `init="mdn"` の使い方
  * model_path の指定方法（config/env/default）
* `docs/MDN_INIT_GUIDE.md` は任意（後回しでOK）

---

# 6. export.py の役割

**当面は “未使用でもOK”**。役割は次に固定します：

* 目的：チェックポイント（`.pt`）と `metadata.json` を **1つの配布単位に整えるユーティリティ**

  * 例：モデル名の自動付与（`mdn_init_v1_N64_K5.pt`）
  * metadata の整形・検証
* **ONNX 変換はしない**（将来拡張）

---

# 7. 学習データ生成の実行タイミング

* **手動実行でOK**（まずは手順を README に書く）
* make 追加は任意。余裕があれば：

  * `make mdn-generate-data`
  * `make mdn-train`
    を追加しても良い

---

# 8. モデルファイルの Git 管理

* `.gitignore` に以下を追加（確定）：

  * `ml_init/checkpoints/*.pt`
  * `ml_init/data/*.npz`

理由：サイズが大きくなりやすく、再生成可能だからです。
（metadata.json はコミットしてもよいが、チェックポイントとセットで運用するなら ignore してもよい。運用方針に合わせてどちらでも可。）

---

## 最終結論（あなたの優先項目）

* **(2) 配置**：`ml_init` は **`src/ml_init/`** に置く（パッケージとして import 安定）
* **(3) 依存**：PyTorch は **`requirements-ml.txt`** に分離し、**`torch>=2.0.0`**

この確定仕様で実装に進めてください。
