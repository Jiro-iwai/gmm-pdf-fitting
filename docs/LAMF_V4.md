# LAMF V4 モデル仕様書

## 概要

LAMF (Learned Accelerated Mixture Fitter) は、PDF（確率密度関数）から GMM（ガウス混合モデル）パラメータを直接推定するニューラルネットワークです。従来の EM アルゴリズムの反復最適化を、学習済みの固定ステップ数の refinement に置き換えることで、高速かつ安定した推定を実現します。

V4 モデルは、複数の安定化手法と過学習対策を組み込んだ最終版です。

---

## アーキテクチャ

### 全体構成

```
Input PDF (N=96) → InitNet → RefineBlock × T → GMM Parameters (K=5)
                              ↑
                         Deep Supervision
```

### 1. InitNet（初期パラメータ生成）

- **入力**: 正規化された PDF `w` (shape: `[batch, N]`)
- **出力**: unconstrained パラメータ `(α, c, β, γ)` (shape: `[batch, K, 4]`)
- **構造**: 
  - 3層 MLP（256次元）
  - ReLU 活性化 + Dropout (0.2)
  - 出力を `ParameterTransform.project()` で制約付きパラメータに変換

### 2. RefineBlock（反復的パラメータ更新）

- **入力**: 現在の GMM パラメータ `(π, μ, σ)` と入力 PDF
- **出力**: 更新された GMM パラメータ
- **構造**:
  - EM-like 十分統計量の計算
  - UpdateNet（2層 MLP, 128次元）による補正量予測
  - EMA blend: `param_new = (1-τ)*param + τ*param_em + corr`
- **重要な安定化手法**:
  - `sigma_min`, `sigma_max` によるクランプ
  - `corr_scale * tanh(corr)` による補正量の有界化
  - 学習可能な `τ` (blend ratio) と `λ` (EM strength)

### 3. ParameterTransform（制約変換）

Unconstrained パラメータから制約付きパラメータへの変換：

| パラメータ | 変換 | 制約 |
|-----------|------|------|
| `π` (mixing weights) | `softmax(α)` | `Σπ = 1`, `π ≥ pi_min` |
| `μ` (means) | 累積 softplus | 単調増加順序 |
| `σ` (std devs) | `softplus(γ) + sigma_min` | `σ ∈ [sigma_min, sigma_max]` |

---

## V4 モデルパラメータ

### モデル構成

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `N` | 96 | 入力 PDF グリッド点数 |
| `K` | 5 | GMM コンポーネント数 |
| `T` | 6 | RefineBlock の反復回数 |
| `init_hidden_dim` | 256 | InitNet 隠れ層次元 |
| `init_num_layers` | 3 | InitNet 層数 |
| `refine_hidden_dim` | 128 | RefineBlock 隠れ層次元 |
| `refine_num_layers` | 2 | RefineBlock 層数 |
| `share_refine_weights` | true | RefineBlock 重み共有 |
| `dropout` | 0.2 | Dropout 率 |

### 制約パラメータ

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `sigma_min` | 0.01 | 最小標準偏差 |
| `sigma_max` | 5.0 | 最大標準偏差 |
| `pi_min` | 0.0 | 最小混合重み |
| `corr_scale` | 0.5 | 補正量スケール |

### データ仕様

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `z_min` | -15.0 | グリッド範囲（最小） |
| `z_max` | 15.0 | グリッド範囲（最大） |
| 学習データ | 150,000 | V5 データセット |
| 検証データ | 15,000 | |
| テストデータ | 15,000 | |

---

## 学習条件

### 最終学習パラメータ（V4）

```bash
python -m src.lamf.train \
  --data_dir ml_init/data_v5 \
  --output_dir lamf/checkpoints_v4 \
  --epochs 50 \
  --batch_size 64 \
  --lr 1e-3 \
  --lambda_pdf 0.2 \
  --sigma_min 0.01 \
  --sigma_max 5.0 \
  --corr_scale 0.5 \
  --dropout 0.2 \
  --weight_decay 1e-3 \
  --patience 20 \
  --warmup_epochs 5 \
  --eta_schedule final_only \
  --scheduler_type cosine \
  --grad_clip 1.0 \
  --num_workers 4 \
  --device cpu
```

### ハイパーパラメータ詳細

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `lr` | 1e-3 | 初期学習率 |
| `batch_size` | 64 | バッチサイズ |
| `epochs` | 50 | 最大エポック数 |
| `warmup_epochs` | 5 | 学習率ウォームアップ |
| `scheduler_type` | cosine | 学習率スケジューラ |
| `lambda_pdf` | 0.2 | PDF L2 損失の重み |
| `weight_decay` | 1e-3 | L2 正則化 |
| `dropout` | 0.2 | Dropout 率 |
| `grad_clip` | 1.0 | 勾配クリッピング |
| `patience` | 20 | Early stopping patience |
| `eta_schedule` | final_only | Deep supervision スケジュール |

---

## 損失関数

### 構成

```
Loss = (1 - λ_pdf) * CE_loss + λ_pdf * PDF_L2_loss
```

### Cross-Entropy 損失（CE）

離散化された負対数尤度：

```
CE = -Σ w[i] * log(Σ π_k * N(z[i]; μ_k, σ_k))
```

### PDF L2 損失

予測 PDF と目標 PDF の L2 ノルム：

```
PDF_L2 = Σ (f_pred[i] - f_target[i])² * Δz
```

### Deep Supervision

最終出力のみに損失を適用（`eta_schedule = final_only`）

---

## 学習履歴

### Early Stopping

- **最良エポック**: 13
- **停止エポック**: 33
- **学習時間**: 123.1 分

### 学習曲線

| エポック | Train CE | Val CE | Val PDF L∞ | Val CDF L∞ |
|---------|----------|--------|------------|------------|
| 1 | 1.1751 | 1.0612 | 1.6101 | 0.7394 |
| 5 | 1.2512 | 1.2654 | 0.0153 | 0.0079 |
| 13 (best) | 1.2553 | 1.2640 | **0.0120** | **0.0048** |
| 33 (stop) | 1.2548 | 1.2629 | 0.0437 | 0.0170 |

---

## テスト性能

### 最終評価

| 指標 | 値 |
|------|-----|
| Test CE | 1.2570 |
| **Test PDF L∞** | **0.0121** |
| **Test CDF L∞** | **0.0048** |

### EM 法との比較

| 手法 | PDF L∞ Mean | Time Mean |
|------|-------------|-----------|
| **LAMF V4** | **0.0087** | **1.94 ms** |
| EM + Quantile (n=1) | 0.0098 | 15.25 ms |
| EM + MDN | 0.0121 | 74.23 ms |
| EM + LAMF (init) | 0.0097 | 70.54 ms |

**LAMF V4 は EM + Quantile より 11% 高精度、7.9倍高速**

---

## 開発履歴と試行錯誤

### Phase 1: 基本実装

- InitNet + RefineBlock の基本構造を実装
- CE 損失のみで学習
- 問題: 学習が不安定、CDF L∞ > 1 が頻発

### Phase 2: PDF L2 損失の追加

- `lambda_pdf` パラメータを導入
- `lambda_pdf=0.5` で学習崩壊を確認
- `lambda_pdf=0.2` で安定化、精度向上

### Phase 3: 安定化手法の導入

外部 AI レビュー（`docs/lamf_check.md`）に基づく改善：

1. **`sigma_max` クランプ**: `log_sigma_blend` の上限設定で exp overflow 防止
2. **`corr_scale * tanh(corr)`**: 補正量の有界化
3. **NaN/Inf バッチスキップ**: 異常な loss を持つバッチをスキップ

### Phase 4: 過学習対策（V4）

- `dropout=0.2` と `weight_decay=1e-3` を追加
- `sigma_min=0.01` に戻して表現力を維持
- `eta_schedule=final_only` で Deep supervision を最終出力のみに

### 失敗した試み

| 試行 | 結果 | 問題 |
|------|------|------|
| `lambda_pdf=0` (CE only) | 崩壊 | σが極小化、π=0 問題 |
| `lambda_pdf=0.9` | 部分的成功 | EM 初期化時の性能低下 |
| `sigma_min=0.1` | 低精度 | 鋭いPDFを表現できない |
| EMA (`ema_decay=0.999`) | 低精度 | 収束が遅すぎる |
| `lambda_pdf` curriculum | 過学習 | λ増加時に不安定化 |

---

## 使用方法

### 学習

```bash
python -m src.lamf.train \
  --data_dir ml_init/data_v5 \
  --output_dir lamf/checkpoints_v4 \
  --epochs 50 \
  --lr 1e-3 \
  --lambda_pdf 0.2 \
  --dropout 0.2 \
  --weight_decay 1e-3
```

### 推論

```python
from src.lamf.infer import fit_gmm1d_to_pdf_lamf

result = fit_gmm1d_to_pdf_lamf(
    z=z,  # shape: (N,)
    f=f,  # shape: (N,)
    model_path="lamf/checkpoints_v4",
    device="cpu",
)

pi = result["pi"]      # shape: (K,)
mu = result["mu"]      # shape: (K,)
var = result["var"]    # shape: (K,)
```

### EM 初期化として使用

```python
from src.gmm_fitting.em_method import fit_gmm1d_to_pdf_weighted_em

params, ll, n_iter = fit_gmm1d_to_pdf_weighted_em(
    z, f,
    K=5,
    init="lamf",
    lamf_model_path="lamf/checkpoints_v4",
)
```

### Web UI

1. Method: "EM Algorithm" を選択
2. Initialization: "LAMF (Neural Network)" を選択
3. K は自動的に 5 に設定される

または

1. Method: "LAMF (Neural Network)" を直接選択

---

## ファイル構成

```
lamf/
├── checkpoints_v4/
│   ├── best_model.pt        # モデル重み
│   ├── metadata.json        # モデル設定
│   └── training_curve.png   # 学習曲線
├── training_v4.log          # 学習ログ
└── ...

src/lamf/
├── __init__.py
├── model.py      # LAMFFitter, InitNet, RefineBlock
├── train.py      # 学習スクリプト
├── infer.py      # 推論 API
├── dataset.py    # データセットクラス
├── metrics.py    # 評価指標
└── eval.py       # 評価・比較スクリプト
```

---

## 制限事項と注意点

1. **K=5 固定**: V4 モデルは K=5 でのみ学習されています
2. **グリッドサイズ**: 入力 PDF は N=96, z∈[-15,15] に自動リサンプリングされます
3. **相対座標系**: 入力 PDF は平均 0 に正規化されます
4. **σ の範囲**: σ_min=0.01 より小さい成分は表現できません

---

## 参考文献

- Issue #7: LAMF 設計提案
- `docs/lamf_check.md`: 外部 AI による安定性レビュー
- `docs/lamf_improve.md`, `docs/lamf_improve2.md`: 最適化提案

