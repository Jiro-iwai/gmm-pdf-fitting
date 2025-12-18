# 実装上の不明点・確認事項

## 1. 生モーメント計算関数

### ✅ 既存実装確認
- `compute_component_raw_moments` は既に `gmm_utils.py` に存在（272行目）
- `compute_pdf_raw_moments` は未実装（新規追加が必要）

### ❓ 不明点

#### Q1-1: `compute_pdf_raw_moments` の返り値形式
設計書では `np.ndarray` とあるが、具体的な形状は？
- **提案**: `shape (max_order+1,)` で `M[0], M[1], ..., M[max_order]` を返す
- **確認**: M[0] は常に1.0を返すべきか、それとも実際の積分値か？

#### Q1-2: PDF正規化のタイミング
設計書では「入力PDFが正規化されていない場合は正規化してから計算」とあるが：
- 既に正規化されているPDFを再度正規化しても問題ないか？
- 正規化の閾値（例：`abs(area - 1.0) < 1e-6`）は？

## 2. 辞書生成の拡張（テール重視分位点）

### ✅ 既存実装確認
- `build_gaussian_dictionary` は既に存在（`lp_method.py` 109行目）
- `mu_mode="quantile"` をサポートしている
- `tail_focus` と `tail_alpha` は未実装

### ❓ 不明点

#### Q2-1: 左裾（left tail）の実装方法
設計書では `tail_focus: "none" | "right" | "left" | "both"` とあるが：
- 左裾重視の場合の分位点レベル生成式は？
  - **提案**: `p_j = u_j^α` （右裾は `1-(1-u_j)^α`）
- `both` の場合の実装方法は？
  - **提案**: 左右対称に配置（例：`p_j = 0.5 + sign(u_j-0.5) * (|u_j-0.5|)^α`）

#### Q2-2: `tail_alpha` のデフォルト値と範囲
- デフォルト値は？
- 範囲制約（`α ≥ 1`）のチェックは必要か？
- `α=1` の場合は通常の分位点（`p_j = u_j`）になるか？

#### Q2-3: `quantile_levels` が指定された場合の `tail_focus` の扱い
- `quantile_levels` が指定されている場合、`tail_focus` と `tail_alpha` は無視するか？
- それとも、`quantile_levels` に `tail_focus` を適用するか？

#### Q2-4: CDFの単調性保証
設計書では「`F` は単調増加になっていること」とあるが：
- `pdf_to_cdf_trapz` は既に単調性を保証しているか？
- 数値誤差で単調性が崩れた場合の処理は？

## 3. LPソルバ `solve_lp_pdf_rawmoments_linf`

### ❓ 不明点

#### Q3-1: 目的関数の方式選択
設計書では方式A（推奨）と方式Bの両方を実装可能とあるが：
- デフォルトは方式Aか？
- 方式を切り替えるパラメータ名は？（例：`objective_form: "A" | "B"`）

#### Q3-2: `lambda_raw` のデフォルト値
設計書では `lambda_raw: [1.0, 1.0, 1.0, 1.0]` の例があるが：
- これが推奨デフォルトか？
- モーメントのスケールが異なる場合（例：M₁は小さい、M₄は大きい）、重みを調整すべきか？

#### Q3-3: `pdf_tolerance` が `None` の場合の扱い
- 方式Aの場合、`pdf_tolerance=None` は許容されるか？
- 方式Bの場合、`pdf_tolerance` は無視されるか？

#### Q3-4: LP変数の順序と制約行列の構築
設計書では変数は `[w_1,...,w_m, t_pdf, t_1, t_2, t_3, t_4]` とあるが：
- `scipy.optimize.linprog` の `A_ub`, `b_ub` の構築方法の詳細は？
- PDF制約とモーメント制約を1つの `A_ub` にまとめるか？

#### Q3-5: スパース行列の使用条件
設計書では「必要なら `scipy.sparse.csr_matrix` を使用する」とあるが：
- どの程度のサイズ（N, m）からスパース行列を使うべきか？
- **提案**: `N * m > 1e6` の場合にスパース行列を使用

#### Q3-6: `pdf_tolerance` の fallback 処理
設計書では「失敗時に `pdf_tolerance *= 10` を最大数回まで緩める」とあるが：
- 最大回数は？
- **提案**: 最大3回（`pdf_tolerance` を10倍、100倍、1000倍）

## 4. Hybrid法の実装

### ❓ 不明点

#### Q4-1: `init="custom"` の実装詳細
設計書では `fit_gmm1d_to_pdf_weighted_em` の `init` に `"custom"` を追加とあるが：
- `init_params` の引数名は？（例：`init_params: dict` か、個別引数か？）
- **提案**: `init_params: Optional[Dict[str, np.ndarray]] = None` で、`{"pi": ..., "mu": ..., "var": ...}` の形式

#### Q4-2: LPからEMへの初期値の変換
設計書では「LP解の重み `w_all` から上位K成分を選ぶ」とあるが：
- 上位K成分の選択基準は重みのみか？
- 重みが同じ場合の順序は？（例：インデックス順）
- **提案**: `np.argsort(w_all)[::-1][:K]` で重み降順に選択

#### Q4-3: EM初期値の正規化とクリップ
設計書では「`pi_init /= sum(pi_init)` で正規化」「`var_init >= reg_var` でクリップ」とあるが：
- `mu_init` のクリップは不要か？
- `pi_init` がすべて0に近い場合の処理は？

#### Q4-4: QPでの生モーメント一致保証
設計書では「既存QPで生モーメント一致を保証」とあるが：
- 既存のQP（`_project_moments_qp`）は中心モーメント用
- 生モーメント用のQPを新規実装する必要があるか？
- **提案**: 生モーメント→中心モーメントの変換を行い、既存QPを使用

#### Q4-5: `n_init > 1` の場合の `init="custom"` の扱い
設計書では「`trial>0` で摂動を加える」とあるが：
- 摂動の大きさ（1%）は固定か、設定可能か？
- 摂動を加えない選択肢も必要か？

## 5. 評価指標の追加

### ❓ 不明点

#### Q5-1: `compute_errors` 関数の詳細仕様
設計書では返り値が `linf_pdf, linf_cdf, quantile_errors{p: abs(q_true - q_hat)}, q_true, q_hat` とあるが：
- 返り値の型は `dict` か？
- `q_true`, `q_hat` は辞書形式（`{p: value}`）か、配列か？

#### Q5-2: 分位点の計算方法
- `np.interp` を使用するか？
- CDFが単調でない場合の処理は？
- **提案**: `np.interp(p, F_true, z)` で計算し、CDFの単調性は事前に保証

#### Q5-3: 右裾重み付きL1誤差の計算
設計書では「`p0=0.9` の例」とあるが：
- `p0` は設定可能なパラメータか？
- デフォルト値は？
- **提案**: `tail_weight_p0: float = 0.9` をパラメータとして追加

#### Q5-4: 評価指標の出力形式
- `main.py` の標準出力に追加するか？
- JSON/CSV形式での出力も必要か？

## 6. 数値安定化（ρ→±1）

### ✅ 既存実装確認
- `max_pdf_bivariate_normal` は既に存在（`em_method.py` 80行目）
- `sy_given_x = sy * np.sqrt(1.0 - rho * rho)` の計算で、`rho→±1` で0に近づく

### ❓ 不明点

#### Q6-1: `eps_rho` の値
設計書では「`eps_rho=1e-12`」の例があるが：
- これが推奨値か？
- `gmm_utils.py` の `SIGMA_FLOOR`（1e-12）と統一すべきか？

#### Q6-2: `rho` のクリップ位置
設計書では「`rho_clip = min(max(rho, -1+eps_rho), 1-eps_rho)` を内部で使う」とあるが：
- 関数の最初でクリップするか、計算時にクリップするか？
- **提案**: 計算時に `sqrt(max(1-rho^2, eps))` の形式で処理

#### Q6-3: `std_floor` の値
設計書では「`std_floor = 1e-15` 程度」とあるが：
- `gmm_utils.py` の `VAR_FLOOR`（1e-10）と異なるが、統一すべきか？
- **提案**: `cond_std = max(sigma * sqrt(max(1-rho^2, eps)), SIGMA_FLOOR)` を使用

## 7. 後方互換性

### ❓ 不明点

#### Q7-1: 既存configファイルの扱い
設計書では「既存configしかない場合は `J_dict = K`, `L_dict = L`」とあるが：
- `method="lp"` の場合、既存の `K`, `L` をそのまま使用するか？
- `method="hybrid"` の場合、`lp_params.dict_J`, `lp_params.dict_L` が必須か？

#### Q7-2: `objective_mode` の追加
- 既存の `objective_mode="moments"` との関係は？
- `objective_mode="raw_moments"` は `"moments"` の代替か、並行して存在するか？

#### Q7-3: 既存関数の変更
- `fit_gmm_lp_simple` に `objective_mode="raw_moments"` を追加するが、既存の動作は変更しないか？
- **確認**: `objective_mode="pdf"` と `"moments"` の既存動作は維持されるか？

## 8. テスト計画

### ❓ 不明点

#### Q8-1: テストケースの詳細
設計書では「代表パラメータセットを固定して回す」とあるが：
- 具体的なパラメータ値は？
- **提案**: 
  - ρ=0, 0.5, 0.9, 0.99, -0.9, -0.99
  - K=5, 10, 15
  - `tail_alpha=1.0, 2.0, 3.0`

#### Q8-2: 受け入れ条件の詳細
設計書では「`objective_mode="raw_moments"` のLPが反復なしで解ける」とあるが：
- どの程度の精度（`status=0`）を要求するか？
- infeasible の場合の許容範囲は？

## 9. その他

### ❓ 不明点

#### Q9-1: エラーハンドリング
- LPがinfeasibleの場合のエラーメッセージは既存と同様か？
- Hybrid法でLPが失敗した場合の処理は？

#### Q9-2: ログ出力
- 各段階（LP→EM→QP）の実行時間を記録するか？
- 診断情報（`diagnostics`）の出力形式は？

#### Q9-3: ドキュメント更新
- `README.md` の更新範囲は？
- `method_recommendations.md` への追記方法は？

---

## 優先度の高い確認事項

1. **Q3-4**: LP制約行列の構築方法（実装の核心部分）
2. **Q4-4**: QPでの生モーメント一致保証の方法
3. **Q2-1**: 左裾・両裾の実装方法
4. **Q7-2**: `objective_mode` の設計方針

