
# 実装について

---

## 0. まず確認すべき「学習が壊れる典型パターン」

学習が不安定なとき、ほぼ必ずどこかで以下が起きています。

1. **`w` に負値 or NaN/Inf が混入**
2. **`sigma` が極小/極大になって `log`/`exp` が壊れる**（特に `exp(大きい)` → `Inf`）
3. **成分の“入れ替わり（ラベルスイッチ）”が学習中に頻発し、勾配が破綻**
4. **座標系（zの原点）が学習と推論でズレ**ていて、学習した写像が成立しない

あなたのコードは、2) と 3) と 4) を起こしやすい構造になっています。

---

## 1. データ／`w`（確率質量）の扱いが危険（最優先）

### 1.1 `w = f * weights` の正規化が「負の面積」に極端に弱い

`dataset.py` では、`w = f * weights` を作ってから

* `sums = self.w.sum(...)`
* `sums = np.maximum(sums, 1e-12)`
* `self.w /= sums`

という正規化をしています。

ここで **もし `self.w.sum()` が負、あるいは極小正（≪1e-12）** になるサンプルが混ざると、

* `np.maximum` により **分母が 1e-12 に固定**
* 結果 `w` が **巨大スケール**（しかも符号混在）になり得る
* そのまま `InitNet` で **`log(w + eps)` を取って NaN**（負値なら即死）

という「一発で壊れる」経路が存在します。

> **対策（必須）**
> `w` を作る段階で、最低限これを入れてください。
>
> * `f` を **`f = clip(f, 0, +inf)`**（小さな負値を0へ）
> * `w` も **`w = clip(w, 0, +inf)`**
> * `sum_w = w.sum()` が `<= eps` or 非有限なら、そのサンプルは **捨てる** か **一様分布に置換**
> * さらに `w` は `sum=1` を **assert** で保証（学習中に検出してログ）

学習が「安定しない」場合、まずこれで直るケースがかなり多いです。

---

### 1.2 推論側は「平均ゼロへ座標変換」しているのに、学習側はしていない可能性

`infer.py` では

* `w` から一次モーメント（平均） `M1_input = Σ_i w_i z_i`
* `z_relative = z_model - M1_input`（平均が0になるようにシフト）

を**常に実行**しています。

一方、学習側（`train.py`→`model.forward`）は、`dataset` が持つ `z` をそのまま使っています（少なくともここに“同等のシフト”は見当たりません）。

この不一致は **「学習できない」というより「学習が安定しない／一貫した勾配が出ない」**原因になります。
特に RefineBlock は `z` と `mu` の距離で責務を作るので（E-step相当）、座標系のズレは致命的です。

> **対策（どちらかに統一）**
>
> * (推奨) 学習データ作成時／Datasetで **同じ `z_relative` シフト**を入れる
>   つまり各サンプルで `z` を動かす（あるいは `z` は固定で、入力PDFをシフトして“相対座標系”に揃える）
> * あるいは `infer.py` のシフトをやめて、学習と同じ座標系に戻す

---

## 2. `sigma` 更新が `exp` で壊れやすい（NaN/Infの本命）

`RefineBlock.forward` の最後で

* `log_sigma_blend = ... + corr_gamma`
* `sigma_new = exp(log_sigma_blend)`
* `sigma_new = clamp(min=sigma_min)`（**maxは無い**）

となっています。

ここが非常に危険です。学習が進むと `corr_gamma` は無制限に大きくなり得ます（出力層が線形で制限がない）。すると

[
\sigma_{\text{new}} = \exp(\log\sigma_{\text{blend}})
]

で `log_sigma_blend ≳ 90` 程度になると **float32 で即 `Inf`** になり、その後の `log(sigma)` や `(z-mu)/sigma` 等で NaN が出ます。
`min clamp` は `Inf` を救えません。

> **対策（必須）**
>
> 1. `log_sigma_blend` を **上限クリップ**する
>    例：
>    [
>    \log\sigma_{\min}=\log(\sigma_{\min}),\quad
>    \log\sigma_{\max}=\log(\sigma_{\max})
>    ]
>    で
>    [
>    \log\sigma_{\text{blend}} \leftarrow \mathrm{clip}(\log\sigma_{\text{blend}},\log\sigma_{\min},\log\sigma_{\max})
>    ]
>    `sigma_max` は例えば `0.5*(z_max-z_min)` とか、データから観測上限で良いです。
>
> 2. `corr_gamma` を **tanh等で有界化**する
>    例：`corr_gamma = s_sigma * tanh(raw_corr_gamma)`（`s_sigma` は小さめ、例 0.5）
>
> 3. もしくは `sigma_new = softplus(...) + sigma_min` に置換して `exp` を消す（最も堅牢）

この一点だけでも学習が急に安定する可能性が高いです。

---

## 3. `mu` の順序制約が RefineBlock で崩れる（ラベルスイッチ誘発）

InitNet は `ParameterTransform.project` により

[
\mu_1=c,\quad
\mu_k=\mu_{k-1}+\mathrm{softplus}(\beta_{k-1})+\delta_{\min}\quad (k=2,\dots,K)
]

という **単調増加（順序付き）**を作っています。

ところが RefineBlock は `mu_new = (1-\lambda)\mu + \lambda\mu_{EM} + corr_mu` と **`mu` を直接更新**していて、順序制約が維持されません。

これにより、学習中に

* 成分同士が交差（`mu_k > mu_{k+1}`）
* 同一サンプル内でも反復ごとに「成分の意味」が入れ替わる
* InitNet の出力ヘッド（k番目）が何を表すかが揺れ、勾配が不安定

が起きやすいです。

> **対策（強推奨）**
>
> * **RefineBlockも unconstrained パラメータ（alpha,c,beta,gamma）を更新→`project()` で戻す**
>   これが理想です（順序と `sigma_min` が常に保証される）。
>
> 代替案（簡易）：
>
> * 各反復後に `(mu, pi, sigma)` を `mu` でソートして揃える
>   ※ただしソートは勾配が荒れやすいので、安定化には効くが学習は難しくなることがあります。
>
> 代替案（正則化）：
>
> * 順序違反ペナルティ
>   [
>   L_{\text{order}}=\sum_{k=1}^{K-1}\max(0,\mu_k-\mu_{k+1}+\delta)^2
>   ]
>   を追加

---

## 4. RefineBlockの「補正量」が無制限（学習が暴走しやすい）

RefineBlock の補正 `corr_alpha, corr_mu, corr_gamma` は出力層が線形で、**大きさに上限がありません**。
特に `corr_mu` が大きくなると、PDFの主要質量から遠い位置に成分が飛び、CEが急増→勾配爆発、のパターンがあります。

> **対策（実務的に効く）**
>
> * 補正は「残差」でも **有界化**する（tanh/clip）
> * あるいは **学習可能ステップ幅**を導入して小さく始める
>   例：
>   [
>   \mu_{\text{new}}=\mu_{\text{blend}} + s_\mu \tanh(\Delta_\mu),\quad
>   s_\mu=\mathrm{softplus}(s_{\mu,0})
>   ]
> * `mu` を `global_mean ± c*global_std` の範囲にクリップする（ハードでも効果は大きい）

---

## 5. 損失設計（Deep supervision）が初期に強すぎる可能性

`compute_deep_supervision_loss` は各反復（init含む）に重み `η_t` を付けてCE等を足します。デフォルトが `"linear"` で後半重視ですが、初期の粗い出力にも損失が乗ります。

学習初期は InitNet が未学習で、Refineも未安定なので、**序盤の損失が大きすぎて補正が暴走**しやすいです。

> **対策（おすすめのスケジュール）**
>
> 1. 最初の数epochは `eta_schedule="final_only"`（最後だけ監督）
> 2. ある程度安定してから `"linear"` へ
> 3. さらに必要なら `"uniform"` へ

---

## 6. トレーニングコード側の「検出・遮断」が不足（壊れたバッチを止めない）

現状 `train.py` は勾配クリップはありますが、**NaN/Inf検出→そのバッチをスキップ**等が無いので、一度壊れると全体が終わります。

> **対策（デバッグの最短ルート）**
>
> * まず以下を入れて、どこで壊れているか特定してください
>
>   * `assert torch.isfinite(w).all()`（入力）
>   * 各反復で `pi, mu, sigma` の `isfinite` / min/max ログ
>   * `torch.autograd.set_detect_anomaly(True)`（重いが最初だけ有効）
> * `loss` が NaN/Inf なら `optimizer.step()` をしないでスキップ＋ログ保存

---

# 優先度付き“修正ロードマップ”（最短で安定させる順）

1. **Dataset / infer の `w` 健全化**（`clip` + 正規化失敗の扱い + `assert`）
2. **`sigma` の `exp` オーバーフロー対策**（`log_sigma` 上限clip、または `softplus` 化）
3. **Refine反復でも `mu` の順序制約を維持**（unconstrained更新→project、またはorder正則化）
4. **補正量を有界化**（`tanh` + ステップ幅）
5. **学習初期は final_only**（deep supervisionを弱める）
6. **NaN/Infを検出してバッチ遮断**（壊れた勾配更新を止める）

---


# ログについて
ログだけでも、かなり「原因の当たり」が付けられます。ポイントは **(1) 指標が2つの状態を行き来している**ことと、**(2) CDF L∞ が 1 を大きく超えている**ことです。

---

## 1) ログの挙動は「2つのモード」を行き来しています

### モードA（良い）

* 例: Epoch 4〜8, 13〜19
* **Val PDF L∞ ≈ 0.01〜0.03**, **Val CDF L∞ ≈ 0.005〜0.02**
* かなり良い近似ができている状態

### モードB（崩壊/スパイク）

* 例: Epoch 2〜3, 9〜12, 20〜21
* **Val PDF L∞ が 0.7〜2.3**, **Val CDF L∞ が 0.3〜2.2**
* 近似が壊れている状態

特に Epoch 9 は

* **Val CE: 0.5249（良く見える）**
* **Val PDF L∞: 2.3302, Val CDF L∞: 2.2015（最悪）**
  で、**「CEは良いのに形は最悪」**という典型的な崩壊モードです。

---

## 2) CDF L∞ が 1 を超えるのは “強い異常サイン”

CDF L∞ は「CDFの最大絶対誤差」です。
通常、真のCDF (F_{\text{true}}) と推定CDF (F_{\text{hat}}) がどちらも **0〜1の範囲**にあるなら

[
\max_i |F_{\text{true}}(z_i)-F_{\text{hat}}(z_i)| \le 1
]

のはずです。

それが **2.2015** まで出ているので、ほぼ確実に次のどちらかが起きています。

### (A) 予測PDFが「グリッド解像度より細いスパイク」になっている

validationでCDFを作るとき、あなたの実装は **台形則**

[
F(z_j)\approx \sum_{i\le j} f(z_i),\Delta z_i
]

で積分しています。

ここで、もし推定GMMが **極小 (\sigma)** を出して「ピークが非常に高い」状態になると、
グリッド上の一点の値 (f(z_i)) が大きくなりすぎて、**台形則の離散積分が1を大きく超えてしまう**ことがあります（本当の連続積分は1でも、数値積分が壊れる）。

→ Epoch 9 の **CDF L∞ > 2** はこのパターンが最有力です。

### (B) 入力側の (f_{\text{true}}) が正規化されていない/負値が混ざる

こちらも可能性はありますが、Epoch 4〜8 で CDF L∞ が 0.01 台まで下がっているので、**恒常的に真のPDFがおかしい**というよりは、**モデル側がある時だけ極端な形になる**方が整合的です。

---

## 3) 何が起きているかを一発で確定するログ（超重要）

次の3つを **validate中に必ず出してください**（平均でなくmin/maxが効きます）。

1. (\sigma) の統計

* `sigma_min_batch = sigma.min().item()`
* `sigma_median_batch = sigma.median().item()`
* `sigma_max_batch = sigma.max().item()`
* `ratio_floor = (sigma <= sigma_min*1.001).float().mean().item()`  （下限張り付き率）

2. 予測PDFのピーク

* `f_hat.max()` の min/avg/max（バッチ内）

3. 入力 w の健全性

* `w.min()`, `w.sum(dim=-1)` の min/max

  * **wが負**、または **sumが1からズレる**なら学習が壊れます

これで「Epoch 9 のとき sigma が下限に張り付いてスパイク化してる」がまず確認できるはずです。

---

## 4) 対策の優先順位（効く順）

### 対策1（最優先）: (\sigma_{\min}) を “グリッド幅 (\Delta z)” に合わせて大きくする

グリッド幅を

[
\Delta z = \frac{z_{\max}-z_{\min}}{N-1}
]

とすると、経験的に

* (\sigma_{\min} \approx 0.3\Delta z) 〜 (1.0\Delta z)

くらいにしないと、**スパイクの数値積分が壊れて**学習が不安定になります。
（今の `sigma_min=1e-3` は、N=96なら普通は (\Delta z) より何桁も小さいはずで、危険です）

### 対策2: Refineでの (\sigma) 更新を `exp` から安全形へ

現状は

[
\sigma_{\text{new}}=\exp(\log\sigma_{\text{blend}})
]

なので、学習中に簡単に極端値に飛びます。
少なくとも

* (\log\sigma_{\text{blend}}) を ([\log\sigma_{\min},\log\sigma_{\max}]) にクリップ
* あるいは `sigma = softplus(raw) + sigma_min`

にしてください。

### 対策3: CDF評価は “解析的CDF” で出す（台形則をやめる）

GMMなら CDF は

[
F_{\text{hat}}(z)=\sum_{k=1}^{K}\pi_k ,\Phi!\left(\frac{z-\mu_k}{\sigma_k}\right)
]

で計算できます（(\Phi) は標準正規CDF）。
これにすれば **CDF L∞ が 1 を超える**みたいな“数値積分崩壊”は消えますし、評価が安定します。

### 対策4: Refineで (\mu) の順序制約が壊れる問題も潰す

これはログから直接は断定しませんが、今の実装は Refineで (\mu) を直接動かすので成分が交差しやすく、学習が揺れやすいです。
（`project()` を毎反復で通す、または順序ペナルティを入れる）

---

## 5) まとめ（このログから言える結論）

* モデルは **一度は非常に良い精度（Val PDF L∞ ≈ 0.01）**に到達できています
* しかし時々、**極小σによるスパイク解**に落ちて評価が崩壊している可能性が高い
* その兆候が **CDF L∞ が 1 を大幅に超えている**点に現れています
* 最短の修正は **sigma_min を Δz オーダに上げる**＋**sigma更新の安全化**＋**CDF評価を解析式に変える**です

---

#σの解像度について

---

## 1) グリッド間隔 (\Delta z) の計算

あなたの設定は

* (z_{\min}=-15)
* (z_{\max}=15)
* (N=96)

なので、等間隔グリッドなら

[
\Delta z=\frac{z_{\max}-z_{\min}}{N-1}
=\frac{15-(-15)}{96-1}
=\frac{30}{95}
=0.3157894736\ldots
]

つまり **(\Delta z \approx 0.316)** です。

---

## 2) 今の挙動（CDF L∞ が 2 まで跳ねる）と (\sigma_{\min}=10^{-3}) は整合しています

正規分布（ガウス）の1成分のピーク値は

[
\max_z \mathcal{N}(z;\mu,\sigma^2)=\frac{1}{\sigma\sqrt{2\pi}}
]

です。

今の実装は CDF を **台形則**（離散積分）で

[
F(z_j)\approx \sum_{i\le j} f(z_i),w_i,\quad
w_0=w_{N-1}=\frac{\Delta z}{2},;w_i=\Delta z
]

としているので、もし (\sigma) が極小で (f(z)) が「グリッド点で非常に大きい」スパイクになると、
(\sum f(z_i)w_i) が 1 を大きく超えて **CDF が 1 を超える**（→差も 1 を超える）現象が起きます。

あなたのログで **Val CDF L∞ が 2.2** まで出ているのは、ほぼ確実にこのタイプです。

---

## 3) 具体的に推奨する (\sigma_{\min})（結論）

### 推奨デフォルト

**[
\sigma_{\min}=0.2
]**

根拠は「グリッド解像度 (\Delta z \approx 0.316) と同じオーダに制限する」ことです。
経験則として

[
\sigma_{\min}=c,\Delta z,\quad c\in[0.5,,0.8]
]

が安定しやすいです。

今回だと

* (0.5\Delta z \approx 0.158)
* (0.6\Delta z \approx 0.189)
* (0.7\Delta z \approx 0.221)

なので、実装しやすく丸めて **0.2** を推します。

> もし「分布がかなり鋭い（本当に (\sigma\ll 0.1) が必要）」という事情があっても、
> **この“グリッド学習”では (\sigma\ll\Delta z) は数値的に別問題**になるので、まず 0.2 で安定化→必要なら評価方法を「ビン確率（CDF差分）」に変える、が安全です。

---

## 4) 変更すべき場所（最小修正で効く順）

### (A) `sigma_min` を 0.2 に上げる（最優先）

* `train.py` の引数 `--sigma_min` を **0.2** に変更して再学習
* `model.py` 側のデフォルトも **1e-3 → 0.2** に揃える（事故防止）

### (B) RefineBlock の (\sigma) 更新に上限（(\sigma_{\max})）も付ける

現状は

[
\sigma_{\text{new}}=\exp(\log\sigma_{\text{blend}})
]
[
\sigma_{\text{new}}\leftarrow \max(\sigma_{\text{new}},\sigma_{\min})
]

で **上限が無い**ので、学習が揺れます。
最低限、

[
\log\sigma_{\text{blend}} \leftarrow \mathrm{clip}\left(\log\sigma_{\text{blend}},\log\sigma_{\min},\log\sigma_{\max}\right)
]

を入れてください。

おすすめは
[
\sigma_{\max}=5
]
（範囲 30 の中で十分大きく、極端な広がりを抑えられる）

---

## 5) この修正でログがどう変わるべきか

* **Val CDF L∞ が 1 を超える跳ね方が消える**
* Epoch時間が突然 10〜20倍になる現象（あなたのログの Epoch 8〜12 みたいなやつ）が減る可能性が高い
  （(\sigma) が小さすぎると `exp` 計算が“遅い状態”に入りやすいことがあります）

---

## 6) 追加で「一発で原因確定」できるログ（おすすめ）

次回の学習で、各 epoch の val で一度だけで良いので出してください：

* (\sigma) の最小値：`sigma.min()`
* (\sigma) の中央値：`sigma.median()`
* (\sigma) の最大値：`sigma.max()`

もし “跳ねたepoch” で `sigma.min()` が 0.05 とか 0.01 とかに落ちていれば、今回の診断はほぼ確定です。

---

それでも学習が揺れるなら、次は「CDF誤差の計算を解析的CDFに切り替える」か「損失を“ビン確率”ベースに変える」の順で手当てするのが最短です。
