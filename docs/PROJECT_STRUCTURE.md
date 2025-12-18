# プロジェクト構造

```
ssta/
├── configs/                    # 設定ファイル（JSON）
│   ├── config_example.json     # 基本的な設定例
│   ├── config_lp.json         # LP法の設定例
│   └── config_moments_example.json  # モーメントモードの設定例
│
├── examples/                   # 実行例スクリプト
│   ├── example_pdf_mode.py     # PDF誤差最小化モードの例
│   └── example_moments_mode.py # モーメント誤差最小化モードの例
│
├── benchmarks/                 # ベンチマークスクリプトと結果
│   ├── benchmark.py            # ベンチマークメインスクリプト
│   ├── benchmark_hybrid.py     # Hybrid法ベンチマーク
│   ├── compare_methods.py      # メソッド比較スクリプト
│   ├── analyze_kurtosis_error.py  # 尖度誤差分析スクリプト
│   ├── run_comprehensive_benchmark.sh  # 包括的ベンチマーク実行スクリプト
│   ├── run_quick_benchmark.sh  # クイックベンチマーク実行スクリプト
│   └── results/                # ベンチマーク結果ファイル（JSON、PNG）
│
├── docs/                       # ドキュメント
│   ├── BENCHMARK_GUIDE.md      # ベンチマーク実行ガイド
│   ├── CONFIG_GUIDE.md         # 設定ファイルの詳細ガイド
│   ├── config_examples.md      # 設定例の説明
│   ├── HYBRID_METHOD_GUIDE.md  # Hybrid法実行ガイド
│   ├── LP_RAW_MOMENTS_GUIDE.md # LP法Raw Momentsモードガイド
│   ├── lp_method.md            # LP法の実装仕様書
│   ├── moment_em.md            # モーメントマッチングの説明
│   ├── initial_guess_spec.md   # 初期化方法の仕様
│   ├── TESTING.md              # テストの説明
│   ├── kurtosis_error_analysis_report.md  # 尖度誤差分析レポート
│   └── method_recommendations.md  # 手法推奨事項
│
├── outputs/                    # 生成された出力ファイル（PNG等）
│   └── *.png                   # プロット画像（gitignore対象）
│
├── tests/                      # テストファイル
│   ├── test_config.py
│   ├── test_gmm_fitting.py
│   ├── test_lp_method.py
│   ├── test_moments.py
│   ├── test_output_formatting.py
│   ├── test_pdf_calculation.py
│   └── test_statistics.py
│
├── src/
│   └── gmm_fitting/            # GMMフィッティングパッケージ
│       ├── __init__.py
│       ├── em_method.py        # EM法の実装
│       ├── lp_method.py        # LP法の実装
│       └── gmm_utils.py        # GMMユーティリティ関数
├── main.py                     # メイン実行スクリプト
├── README.md                    # メインドキュメント
├── requirements.txt             # 依存パッケージ
├── Makefile                     # ビルドスクリプト
├── pytest.ini                  # pytest設定
└── .gitignore                   # Git除外設定
```

## ディレクトリの説明

### configs/
設定ファイルの例を格納します。`main.py`の`--config`オプションで指定できます。

### examples/
実行例スクリプトを格納します。プロジェクトルートから実行してください：
```bash
python examples/example_pdf_mode.py
```

### docs/
詳細なドキュメントを格納します。設定方法や実装の詳細については、このディレクトリ内のファイルを参照してください。

### outputs/
実行時に生成されるPNGファイルなどの出力を格納します。`.gitignore`で除外されています。

### benchmarks/
ベンチマークスクリプトと結果を格納します。プロジェクトルートから実行してください：
```bash
python benchmarks/benchmark.py --config configs/config_default.json --output benchmarks/results/benchmark.json
./benchmarks/run_comprehensive_benchmark.sh
```

### tests/
すべてのテストファイルを格納します。`make test`で実行できます。
