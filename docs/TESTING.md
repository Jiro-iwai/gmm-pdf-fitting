# テストガイド

このプロジェクトにはpytestを使用した単体テストが含まれています。

## テストの実行

### すべてのテストを実行

```bash
# 仮想環境を有効化
source .venv/bin/activate

# すべてのテストを実行
pytest

# 詳細な出力で実行
pytest -v

# カバレッジレポート付きで実行
pytest --cov=em_method --cov-report=term-missing
```

### 特定のテストファイルを実行

```bash
# PDF計算のテストのみ実行
pytest tests/test_pdf_calculation.py

# GMMフィッティングのテストのみ実行
pytest tests/test_gmm_fitting.py

# 特定のテストクラスを実行
pytest tests/test_pdf_calculation.py::TestNormalPDF
```

### 特定のテスト関数を実行

```bash
# 特定のテスト関数を実行
pytest tests/test_pdf_calculation.py::TestNormalPDF::test_normal_pdf_basic
```

## テスト構成

テストは以下のファイルに分かれています：

- **`tests/test_pdf_calculation.py`**: PDF計算関数のテスト
  - `normal_pdf`: 正規分布のPDF計算
  - `max_pdf_bivariate_normal`: 2変量正規分布の最大値PDF
  - `max_pdf_bivariate_normal_decomposed`: 分解されたPDF
  - `normalize_pdf_on_grid`: PDFの正規化

- **`tests/test_gmm_fitting.py`**: GMMフィッティングのテスト
  - `GMM1DParams`: GMMパラメータのデータクラス
  - `gmm1d_pdf`: GMM PDFの評価
  - `fit_gmm1d_to_pdf_weighted_em`: 重み付きEMアルゴリズム

- **`tests/test_moments.py`**: モーメント計算のテスト
  - `_compute_central_moments`: 中心モーメントの計算
  - `_central_to_raw_moments`: 中心モーメントからrawモーメントへの変換
  - `_compute_component_raw_moments`: 成分のrawモーメント計算
  - `_project_moments_qp`: QP投影によるモーメント一致

- **`tests/test_statistics.py`**: 統計量計算のテスト
  - `compute_pdf_statistics`: PDFから統計量（平均、標準偏差、歪度、尖度）を計算

- **`tests/test_config.py`**: 設定読み込みのテスト
  - `load_config`: JSON設定ファイルの読み込み
  - `prepare_init_params`: 初期化パラメータの準備

- **`tests/test_output_formatting.py`**: 出力フォーマットのテスト
  - 各種出力関数のテスト

## テストカバレッジ

現在のテストカバレッジは約62%です。主要な機能はカバーされていますが、以下の領域は追加のテストが必要です：

- エッジケースの処理
- エラーハンドリング
- プロット関数（視覚的な出力のためテストが困難）

## テストの追加

新しい機能を追加する際は、対応するテストも追加してください：

1. 適切なテストファイルにテスト関数を追加
2. テスト関数名は`test_`で始める
3. テストクラスは`Test`で始める
4. アサーションを使用して期待される動作を検証

例：

```python
def test_new_functionality():
    """Test description."""
    # Arrange
    input_data = ...
    
    # Act
    result = function_to_test(input_data)
    
    # Assert
    assert result is not None
    assert np.allclose(result, expected_value)
```

