# GMM Fitting Web Application

FastAPI + React によるWebアプリケーション実装です。

## 主な機能

- **3つのフィッティング方式**: EM、LP、Hybrid方式を選択可能
- **インタラクティブなプロット**: Plotly.jsによる動的なプロット表示
  - 線形/対数スケールの切り替え
  - プロット範囲のカスタマイズ（X軸、Y軸）
  - 色、線のスタイル、線の太さの調整
  - グリッドポイントの表示/非表示
- **統計比較テーブル**: 相対誤差に応じた色分け表示（誤差が小さい=青、普通=黄、大きい=赤）
- **ダークモード**: ライト/ダークモードの切り替え（設定はlocalStorageに自動保存）
- **設定のエクスポート/インポート**: 現在のパラメータをJSONファイルとして保存・読み込み可能
- **リアルタイム計算**: パラメータ変更後すぐに結果を確認

## ディレクトリ構造

```
webapp/
├── __init__.py
├── models.py          # Pydanticモデル（APIリクエスト/レスポンス）
├── api.py             # FastAPIアプリケーション
├── requirements.txt   # Python依存パッケージ
├── README.md         # このファイル
└── frontend/         # Reactフロントエンド
    ├── package.json
    ├── vite.config.js
    ├── index.html
    └── src/
        ├── main.jsx
        ├── App.jsx
        ├── App.css
        ├── index.css
        └── components/
            ├── ParameterForm.jsx
            ├── ResultDisplay.jsx
            ├── PlotViewer.jsx
            └── StatisticsTable.jsx
```

## セットアップ

### 1. バックエンド（FastAPI）のセットアップ

```bash
# 仮想環境を有効化（既存のプロジェクトの仮想環境を使用）
source .venv/bin/activate

# Webアプリ用の依存パッケージをインストール
pip install -r webapp/requirements.txt
```

### 2. フロントエンド（React）のセットアップ

```bash
# frontendディレクトリに移動
cd webapp/frontend

# 依存パッケージをインストール
npm install
```

## 実行方法

### 方法1: Makefileを使用（推奨）

```bash
# 1. 依存パッケージのインストール（初回のみ）
make webapp-install

# 2. Webアプリの起動（バックエンド + フロントエンド）
make webapp-start

# 3. Webアプリの停止
make webapp-stop

# その他のコマンド
make webapp-status    # ステータス確認
make webapp-logs      # ログ表示
make webapp-backend   # バックエンドのみ起動
make webapp-frontend  # フロントエンドのみ起動
make webapp-clean     # ログ・PIDファイルのクリーンアップ
```

### 方法2: 手動で起動

#### 1. バックエンドサーバーの起動

```bash
# プロジェクトルートから実行
cd /path/to/ssta
source .venv/bin/activate

# FastAPIサーバーを起動（ポート8000）
python -m webapp.api
# または
uvicorn webapp.api:app --reload --host 0.0.0.0 --port 8000
```

APIサーバーは `http://localhost:8000` で起動します。
APIドキュメントは `http://localhost:8000/docs` で確認できます。

#### 2. フロントエンド開発サーバーの起動

別のターミナルで：

```bash
cd webapp/frontend
npm run dev
```

Reactアプリは `http://localhost:3000` で起動します。

## APIエンドポイント

### `POST /api/compute`

GMMフィッティングを実行します。

**リクエスト例:**

```json
{
  "bivariate_params": {
    "mu_x": 0.0,
    "sigma_x": 0.8,
    "mu_y": 0.0,
    "sigma_y": 1.6,
    "rho": 0.9
  },
  "grid_params": {
    "z_range": [-6.0, 8.0],
    "z_npoints": 2500
  },
  "K": 3,
  "method": "em",
  "em_params": {
    "max_iter": 400,
    "tol": 1e-10,
    "n_init": 8,
    "init": "quantile",
    "use_moment_matching": false
  }
}
```

**レスポンス:**

```json
{
  "success": true,
  "method": "em",
  "z": [...],
  "f_true": [...],
  "f_hat": [...],
  "gmm_components": [...],
  "statistics_true": {...},
  "statistics_hat": {...},
  "error_metrics": {...},
  "execution_time": {...}
}
```

**注意**: プロットはフロントエンドでPlotly.jsを使用して動的に生成されます。バックエンドからはプロット画像は返されません。

### `POST /api/load-config`

設定ファイル（JSON）を読み込み、パラメータを返します。

**リクエスト:**
- `multipart/form-data`形式でJSONファイルをアップロード

**レスポンス:**
- `ComputeRequest`形式のパラメータオブジェクト

### `GET /api/health`

ヘルスチェックエンドポイント。

## 開発

### バックエンドの開発

- `webapp/models.py`: Pydanticモデルの定義
- `webapp/api.py`: FastAPIエンドポイントの実装

### フロントエンドの開発

- `webapp/frontend/src/App.jsx`: メインアプリケーションコンポーネント（テーマ管理、状態管理）
- `webapp/frontend/src/components/ParameterForm.jsx`: パラメータ入力フォーム（設定のエクスポート/インポート機能含む）
- `webapp/frontend/src/components/ResultDisplay.jsx`: 結果表示コンポーネント
- `webapp/frontend/src/components/PlotViewer.jsx`: Plotly.jsによるインタラクティブプロット表示
- `webapp/frontend/src/components/StatisticsTable.jsx`: 統計比較テーブル（相対誤差による色分け）

## トラブルシューティング

### CORSエラーが発生する場合

`webapp/api.py` の `CORSMiddleware` 設定で、フロントエンドのURLを許可リストに追加してください。

### プロットが表示されない場合

- ブラウザのコンソールでエラーを確認してください
- Plotly.jsが正しく読み込まれているか確認してください（`npm install`を実行）
- プロットデータ（`z`, `f_true`, `f_hat`）が正しく返されているか確認してください

### ポートが既に使用されている場合

- バックエンド: `uvicorn` コマンドの `--port` オプションで変更
- フロントエンド: `vite.config.js` の `server.port` で変更

## 本番環境へのデプロイ

### バックエンド

- Gunicorn + Uvicorn workers を使用
- 環境変数で設定を管理

### フロントエンド

```bash
cd webapp/frontend
npm run build
```

ビルドされたファイルは `webapp/frontend/dist` に生成されます。

