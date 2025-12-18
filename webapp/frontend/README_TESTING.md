# Frontend Testing Guide

このプロジェクトでは、JestとReact Testing Libraryを使用してフロントエンドのUIテストを実装しています。

## セットアップ

### 依存パッケージのインストール

```bash
cd webapp/frontend
npm install
```

## テストの実行

### すべてのテストを実行

```bash
npm test
```

### ウォッチモードで実行（開発中に推奨）

```bash
npm run test:watch
```

### カバレッジレポートを生成

```bash
npm run test:coverage
```

カバレッジレポートは `coverage/` ディレクトリに生成されます。

## テストファイルの構造

テストファイルは各コンポーネントと同じディレクトリに `__tests__` フォルダを作成して配置します：

```
src/
  components/
    __tests__/
      ParameterForm.test.jsx
      StatisticsTable.test.jsx
      ResultDisplay.test.jsx
      PlotViewer.test.jsx
    ParameterForm.jsx
    StatisticsTable.jsx
    ResultDisplay.jsx
    PlotViewer.jsx
```

## テストカバレッジ

現在のテストは以下のコンポーネントをカバーしています：

1. **ParameterForm** - パラメータ入力フォーム
   - 入力フィールドの表示
   - 値の更新
   - フォーム送信
   - バリデーション
   - メソッド固有のパラメータ表示

2. **StatisticsTable** - 統計比較テーブル
   - 統計値の表示
   - 相対誤差の計算と表示
   - エラーハンドリング

3. **ResultDisplay** - 結果表示コンポーネント
   - 結果の表示/非表示
   - 各セクションの条件付きレンダリング
   - エラーメトリクス、GMMコンポーネント、実行情報の表示

4. **PlotViewer** - プロット表示コンポーネント
   - プロットの表示
   - スケールモードの切り替え
   - プロット設定の変更
   - エラーハンドリング

## テストの書き方

### 基本的なテスト構造

```javascript
import React from 'react'
import { render, screen, fireEvent } from '@testing-library/react'
import { ThemeProvider, createTheme } from '@mui/material/styles'
import YourComponent from '../YourComponent'

const theme = createTheme()

const renderWithTheme = (component) => {
  return render(
    <ThemeProvider theme={theme}>
      {component}
    </ThemeProvider>
  )
}

describe('YourComponent', () => {
  test('renders correctly', () => {
    renderWithTheme(<YourComponent />)
    expect(screen.getByText(/expected text/i)).toBeInTheDocument()
  })
})
```

### Material-UIコンポーネントのテスト

Material-UIコンポーネントを使用している場合、`ThemeProvider`でラップする必要があります。`renderWithTheme`ヘルパー関数を使用してください。

### ユーザーインタラクションのテスト

```javascript
test('handles user input', async () => {
  renderWithTheme(<YourComponent />)
  
  const input = screen.getByLabelText(/input label/i)
  fireEvent.change(input, { target: { value: 'test value' } })
  
  await waitFor(() => {
    expect(input).toHaveValue('test value')
  })
})
```

### 非同期処理のテスト

```javascript
test('handles async operations', async () => {
  renderWithTheme(<YourComponent />)
  
  const button = screen.getByRole('button', { name: /submit/i })
  fireEvent.click(button)
  
  await waitFor(() => {
    expect(screen.getByText(/success/i)).toBeInTheDocument()
  })
})
```

## モック

### react-plotly.jsのモック

Plotlyコンポーネントはテスト環境で問題を起こす可能性があるため、`PlotViewer.test.jsx`ではモックを使用しています。

## CI/CDでの実行

GitHub ActionsなどのCI/CDパイプラインでテストを実行する場合：

```yaml
- name: Install dependencies
  run: |
    cd webapp/frontend
    npm ci

- name: Run tests
  run: |
    cd webapp/frontend
    npm test -- --coverage --watchAll=false
```

## トラブルシューティング

### テストが失敗する場合

1. 依存パッケージがインストールされているか確認：
   ```bash
   cd webapp/frontend
   npm install
   ```

2. キャッシュをクリア：
   ```bash
   npm test -- --clearCache
   ```

3. テスト環境を確認：
   - Node.jsのバージョンが適切か確認
   - `node_modules`を削除して再インストール

### Material-UIのテーマエラー

Material-UIコンポーネントを使用するテストでは、必ず`ThemeProvider`でラップしてください。

