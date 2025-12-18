# GitHubリポジトリのセットアップ手順

## 1. GitHubでリポジトリを作成

1. GitHubにログインして、https://github.com/new にアクセス
2. リポジトリ名を `gmm-pdf-fitting` に設定
3. 説明を追加（例: "Gaussian Mixture Model PDF Fitting Library"）
4. Public または Private を選択
5. **「Initialize this repository with a README」はチェックしない**（既にローカルにリポジトリがあるため）
6. 「Create repository」をクリック

## 2. リモートリポジトリを追加

GitHubでリポジトリを作成した後、以下のコマンドを実行してください：

```bash
# リモートリポジトリを追加（YOUR_USERNAMEを実際のGitHubユーザー名に置き換えてください）
git remote add origin https://github.com/YOUR_USERNAME/gmm-pdf-fitting.git

# またはSSHを使用する場合
git remote add origin git@github.com:YOUR_USERNAME/gmm-pdf-fitting.git
```

## 3. コードをプッシュ

```bash
# メインブランチをプッシュ
git push -u origin main
```

## 4. リポジトリの設定（オプション）

GitHubリポジトリの設定ページで以下を設定することを推奨します：

- **Description**: "Gaussian Mixture Model PDF Fitting Library - Approximate PDF of max(X,Y) using GMM with EM, LP, and Hybrid methods"
- **Topics**: `gmm`, `gaussian-mixture-model`, `pdf-approximation`, `statistics`, `python`, `machine-learning`
- **Website**: （Webアプリがある場合はURLを追加）

## 5. リリースの作成（オプション）

最初のリリースを作成する場合：

```bash
# タグを作成
git tag -a v0.1.0 -m "Initial release"

# タグをプッシュ
git push origin v0.1.0
```

その後、GitHubのリリースページでリリースノートを追加できます。

