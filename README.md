# Luqnexa-AI

Luqnexaの端末内AIと候補ランキングのソースです。Rustの `nextime-ai`・`nextime-ranker`、ローカルQwen接続、公開文章による文字統計モデルの学習スクリプトを収録します。

実装済み: 直前の確定文からの候補順位づけ、大文字小文字の選択、端末内の選択・確定文からの軽量学習、取消と手動選択の尊重。文章の自由生成・返信の自動送信・ニューラルモデルの追加学習・全OS接続は未実装です。

```sh
cargo test --workspace --locked
python scripts/test-train-context.py
python scripts/test-train-literature.py
python -m unittest discover -s tests -p test_local_ai_server.py
```

Windowsの任意ローカルモデル: `python scripts/setup-local-ai.py`、`./scripts/start-local-ai.ps1`。取得するQwenモデルは約2.5GBで、実行には追加RAMが必要です。モデルと実行ファイルはこのリポジトリに同梱せず、公式配布元から固定版・SHA-256検証付きで取得します。実モデルテストは明示的な起動時だけ実行します。

入力は固定ループバックで処理し、外部送信しません。履歴・鍵・トークン・個人学習データ・取得キャッシュを同梱していません。組み込みの統計モデルは公開コーパスから作成したもので、出典・ライセンス・著作者表示を `models/` に保持しています。

ソースはAGPL-3.0-only。第三者モデル・コーパス・ランタイムには元のライセンスが適用されます。モデル統合は [Luqnexa](https://github.com/Luqnexa/Luqnexa)、公開文章収集は [Luqnexa-Server](https://github.com/Luqnexa/Luqnexa-Server) を参照してください。


## 2026-10-13 更新

端末内の独自候補ランキングと教材点検・学習・評価スクリプト、学習状況画面を更新しました。[担当アシスタント学習](docs/assistant-learning.md)と[教材貼り付け](docs/paste-training.md)を参照してください。欠席下書きは定型文とランキングです。未採用の実験重みは含めません。Qwenの常駐学習は現在停止しており、任意起動スクリプトを収録していても自動起動しません。
