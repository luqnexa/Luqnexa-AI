# Rust製の軽量ニューラルモデル

ユーザーの新方針に従い、旧Python学習系列の重みを引き継がず、`crates/luqnexa-ml` で学習・推論をRustに実装しました。旧モデル・教材・辞書は削除しません。新系列の記録は `.cache/rust-neural-reset` で分離しています。旧Python系列の定期学習はRust系列へ変更しました。稼働IMEのインストールやモデル切り替えは行いません。

独自モデルは、読み・直前の文脈・候補表記のUnicode文字1～3-gramを8,192バケット、16次元の学習可能な埋め込みへ写します。各入力の平均を連結した48次元から、32ユニットのReLU層を経て候補スコアを出します。ペアワイズのロジスティック損失をRustの逆伝播で最小化し、モデル行ごとのint8量子化を行います。既存LLMや圧縮エンジンの流用はありません。候補の順位を学習するニューラルモデルで、汎用文章生成や大規模な意味理解を完成させたものではありません。

モデルファイルは165,692 bytes、常駐データは165,688 bytesです。読み64文字、最近の文脈48文字、候補96文字を上限に処理します。モデルを一度ロードした後、入力の準備と推論はヒープ確保を行わず、候補間で読み・文脈の埋め込みを共有できます。外部通信、入力履歴の保存、辞書アクセス、自動確定はありません。`nextime-ai` の `mobile-neural` featureには既存候補だけを順位付けする明示的なアダプターを用意しました。標準のIMEには自動で組み込みません。

```powershell
cargo test -p luqnexa-ml --features trainer
cargo run -p luqnexa-ml --features trainer --release --bin luqnexa-train -- --data DATA.jsonl --output NEW-RUN --epochs 200 --seed 1337
cargo bench -p luqnexa-ml --bench inference
cargo check -p luqnexa-ml --lib --no-default-features --target aarch64-linux-android
cargo check -p luqnexa-ml --lib --no-default-features --target aarch64-apple-ios
```

出力先は新しいディレクトリのみ受け付け、既存ファイルを上書きしません。JSONLには `input`, `context`, `good`, `bad`, `split` を指定します。train・validation・testを分離し、入力・正答表記の分割間重複は拒否します。モデルは毎回明示的なseedから新規初期化します。モデル選択は量子化後のvalidation正解数と損失のみで行い、選択後にtestを評価します。次の学習では点検した累積教材から新しいrunを作り、固定validation/testとハッシュを保持してください。単語置換の大量反復を独立した文章として数えてはいけません。

最初のリセット実験は、点検済みの訓練24文と固定validation12文・test24文から開始しました。旧1,000例の単語置換教材は新系列に混ぜていません。seed 1337・200epochで2回実行し、同一モデルSHA-256を検証しました。候補ペア評価はtrain24/24・validation9/12・test17/24でした。合成教材の自己点検による評価で、独立した人間の検証やIME全体の変換精度ではありません。

`nextime-core` の `mobile-neural` featureで `Engine::convert_with_neural` を使えます。辞書から最大32候補を生成してから新Rustモデルで順位を付け、希望件数へ絞ります。旧LLMを呼ばず、入力された直前の文脈だけを使います。学習済みモデルを指定する隔離デモは次のとおりです。設定や稼働IMEは変更しません。

```powershell
cargo run --release -p nextime-core --features mobile-neural --example mobile_neural -- .cache/rust-neural-reset/run-a/model.lqml hashi 夕食
$env:LUQNEXA_BENCH_MODEL = (Resolve-Path .cache/rust-neural-reset/run-a/model.lqml).Path
cargo bench -p luqnexa-ml --bench inference
Remove-Item Env:LUQNEXA_BENCH_MODEL
```

学習済みモデルでの同条件ホスト再測定はp50 12.0µs・p95 15.6µsでした。ベンチマークのモデル指定は絶対パスを使ってください。新経路を含む `nextime-core` 全体のAndroid・iOS向けコンパイル確認も通過しています。`crates/luqnexa-ml/tests/allocations.rs` はロード後の繰り返しUnicode推論でヒープ確保がないことを測ります。候補プールや辞書・アプリ自体は別途メモリを使います。

## スマホから呼ぶC API

`nextime-ffi --features mobile-neural` にモデルの作成・スコア計算・解放のC ABIを追加しました。宣言は `crates/nextime-ffi/include/luqnexa_model.h`。モデルは明示的に渡したバイト列から読み込み、パス・ネットワーク・入力履歴へアクセスしません。同時ロードは最大4個です。候補の順位はホストが決め、文章の挿入・確定はユーザー操作で行います。長さ制限、UTF-8、破損モデル、ハンドルの解放、Rust APIと同じスコアを返すことをテストしています。

WindowsのCプログラムから学習済みモデルのロード・日本語候補のスコア計算・解放を実行しました。`platforms/common/neural-demo.c` と `scripts/test-neural-ffi.ps1 -ModelPath .cache/rust-neural-reset/run-a/model.lqml` で再現できます。Android・iOS向けFFIの `cargo check` も通過しています。

スマホ用ライブラリのリンクにはAndroid NDK、iOSではXcode SDKが必要です。本環境でのcross checkはSDKを使った最終リンクや実機実行を証明しません。AndroidのInputMethodServiceとiOSのCustom KeyboardへのUI接続、正規のホスト連携、実機のp95・メモリ・電力は未完了です。モデルだけの常駐サイズをキーボード拡張全体のメモリ量として扱わないでください。

## Ubuntuと実機測定

Macはなく、iPhoneとUbuntuが利用可能という現在の環境では、学習とLinuxネイティブの推論検証をUbuntuで進めます。iPhoneネイティブアプリの最終ビルド・署名・配布には、別途macOS/Xcodeのビルド環境と適切な署名設定が必要です。CIのmacOSコンパイル確認をiPhoneでの動作確認とは呼びません。Appleの実機手順: https://developer.apple.com/documentation/xcode/running-your-app-on-simulated-or-physical-devices

推論専用の `mobile-probe` は依存ライブラリなしで動き、学習済みモデルをロードして1・8・32候補のp50・p95・p99をJSONへ記録します。通常文字長とモデル上限のUnicode文字長を分けます。候補は固定の繰り返しで、意味理解精度・辞書検索・UI・電力の測定ではありません。OSとCPUアーキテクチャも記録します。ロード後モデルの常駐データ量はプロセス全体のRSSではありません。

```sh
# Ubuntu上で実行する。出力先は新しい測定名を使う。
cargo test -p luqnexa-ml
cargo run --release -p luqnexa-ml --bin mobile-probe -- .cache/rust-neural-reset/run-a/model.lqml 2000 > probe-ubuntu.json
```

Windowsでの最初のprobe結果は `.cache/rust-neural-reset/portable-probe-windows.json`。通常32候補はp95 114.3µs、最大文字長32候補はp95 267.9µsでした。これはWindows x86_64ホストの結果であり、Ubuntu・A13・iPhoneの実測ではありません。

Windowsのホスト測定では、入力準備と4候補のスコア計算がp95 15.4µsでした。初期化モデル、2,000反復、ウォームアップ100回の測定です。辞書検索、UI、電力、スマホ実機は含みません。Android/iOSのAArch64向け `cargo check` は通過しましたが、アプリへのリンク・キーボード拡張への組み込み・A13実機の速度、発熱、電力は未検証です。次段階は教材の多様性を保った拡充、実辞書候補への適用評価、スマホ実機のメモリ・p95測定です。
