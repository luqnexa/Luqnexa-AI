# 学習順序の偏りを分離する実験

2026-10-13。現在の教材では同じ方向の例や逆向きの対比が隣接します。順序の影響を分離するため、Rust学習CLIに`--schedule shuffled`を追加しました。既定は従来の`cyclic`であり、過去の再現性を保持します。推論やモデル形式、学習率は変更しません。

Fisher–Yatesで訓練インデックスだけをepochごとに並べ替えます。初期化と独立した乱数状態を同じseedから作り、全訓練例を毎epoch一回ずつ使います。特定の例を反復して件数を増やさず、validation/testは順序作成や勾配計算に使いません。単体テストで全例の一回ずつの使用と同条件での順序再現性を確認しています。

```powershell
cargo run --release -p luqnexa-ml --features trainer --bin luqnexa-train -- --data DATA.jsonl --output NEW-RUN --epochs 2000 --seed 1337 --architecture interaction-v2 --schedule shuffled
```

第20教材群を含む72訓練例で独立二度実行しました。モデルSHA-256は`a81de2772f732bf9a2a1a39cb27a4230e3bbb5f2bec810e1950c246d2195e8e6`で一致しました。validationのみで選択した4epoch候補は訓練41/72、validation8/12、test12/24です。元モデルのvalidation9/12・test17/24より低く、不採用にしました。固定評価の内容は変更せず、過去成果物と稼働IMEは保持します。

順序を混ぜただけでは汎化の課題を解決しないという診断です。同じハッシュや再学習を精度向上と呼びません。モデルを選ぶ標本が12例と小さく、わずかな差や初期段階の順位変動が選択を左右する点も課題です。固定評価を狙って教材や重みを調整する方法は使いません。

通常変換90%以上・変換p95が5ms以内・誤字補正80%以上は未達です。今回の数値は正解候補を与えた合成ペア比較であり、IME全体やiPhoneの実測ではありません。予定タスク・Qwen・外部APIは起動しません。
