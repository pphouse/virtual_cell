# Virtual Cell Challenge 2026 — CCDT

Arc Institute の [Virtual Cell Challenge 2026](https://arcinstitute.org/news/virtual-cell-challenge-2026)
への提出用パイプライン。**未知の細胞株**における CRISPRi ノックダウン応答を、
その細胞株の未摂動細胞と標的遺伝子リストだけから予測する（ゼロショット）。

締切: **2026-11-05 23:59 UTC**（最終テストセットは 10-22 公開）

## 何が入っているか

| ドキュメント | 内容 |
|---|---|
| [`docs/01-課題整理.md`](docs/01-課題整理.md) | 課題・データ・日程、2025 からの変更点と設計への含意 |
| [`docs/02-評価指標の解剖.md`](docs/02-評価指標の解剖.md) | cell-eval の実装を読んで分かったこと、実測で見つけた 2 つの落とし穴 |
| [`docs/03-モデル設計.md`](docs/03-モデル設計.md) | CCDT の設計と、各項をそう置いた理由 |
| [`docs/04-実行手順とロードマップ.md`](docs/04-実行手順とロードマップ.md) | 実行コマンドと締切までの優先順位付き計画 |

コードは `src/vcc/`、実行スクリプトは `scripts/`、
チャレンジデータ不要で全体を回すシミュレータとテストが `tests/`。

## 手法の要点

```
delta(g, c*) = alpha * M(g, c*) * [ b * D(g, c*) + (1 - b) * K(g, c*) ]
               標的遺伝子成分のみオンターゲットモデルで上書き
```

- **D** 実測シグネチャを *fold change として* 未知細胞株にリベース。
  加算的な delta は文脈間で保存されないが、乗算的な効果は保存される。
- **K** どのソースでも摂動されていない標的（＝大多数）向けに、
  低ランク応答プログラムの負荷量を遺伝子特徴から予測して展開。
- **M** 摂動ごとの効果量。PDS が最も敏感な項。
- **alpha** MAE と PDS のトレードオフを握る唯一のノブ。細胞株ホールドアウトで較正。

end-to-end の深層モデルにしていないのは、2025 の優勝チーム自身が
「純粋な AI ベース手法は統計ベースラインを安定して上回らなかった」と
結論しているため。各項が独立に検査・較正できることを優先している。

## 実測で確認した 2 つの落とし穴

どちらも生物学ではなく実装の問題だが、スコアへの影響は生物学より大きかった。
いずれも本物の `cell-eval` を通した公式スコアリング経路での実測。

**1. 正規化スケール** — cell-eval は `scanpy.pp.normalize_total` を
`target_sum` なしで呼ぶ。scanpy のデフォルトは 1e4 ではなく
**中央値ライブラリサイズ**。同じ予測をスケールだけ変えて出し分けると:

| | MAE | MSE | avg_score |
|---|---|---|---|
| 1e4 スケール | 0.6474 | 0.5106 | 0.3106 |
| 中央値スケール | **0.0482** | **0.0047** | **0.3955** |

生物学的予測を 1 ビットも変えずに MAE が 13 倍、集約スコアが 27% 改善する。

**2. 疑似バルク平均の再サンプリングノイズ** — 摂動細胞をコントロールプールから
引くと、部分集合の平均のずれが疑似バルク平均に乗る。ベースラインはこの
ノイズを払っていないので一方的に不利。各アームの平均をプール全体の平均に
合わせ込むと、モデルを触らずに集約スコアが **0.2435 → 0.3955（+62%）**。

詳細は [`docs/02`](docs/02-評価指標の解剖.md)。

## クイックスタート

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[eval,dev]'
sudo apt-get install -y zstd          # cell-eval prep が要求する

.venv/bin/python -m pytest tests -q          # 13 tests、データ不要
.venv/bin/python scripts/benchmark_synthetic.py   # LOCO ベンチと alpha 掃引
```

実データでの手順は [`docs/04`](docs/04-実行手順とロードマップ.md)。

```bash
scripts/download_public_data.sh data/                 # VCC 2025 (公開)
.venv/bin/python scripts/build_library.py  --genes ... --source H1=... --out outputs/library.npz
.venv/bin/python scripts/predict.py --library outputs/library.npz ... --out outputs/submission.h5ad --prep
```

## 検証について

ゼロショットは **細胞株ごと丸ごとホールドアウト** しないと測れない
(`vcc.calibrate.evaluate_holdout`)。同一株内で摂動をホールドアウトするのは
別の、そしてずっと簡単な問題を測っており、転移が全く効いていなくても
良い数字が出る。

`tests/synthetic.py` のシミュレータの数字は **チャレンジデータではない**。
配線が正しく通っていることの証拠であって、リーダーボード性能の予測ではない。

## 状態

- パイプラインは h5ad 入力から `.vcc` 提出物まで通しで動作確認済み
  （`cell-eval prep` 0.8.2 で実際に `.vcc` を生成）。
- 公式スコアリング経路（`cell-eval baseline` → `run` → `score`）でも動作確認済み。
- **2026 の検証／テストデータには未接続** — 登録アカウント経由でのみ配布されるため。
  データを `data/vcc2026/` に置けば `scripts/predict.py` がそのまま走る。
