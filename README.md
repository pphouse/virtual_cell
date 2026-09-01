# Virtual Cell Challenge 2026 — CCDT

Arc Institute の [Virtual Cell Challenge 2026](https://arcinstitute.org/news/virtual-cell-challenge-2026)
への提出パイプライン。**未知の細胞株**における CRISPRi ノックダウン応答を、
その株の未摂動細胞と標的遺伝子リストだけから予測する（ゼロショット）。

締切: **2026-11-05 23:59 UTC**（最終テストセット公開 10-22）

## 現状

チャレンジ CLI に接続済み。データ取得 → 予測 → `.vcc` → 提出 → スコア取得まで
実際に通っている。

| 提出 | 内容 | Overall |
|---|---|---|
| v0 | on-target のみ（残存率 0.30） | −0.0638 |
| v1 | 排出モデル修正 + 実測残存率 0.148 | −0.3035 |
| v2 | STRING 転移（H1 300 署名）、排出モデル再設計 | 未提出（v3 に置換） |
| v3 | H1/K562/RPE1 の 2,596 署名から転移 | 採点中 |

v0→v1 の悪化が最も有益だった: **DE 系 4 指標は yield（有意遺伝子を出す量）を見ており、
何も予測しないのは棄権ではなく大きな失点**だと分かった（`docs/05 §1b`）。

転移元は 3 細胞株 4,894 署名（Arc の 2025 公開データ + Replogle K562/RPE1）。
遺伝子類似度は STRING、評価は 2025 データの leave-one-out（`docs/05 §3c`）。

## ドキュメント

| | 内容 |
|---|---|
| [`docs/01-課題整理.md`](docs/01-課題整理.md) | 課題・データ・日程 |
| [`docs/02-評価指標の解剖.md`](docs/02-評価指標の解剖.md) | 6 指標の実仕様、提出フォーマット、旧版の誤りの訂正 |
| [`docs/03-モデル設計.md`](docs/03-モデル設計.md) | CCDT の設計 |
| [`docs/04-実行手順とロードマップ.md`](docs/04-実行手順とロードマップ.md) | 実行コマンドと締切までの計画 |
| **[`docs/05-実測でわかったこと.md`](docs/05-実測でわかったこと.md)** | **実データとリーダーボードで測った結果だけ。ここが本体** |

## 手法

```
delta(g, c*) = alpha * M * [ b * D + (1 - b) * K ]   標的遺伝子はオンターゲットモデルで上書き
```

- **D** 実測署名を *fold change として* 未知株にリベース。加算的な delta は
  文脈間で保存されないが、乗算的な効果は保存される。
- **K** どのソースでも摂動されていない標的（2026 では 287/300）向けに、
  低ランク応答プログラムの負荷量を遺伝子類似度から予測して展開。
- **M** 摂動ごとの効果量。`pds` が最も敏感な項。
- **alpha** 大域スケール。

end-to-end の深層モデルにしていないのは、2025 の優勝チーム自身が
「純粋な AI ベース手法は統計ベースラインを安定して上回らなかった」と
結論しているため。各項が独立に検査・較正できることを優先している。

## 実測で分かった主要な事実

すべて `docs/05` に、再現スクリプト付きで記録。

**排出モデルにバグが 4 つあった。** 初回スコアの損失は生物学ではなく排出モデル
だった。1 つ直すたびに測って次が見つかった: 予測ゼロの遺伝子に偽 DE を撒く、
ノックダウンを希釈する、Poisson 再抽出が系統的に DOWN に偏る、
擬似カウントが up を過剰配送する。**分散と検出率を実細胞に 1% 以内で合わせても
すり抜けた** — Wilcoxon 検定は分布の形そのものを見る。今は
「予測がゼロなら恒等写像」「要求した fold change が両方向で届く」を
構成上保証し、全遺伝子を引き直しても偽 DE は 0 件。

**細胞株間転移は機能している。** 片方の株だけから作ったライブラリで
もう片方を予測すると、方向一致率は 0.60〜0.74（ベースライン 0.511）を維持し、
`pds` はほとんど劣化しない。失っているのは「正しさ」ではなく **yield**。

**転移元の署名数が、細胞株の近さより効く。** RPE1 を予測するとき、
K562 の 2,057 署名（最も遠い株）は H1 の 300 署名を大きく上回る
（pds 0.704 vs 0.598）。300 署名では STRING kNN が近傍を見つけられない。

**ノックダウン効率は転移しない。** トランス署名は生物学なので株をまたぐが、
効率は試薬の性質。Arc の CRISPRi は残存率 0.13〜0.16、Replogle は 0.54〜0.58。
プールすると数の多い方に引きずられる。

**コントロール細胞の共発現は、トランス応答を予測しない。** 実測 150 標的で
2 通り測って両方ゼロ（相関 −0.003、正は 44.7% で偶然以下）。

**`pds` は効果量に一切依存しない。** コサインなのでスケール不変。
真の方向を与えると 1.000、真の効果量を与えても変化なし。上げる道は表現の質だけ。

## クイックスタート

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
python3 -m venv .venv-cli && .venv-cli/bin/pip install vcc-cli   # 公式 CLI は別 venv
.venv/bin/python -m pytest tests -q                              # 27 tests、データ不要
```

公式 CLI を別の venv に入れているのは、`vcc` という配布名が公式 CLI のもので、
本パッケージ（`vcc2026`）と衝突するため。

```bash
export VCC_TOKEN=<your key>
.venv-cli/bin/vcc datasets download controls -d data/vcc

.venv/bin/python scripts/predict_vcc2026.py \
  --bundle data/vcc --out out/sub.h5ad --vcc out/sub.vcc \
  --library out/lib.npz --string-adj out/string_adj.npz \
  --knockdown-lines H1,H1val,H1test \
  --alpha 0.85 --magnitude-gamma 1.0 --n-components 100 \
  --neighbour-power 2.0 --n-neighbours 100

.venv-cli/bin/vcc submit out/sub.vcc -m "..." --wait
```

設定は提出ではなくオフラインで決める（1 提出は実時間 1 時間、1 日 2 回まで）:

```bash
.venv/bin/python scripts/offline_score_2025.py --library out/lib.npz \
  --string-adj out/string_adj.npz --sweep "alpha=0.7,0.85,1.0"
.venv/bin/python scripts/cross_line_transfer.py --library out/lib.npz \
  --string-adj out/string_adj.npz --case "H1+K562->RPE1"
.venv/bin/python scripts/null_emission_test.py --controls data/vcc/context_A.h5ad --out out/null.json
```

`vcc prep` はこの環境では動かない（2.06e9 stored entries に対し約 16 GiB の
常駐メモリを要求し、RAM は 15 GB）。`.vcc` は
[`vcc2026/vccfile.py`](src/vcc2026/vccfile.py) が直接ストリーミング生成し、
**本物の `vcc prep` との一致を [`tests/test_vcc_parity.py`](tests/test_vcc_parity.py)
で保証**している（小規模な提出物を両経路に通し、アーカイブから戻る AnnData を比較）。

## 検証について

ゼロショットは **文脈ごと丸ごとホールドアウト** しないと測れない。
同一株内で摂動をホールドアウトするのは別の、そしてずっと簡単な問題を測る。

`tests/synthetic.py` のシミュレータの数字は **チャレンジデータではない**。
配線が正しいことの証拠であって、リーダーボード性能の予測ではない。
