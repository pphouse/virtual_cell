# Virtual Cell Challenge 2026 — CCDT

Arc Institute の [Virtual Cell Challenge 2026](https://arcinstitute.org/news/virtual-cell-challenge-2026)
への提出パイプライン。**未知の細胞株**における CRISPRi ノックダウン応答を、
その株の未摂動細胞と標的遺伝子リストだけから予測する（ゼロショット）。

締切: **2026-11-05 23:59 UTC**（最終テストセット公開 10-22）

## 現状

チャレンジ CLI に接続済み。データ取得 → 予測 → `.vcc` → 提出 → スコア取得まで
実際に通っている。

| 提出 | 内容 | Overall | 備考 |
|---|---|---|---|
| v0 | on-target のみ（残存率 0.30） | **−0.0638** | 損失はほぼ全て `fid` (−0.342)。排出モデルのバグを暴いた |
| v1 | 排出モデル修正 + 実測残存率 0.148 | 採点中 | |

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

**排出モデルが偽の DE を撒いていた。** 初回提出は、摂動を予測していない
遺伝子に対して全 300 摂動 × 3 文脈で同じ向きの偽 DE を約 93 個ずつ渡していた。
実際のノックダウン応答は down が優勢なので、これだけで方向一致率が偶然を下回り
`fid = −0.34` を生んでいた。**分散と検出率を実細胞に 1% 以内で合わせても
すり抜けた** — Wilcoxon 検定は分布の形そのものを見る。
今は「予測がゼロなら排出は恒等写像」を構成上保証している。

**コントロール細胞の共発現は、トランス応答を予測しない。** 2025 の実測 150 標的で
2 通り測って両方ゼロ。相関の平均 −0.003、正だった摂動は 44.7%（偶然以下）。
leave-one-out では「全摂動の平均署名を出すだけ」に負ける（r 0.165 vs 0.188、
discrimination 0.509 vs 0.500）。

**ボトルネックは遺伝子表現で、モデルの形ではない。** 同じ枠組みに真の署名類似度
（oracle）を入れると r=0.471 / discrimination 0.748 に届く。STRING の機能的
相互作用ネットワークは本物の改善（r=0.235 / discrimination 0.584）だが、
oracle にはまだ遠い。

**CRISPRi の実効率は残存率 0.148**（150/150 が下がる）。文献値 0.30 は 2 倍高すぎた。

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
  --library out/lib_h1.npz --string-adj out/string_adj.npz

.venv-cli/bin/vcc submit out/sub.vcc -m "..." --wait
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
