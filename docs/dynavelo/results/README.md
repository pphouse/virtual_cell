# DynaVelo の結果ファイル

`docs/11-DynaVeloの実行結果.md` の元データ。GitHub の 1 ファイル 100 MB 制限に収めるため、
一部を間引き・圧縮してある（gzip 圧縮の h5ad）。

| ファイル | 内容 |
|---|---|
| `adata_rna_pred_slim.h5ad` | 5,225 細胞 × 3,381 遺伝子。`X` = log1p 発現（疎）、`layers`: `x_pred_mean`（再構成）・`vx_pred_mean`（DynaVelo の RNA 速度）は float16、`velocity_scvelo`（scVelo の速度、推定できなかった遺伝子は 0。`var['has_scvelo_velocity']` で判別）。`obs`: 細胞種・潜在時間（`latent_time_mean` / `latent_time_scvelo`）・`n_contextA_cells`。`obsm`: UMAP、潜在表現 `z_mean`（RNA 50 + モチーフ 50）、`vz_mean`、TF ノックダウンごとの潜在時間変化 |
| `adata_motif_pred.h5ad` | 5,225 細胞 × 171 TF。`X` = chromVAR z-score（平滑化済み）、`layers`: `z_raw`・`y_pred_mean`・`vy_pred_mean`（モチーフ速度）ほか |
| `contextA_dynavelo_perturbation.h5ad` | 3,381 摂動 × 3,381 遺伝子。context A 対応細胞で重み付け平均した in-silico ノックダウン。`X` = デコード発現の変化、`layers['delta_velocity']`、`obsm` = モチーフ活性・モチーフ速度の変化 |
| `dynavelo_pbmc_v1.pth` | 上の 3 ファイルを作った学習済みモデル（3,381 遺伝子 × 171 TF） |
| `dynavelo_pbmc_v2_with_targets.pth` | 2026 の 300 標的と 2025 検証標的を遺伝子セットに加えて再学習したモデル（3,625 遺伝子 × 189 TF） |
| `dynavelo_unit_response.npz` | v2 モデルで、346 標的それぞれの入力発現を 1 だけ下げたときのデコード発現の変化（346 × 3,625） |
| `dynavelo_gene_features.npz` | 上を遺伝子方向に中心化し、第 1 成分を除いた 50 次元の遺伝子特徴（`vcc2026.coessentiality.load_features` で読める） |

落としたもの: 生カウント・spliced/unspliced・scVelo のモーメントと当てはめ結果、`vx_pred_std`（速度の不確かさ）。
`dynavelo/` のスクリプトで再生成できる。
