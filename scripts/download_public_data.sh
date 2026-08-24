#!/usr/bin/env bash
# Fetch the *public* Arc Virtual Cell Atlas data usable as training sources.
#
#   scripts/download_public_data.sh data/
#
# The 2026 validation and test sets are NOT here -- they are distributed
# through a registered account at https://virtualcellchallenge.org/.  Only the
# 2025 release is in the open bucket, and it is still the single most useful
# CRISPRi source for this task.
set -euo pipefail

DEST=${1:-data}
BASE="https://storage.googleapis.com/arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025"

mkdir -p "$DEST/vcc2025"
echo "Downloading the 2025 Virtual Cell Challenge release into $DEST/vcc2025"
echo "  train h5ad is ~15 GB and validation ~7 GB -- check free disk first."

for f in \
  "gene_names.csv" \
  "train/adata_Training.h5ad" \
  "train/pert_counts_Training.csv" \
  "validation/adata_Validation.h5ad" \
  "validation/pert_counts_Validation.csv" \
  "test/adata_Test.h5ad" \
  "test/pert_counts_Test.csv"
do
  out="$DEST/vcc2025/$(basename "$f")"
  if [[ -f "$out" ]]; then
    echo "  have $(basename "$f")"
    continue
  fi
  echo "  fetching $f"
  curl -fL --retry 4 --retry-delay 2 -o "$out.part" "$BASE/$f"
  mv "$out.part" "$out"
done

cat <<'NOTE'

Done.  Other sources worth adding as extra cell-line contexts (they are what
make the zero-shot transfer possible at all):

  Replogle 2022 K562 / RPE1 / HepG2 genome-wide CRISPRi
  Tahoe-100M (drug perturbations across ~50 lines)

Both are listed at https://github.com/ArcInstitute/arc-virtual-cell-atlas
Rename their perturbation column to `target_gene` and their control label to
`non-targeting`, or pass --pert-col / --control-name to build_library.py.
NOTE
