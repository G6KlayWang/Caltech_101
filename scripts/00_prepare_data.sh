#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

RAW_DIR="data/raw"
SPLITS_DIR="data/splits"
SEED="42"
DOWNLOAD_IF_MISSING="0"
KAGGLE_DATASET="jessicali9530/caltech101"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --raw_dir) RAW_DIR="$2"; shift 2 ;;
    --splits_dir) SPLITS_DIR="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --download_if_missing) DOWNLOAD_IF_MISSING="$2"; shift 2 ;;
    --kaggle_dataset) KAGGLE_DATASET="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

echo "[prepare] raw_dir=${RAW_DIR}"
echo "[prepare] splits_dir=${SPLITS_DIR}"
echo "[prepare] seed=${SEED}"

python -m src.data.make_splits \
  --raw_dir "${RAW_DIR}" \
  --splits_dir "${SPLITS_DIR}" \
  --seed "${SEED}" \
  --download_if_missing "${DOWNLOAD_IF_MISSING}" \
  --kaggle_dataset "${KAGGLE_DATASET}"

echo "[prepare] done. Generated:"
echo "  ${SPLITS_DIR}/index.csv"
echo "  ${SPLITS_DIR}/train.csv"
echo "  ${SPLITS_DIR}/val.csv"
echo "  ${SPLITS_DIR}/test.csv"
