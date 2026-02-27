#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

EXP_NAME="baseline_classical"
SEED="42"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp_name) EXP_NAME="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: bash scripts/03_train_baseline.sh [--exp_name NAME] [--seed INT]"
      exit 1
      ;;
  esac
done

python -m src.train.train_classical \
  --model classical_svm \
  --exp_name "${EXP_NAME}" \
  --seed "${SEED}"
