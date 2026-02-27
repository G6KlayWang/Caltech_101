#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

MODEL="classical_svm"
IMG_SIZE="128"
AUG="1"
OPTIMIZER="adam"
EXP_NAME="default_exp"
SEED="42"
SVM_USE_GPU="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --img_size) IMG_SIZE="$2"; shift 2 ;;
    --aug) AUG="$2"; shift 2 ;;
    --optimizer) OPTIMIZER="$2"; shift 2 ;;
    --exp_name) EXP_NAME="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --svm_use_gpu) SVM_USE_GPU="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ "${MODEL}" == "classical_svm" ]]; then
  python -m src.train.train_classical \
    --model "${MODEL}" \
    --exp_name "${EXP_NAME}" \
    --seed "${SEED}" \
    --svm_use_gpu "${SVM_USE_GPU}"
elif [[ "${MODEL}" == "classical_lgbm" ]]; then
  python -m src.train.train_classical \
    --model "${MODEL}" \
    --exp_name "${EXP_NAME}" \
    --seed "${SEED}"
elif [[ "${MODEL}" == "resnet50" || "${MODEL}" == "efficientnet" || "${MODEL}" == "vit" ]]; then
  python -m src.train.train_dl \
    --model "${MODEL}" \
    --img_size "${IMG_SIZE}" \
    --aug "${AUG}" \
    --optimizer "${OPTIMIZER}" \
    --exp_name "${EXP_NAME}" \
    --seed "${SEED}"
else
  echo "Unsupported model: ${MODEL}"
  exit 1
fi
