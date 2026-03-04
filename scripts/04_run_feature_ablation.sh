#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG_PATH="configs/feature_ablation.yaml"
SEED="42"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG_PATH="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: bash scripts/04_run_feature_ablation.sh [--config PATH] [--seed INT]"
      exit 1
      ;;
  esac
done

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}"
  exit 1
fi

ABLATION_NAME="$(python - <<PY
import yaml
cfg = yaml.safe_load(open("${CONFIG_PATH}", "r", encoding="utf-8"))
print(cfg.get("ablation_name", "A4_hog_vs_cnn"))
PY
)"

EXP_ROOT="$(python - <<PY
import yaml
cfg = yaml.safe_load(open("${CONFIG_PATH}", "r", encoding="utf-8"))
print(cfg.get("exp_root", "ablations/A4_hog_vs_cnn"))
PY
)"

IMG_SIZE="$(python - <<PY
import yaml
cfg = yaml.safe_load(open("${CONFIG_PATH}", "r", encoding="utf-8"))
print(int(cfg.get("img_size", 128)))
PY
)"

AUG="$(python - <<PY
import yaml
cfg = yaml.safe_load(open("${CONFIG_PATH}", "r", encoding="utf-8"))
print(int(cfg.get("aug", 1)))
PY
)"

OPTIMIZER="$(python - <<PY
import yaml
cfg = yaml.safe_load(open("${CONFIG_PATH}", "r", encoding="utf-8"))
print(cfg.get("optimizer", "adam"))
PY
)"

readarray -t HOG_MODELS < <(python - <<PY
import yaml
cfg = yaml.safe_load(open("${CONFIG_PATH}", "r", encoding="utf-8"))
for m in cfg.get("hog_models", ["classical_svm"]):
    print(m)
PY
)

readarray -t CNN_MODELS < <(python - <<PY
import yaml
cfg = yaml.safe_load(open("${CONFIG_PATH}", "r", encoding="utf-8"))
for m in cfg.get("cnn_models", ["resnet50", "efficientnet"]):
    print(m)
PY
)

SUMMARY_DIR="runs/${EXP_ROOT}"
SUMMARY_PATH="${SUMMARY_DIR}/summary.csv"
mkdir -p "${SUMMARY_DIR}"
echo "ablation,feature_family,model,img_size,aug,optimizer,accuracy,top5,macro_f1,weighted_f1,run_dir" > "${SUMMARY_PATH}"

append_summary () {
  local run_dir="$1"
  local feature_family="$2"
  local model="$3"
  local img_size="$4"
  local aug="$5"
  local optimizer="$6"

  python - <<PY
import json, os
summary_path = r"${SUMMARY_PATH}"
run_dir = r"${run_dir}"
ablation = r"${ABLATION_NAME}"
feature_family = r"${feature_family}"
model = r"${model}"
img_size = r"${img_size}"
aug = r"${aug}"
optimizer = r"${optimizer}"
metrics_path = os.path.join(run_dir, "metrics.json")
metrics = json.load(open(metrics_path, "r", encoding="utf-8"))
row = [
    ablation,
    feature_family,
    model,
    img_size,
    aug,
    optimizer,
    str(metrics.get("accuracy", "")),
    str(metrics.get("top5_accuracy", "")),
    str(metrics.get("f1_macro", "")),
    str(metrics.get("f1_weighted", "")),
    run_dir,
]
with open(summary_path, "a", encoding="utf-8") as f:
    f.write(",".join(row) + "\\n")
PY
}

run_and_capture () {
  local feature_family="$1"
  local model="$2"
  local img_size="$3"
  local aug="$4"
  local optimizer="$5"

  local exp_name="${EXP_ROOT}/${feature_family}/${model}"
  local output
  output="$(bash scripts/01_train_eval.sh \
    --model "${model}" \
    --img_size "${img_size}" \
    --aug "${aug}" \
    --optimizer "${optimizer}" \
    --exp_name "${exp_name}" \
    --seed "${SEED}")"

  echo "${output}"
  local run_dir
  run_dir="$(echo "${output}" | awk -F'RUN_DIR=' '/RUN_DIR=/{print $2}' | tail -n1)"
  if [[ -z "${run_dir}" ]]; then
    echo "Failed to parse RUN_DIR from output for model=${model}"
    exit 1
  fi
  append_summary "${run_dir}" "${feature_family}" "${model}" "${img_size}" "${aug}" "${optimizer}"
}

for model in "${HOG_MODELS[@]}"; do
  run_and_capture "hog" "${model}" "${IMG_SIZE}" "${AUG}" "${OPTIMIZER}"
done

for model in "${CNN_MODELS[@]}"; do
  run_and_capture "cnn" "${model}" "${IMG_SIZE}" "${AUG}" "${OPTIMIZER}"
done

echo "Feature-extractor ablation complete. Summary saved to ${SUMMARY_PATH}"

