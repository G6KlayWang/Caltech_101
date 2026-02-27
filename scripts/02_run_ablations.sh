#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG_PATH="configs/ablations.yaml"
SEED="42"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG_PATH="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

readarray -t MODELS < <(python - <<PY
import yaml
cfg = yaml.safe_load(open("${CONFIG_PATH}", "r", encoding="utf-8"))
for m in cfg["models"]:
    print(m)
PY
)

OPT_MODEL="$(python - <<PY
import yaml
cfg = yaml.safe_load(open("${CONFIG_PATH}", "r", encoding="utf-8"))
print(cfg.get("optimizer_ablation_model", "resnet50"))
PY
)"

SUMMARY_PATH="runs/ablations/summary.csv"
mkdir -p "runs/ablations"
echo "ablation,model,img_size,aug,optimizer,accuracy,top5,macro_f1,weighted_f1" > "${SUMMARY_PATH}"

append_summary () {
  local run_dir="$1"
  local ablation="$2"
  local model="$3"
  local img_size="$4"
  local aug="$5"
  local optimizer="$6"

  python - <<PY
import json, os
summary_path = r"${SUMMARY_PATH}"
run_dir = r"${run_dir}"
ablation = r"${ablation}"
model = r"${model}"
img_size = r"${img_size}"
aug = r"${aug}"
optimizer = r"${optimizer}"
metrics_path = os.path.join(run_dir, "metrics.json")
metrics = json.load(open(metrics_path, "r", encoding="utf-8"))
row = [
    ablation,
    model,
    img_size,
    aug,
    optimizer,
    str(metrics.get("accuracy", "")),
    str(metrics.get("top5_accuracy", "")),
    str(metrics.get("f1_macro", "")),
    str(metrics.get("f1_weighted", "")),
]
with open(summary_path, "a", encoding="utf-8") as f:
    f.write(",".join(row) + "\\n")
PY
}

run_and_capture () {
  local ablation="$1"
  local model="$2"
  local img_size="$3"
  local aug="$4"
  local optimizer="$5"

  local exp_name="ablations/${ablation}/${model}"
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
    echo "Failed to parse RUN_DIR from output"
    exit 1
  fi
  append_summary "${run_dir}" "${ablation}" "${model}" "${img_size}" "${aug}" "${optimizer}"
}

# A1: image size 64 vs 128, aug ON, optimizer Adam
for model in "${MODELS[@]}"; do
  run_and_capture "A1_img_size" "${model}" "64" "1" "adam"
  run_and_capture "A1_img_size" "${model}" "128" "1" "adam"
done

# A2: augmentation OFF vs ON, size 128, optimizer Adam
for model in "${MODELS[@]}"; do
  run_and_capture "A2_augmentation" "${model}" "128" "0" "adam"
  run_and_capture "A2_augmentation" "${model}" "128" "1" "adam"
done

# A3: optimizer Adam vs SGD, one model family
run_and_capture "A3_optimizer" "${OPT_MODEL}" "128" "1" "adam"
run_and_capture "A3_optimizer" "${OPT_MODEL}" "128" "1" "sgd"

echo "Ablations complete. Summary saved to ${SUMMARY_PATH}"
