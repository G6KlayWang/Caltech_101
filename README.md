# Caltech-101 Mini Project (102 Classes)

This project implements a full Caltech-101 pipeline with:
- Classical baseline: `HOG + Linear SVM`
- Classical boosted model: `HOG + 9 engineered features + LightGBM`
- Transfer learning models: `ResNet-50`, `EfficientNet-B0`, `ViT-B/16` (ImageNet pretrained)

`BACKGROUND_Google` is kept, so the task uses 102 classes total.

## Project Layout

```text
caltech101_project/
  README.md
  requirements.txt
  scripts/
  configs/
  src/
  runs/
  data/
```

## Setup

```bash
cd caltech101_project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

Place Kaggle Caltech-101 files under one of:
- `data/raw/Caltech101/101_ObjectCategories/...`
- `data/raw/101_ObjectCategories/...`

The splitter auto-detects the class root and includes `BACKGROUND_Google`.

## Run Commands

```bash
bash scripts/00_prepare_data.sh
bash scripts/01_train_eval.sh --model classical_svm --exp_name baseline_classical
bash scripts/01_train_eval.sh --model resnet50 --img_size 128 --aug 1 --optimizer adam --exp_name dl_baseline
bash scripts/02_run_ablations.sh
```

## CLI Entrypoints

- Data prep / splitting:
  - `python -m src.data.make_splits --help`
- Classical train/eval:
  - `python -m src.train.train_classical --help`
- DL train/eval:
  - `python -m src.train.train_dl --help`
- Eval from saved predictions:
  - `python -m src.eval.eval_all --help`

## Output Layout

Each run is stored as:

```text
runs/{experiment_name}/{run_id}/
  config.yaml
  logs.txt
  metrics.json
  per_class_accuracy.csv
  confusion_matrix.png
  confusion_matrix_normalized.png
  per_class_accuracy.png
  training_curves.png              # DL only
  model/
  preds/
```

## Notes

- Deterministic seeds are applied to `random`, `numpy`, and `torch`.
- Deep learning uses a freeze-then-unfreeze transfer schedule.
- LightGBM engineered features are exactly 9 dimensions:
  - `width`, `height`, `width/height`, `mean(R,G,B)`, `std(R,G,B)`
