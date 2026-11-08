# ORBIT training and testing examples

Your full dataset is already in the ORBIT project: 921 plastic, 790 metal, 500 paper, and 495 general-trash images (2,706 total).

## Try a photo the model has not trained on

`unseen_test` contains 16 copies from the existing fixed test partition: four per category. Selection uses the first four saved test images per class, without choosing easy or correctly predicted examples. `labels.csv` gives the correct categories and original file names. These photos are excluded from every training subset that uses the current saved split.

From PowerShell:

```powershell
cd "$env:USERPROFILE\OneDrive\Coding Projects\ORBIT"
.\.venv\Scripts\python.exe predict.py ".\examples\unseen_test\Plastic_263.jpg"
```

Try the other photos by replacing the image filename. Compare each prediction with `labels.csv`. Some predictions will be wrong. The initial 100-image baseline scored 52% and the later 400-image model scored 65% on the full 200-image test set. See `../ACCURACY_REPORT.md` for the latest result. Confidence scores are uncalibrated model scores.

## Train on the existing full dataset

```powershell
# First training run: 100 total images, 25 per category.
.\.venv\Scripts\python.exe train.py --train-size 100 --epochs 15

# A larger run: 400 total images, 100 per category.
.\.venv\Scripts\python.exe train.py --train-size 400 --epochs 15

# Largest planned run: 1,600 total images, 400 per category.
.\.venv\Scripts\python.exe train.py --train-size 1600 --epochs 30 --lr-schedule plateau
```

Run whichever command you want to try. Each run automatically evaluates its validation-selected checkpoint on the same 200 held-out test images and prints test accuracy. The original images stay in the four category folders. The `examples` folder is only for convenient manual testing; do not pass it as `--data-dir`.

Keep the test photos separate from training. `artifacts/splits.json` already reserves 100 validation images and 200 test images. After a normal training run, `predict.py` uses that run's saved model. Earlier per-run models and metrics remain under `artifacts/runs`.

## Run the planned repeated experiment

```powershell
.\.venv\Scripts\python.exe train.py --experiment --sizes 100 200 400 800 1600 --repeats 3 --epochs 15
```

This runs 15 independent training jobs and saves CSV summaries and an accuracy graph under `artifacts/experiments`. Keep the test results out of model/hyperparameter tuning.
