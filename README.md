# ORBIT Trash Predictor

ORBIT is a local PyTorch classifier with a custom CNN and an optional pretrained MobileNetV3 deployment model. Inference uses no API keys or paid services.

### Current model

The default model is now **MobileNetV3-Large with ImageNet pretraining**, trained on all **2,406 training-pool images**. It scores **92% (184/200)** on the existing fixed benchmark, up from **85% (170/200)** for the preserved custom CNN. Test recall: Plastic **90%**, Metal **90%**, Paper **100% (50/50)**, Trash **88%**. Validation loss selected the first fine-tuning epoch from the completed 60-epoch head / 8-epoch fine-tuning run.

This is one training seed on a reused benchmark, not a fresh independent test or robot-camera accuracy claim. The pretrained deployment comparison is separate from the original custom-CNN dataset-size experiment. See [TRANSFER_LEARNING_REPORT.md](TRANSFER_LEARNING_REPORT.md) and `artifacts/transfer_improvement.json` for the protocol and verification. `predict.py` uses the new model automatically; the old 85% checkpoint remains in its run folder.

### Model history

The original 100-image, 15-epoch baseline scored **52%** on the fixed 200-image test set. The later 400-image, 15-epoch model scored **65%** (Plastic 84%, Metal 38%, Paper 80%, Trash 58%). Both use training seed 42 and split seed 1729. See `ACCURACY_REPORT.md` for the historical custom-CNN diagnosis and `TRANSFER_LEARNING_REPORT.md` for the current deployment result; `artifacts/latest_run.json` records the current default model. The previous model and source files are preserved in `artifacts/before_accuracy_improvement/`.

Five integration checks passed, covering saved-model loading and an unseen-image prediction, disjoint/nested splits, duplicate/conflicting labels, split reuse and dataset changes, and clear errors for missing images. A separate four-run, one-epoch experiment verified CSV aggregation and graph generation; those short runs are workflow checks, not the final experiment results. Details are in `artifacts/verification.json`.

## Get the project from GitHub

```powershell
git clone https://github.com/Pascal-PI/ORBIT.git
cd ORBIT
```

This private repository includes the 92% deployment checkpoint and preserved 85% custom-CNN checkpoint, fixed split, baseline and improved run results, and the four image folders. Install the Python dependencies using the instructions below, then run `predict.py`; retraining is optional. The `.venv` environment and local backup folder are excluded from Git. Recorded run JSON files retain the original machine paths as provenance; image paths inside the split and training manifests are relative to the project.

The source images are from RealWaste. See [DATASET_NOTICE.md](DATASET_NOTICE.md) for attribution and the dataset license notice.

## Run ORBIT

Use Python 3.12. The project has its own `.venv`; activation is optional when using its Python executable directly.

After installing dependencies, use the included 92% model from the project folder:

```powershell
.\.venv\Scripts\python.exe predict.py plastic\Plastic_263.jpg --json
```

Predict a photo from the fixed, unseen test set:

```powershell
$testImage = (Get-Content .\artifacts\splits.json -Raw | ConvertFrom-Json).test[0].path
.\.venv\Scripts\python.exe predict.py $testImage
```

Predict your own photo, or return JSON for a future dashboard:

```powershell
.\.venv\Scripts\python.exe predict.py "C:\path\to\new-photo.jpg" --json
```

`predict.py` returns one of **Plastic, Metal, Paper, Trash**, a confidence score, and all four class probabilities. Confidence is an **uncalibrated softmax score**, not a proven probability that the answer is correct. For a Raspberry Pi, start with `--device cpu --threads 2`.

To reproduce the preserved 85% custom-CNN training configuration (the better deployment default is kept):

```powershell
.\.venv\Scripts\python.exe train.py --train-size 1600 --epochs 30 --lr-schedule plateau
```

A normal `train.py` run preserves a better existing default; every candidate is still saved in its own run folder. A plain `train.py` command uses 1,600 training images and 30 epochs. The learning rate remains constant unless `--lr-schedule plateau` is supplied; that option halves it after validation loss stalls for four epochs (minimum 0.000001). Learning rates are recorded in each run's `history.csv`, and the schedule name is recorded in `metrics.json`. Checkpoint selection still uses validation loss, with test evaluation afterward. To reproduce the old dataset-size protocol, specify `--epochs 15 --lr-schedule constant` for every size.

### Install on another computer

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The CPU setup works without an NVIDIA GPU. On a Raspberry Pi, use a compatible native PyTorch wheel for the Pi's operating system and Python version, then install the remaining requirements. GPU installation, camera capture, and servo control are future deployment tasks.

### Files and image folders

| File | Purpose |
| --- | --- |
| `model.py` | Custom CNN and MobileNetV3 architectures, their matching preprocessing, and checkpoint loading |
| `train.py` | Fixed dataset splits, balanced training subsets, validation, held-out testing, checkpoints, repeated experiments, and plots |
| `predict.py` | Image classification with readable output or JSON; uses the checkpoint’s architecture and preprocessing |
| `train_transfer.py` | Separate pretrained MobileNetV3 deployment comparison using the existing fixed split |
| `requirements.txt` | Reproducible dependency versions |

The data stays in the existing `plastic`, `metal`, `paper`, and `general trash` folders. The last folder maps to the output label **Trash**. Training does not move or modify the original images.

### Training, validation, and test separation

- The default split reserves **25 validation images per class** and **50 test images per class**: 100 validation and 200 test images in total.
- Exact duplicate files are excluded before splitting; identical images with conflicting labels stop training with an error. Visually similar photos are not automatically detected, so review image groups before making final research claims.
- `artifacts/splits.json` records the file names, labels, content hashes, and fixed split seed. Existing splits are reused. Modified data or changed split settings require a separate `--split-file` for a new experiment.
- Training sizes are **total images across all four categories**, in multiples of four. A 100-image run uses 25 per class; 1,600 uses 400 per class. Smaller subsets are nested inside larger ones.
- Only training images receive random flips and mild brightness/contrast changes. Validation and test preprocessing is deterministic.
- The checkpoint is selected using the **lowest validation loss**. The test set is evaluated afterward, once per completed run. Keep test results out of hyperparameter and architecture selection.

Training seeds control initialization, shuffling, and augmentation. The separate split seed keeps the held-out images unchanged. Reproducibility is best within the same package versions, hardware, and device settings.

### Run the dataset-size experiment

```powershell
.\.venv\Scripts\python.exe train.py --experiment --sizes 100 200 400 800 1600 --repeats 3 --epochs 15
```

Each run starts a fresh CNN. The experiment uses the same saved validation/test sets and nested training subsets. Repeats vary the training seed while keeping the data fixed. All runs use the same requested epochs, batch size, learning rate, and augmentation rules.

Results go into a new `artifacts/experiments/<timestamp>` folder:

- `runs.csv`: every run's training size, seed, validation accuracy, and test accuracy.
- `summary.csv` and `summary.json`: mean accuracy and sample standard deviation for each training size.
- `accuracy_vs_dataset_size.png`: mean test accuracy with error bars of one sample standard deviation. These show variability across training seeds, not confidence intervals.
- Each run's `best.pt`, `history.csv`, `metrics.json`, `test_predictions.csv`, and `train_images.json`.

Single training runs create the same per-run files under `artifacts/runs/`. They update `artifacts/orbit_cnn.pt` (the default for `predict.py`) and `artifacts/latest_run.json` only when no default exists or validation loss improves on the same split. Use `--replace-default` to explicitly override this guard. Earlier run folders are retained. Experiment runs do not replace the default prediction checkpoint; use `predict.py --checkpoint "path\to\best.pt"` to inspect a particular experiment model.

Run `.\.venv\Scripts\python.exe -m unittest discover -s tests -v` for the isolated workflow check. Run `train.py --help` or `predict.py --help` for options. No inference results are fabricated: `predict.py` requires an actual saved checkpoint. A working prediction is the first milestone; measured accuracy determines how useful the model is.

## Pretrained deployment comparison

```powershell
.\.venv\Scripts\python.exe train_transfer.py --promote
```

This trains a pretrained MobileNetV3-Large candidate on all 2,406 training-pool images, using class-balanced loss, a cached-feature classifier stage, and fine-tuning. It promotes the candidate only if validation loss beats the current default on the same split. The first training run downloads the official ImageNet weights; prediction loads all weights from the saved checkpoint and works offline. See [TRANSFER_LEARNING_REPORT.md](TRANSFER_LEARNING_REPORT.md) for the protocol and measured results.

Use `train.py` for the original custom-CNN dataset-size experiment. External ImageNet pretraining changes the research condition, so keep the MobileNet comparison separate from that experiment. Legacy CNN checkpoints remain supported with their original 128-pixel preprocessing; MobileNet checkpoints use their required 224-pixel preprocessing automatically.

## 🚀 Our model development roadmap

### Phase 1 — Build ORBIT-CNN

Write our own neural network in PyTorch, using convolutional layers to learn visual features and a classification layer for our four waste categories.

### Phase 2 — Train the network

Train it on RealWaste images, starting with 100 training images and gradually increasing to 1,600.

### Phase 3 — Conduct our experiment

Evaluate each model on the same unseen test dataset. Repeat training runs, calculate average accuracy, and plot how accuracy changes with dataset size.

### Phase 4 — Build the ORBIT dashboard

Display camera images, AI predictions, confidence scores, and experiment graphs.

### Phase 5 — Deploy to the robot

Save the trained model, run inference on the Raspberry Pi, and send the correct commands to the servo mechanism. Test accuracy on real objects.

## 🧪 What our first model should look like

**Input Image — 128 × 128 RGB**

**Convolutional Neural Network**

3 convolutional blocks + pooling

**Classification Layer — 4 outputs**

- Plastic
- Metal
- Paper
- Trash

## 🎯 What we should accomplish right now

I recommend starting ORBIT AI v0.1, with three Python files:

| File | Purpose |
| --- | --- |
| `model.py` | Defines our custom CNN |
| `train.py` | Trains the model on our images |
| `predict.py` | Classifies a new image |

**Our first success criterion:**

Give ORBIT an image it hasn't trained on → get a category prediction and confidence score.

Then we'll scientifically evaluate whether those predictions are correct.

We should also keep our training, validation, and test sets separate from the beginning. Otherwise, the accuracy results won't mean much.

My recommendation: Build the custom CNN first, conduct the dataset-size experiment with it, and then investigate whether MobileNetV3 performs better for deployment on the physical robot. That keeps our science fair research focused while still giving us a path toward a powerful working system. 🔥
