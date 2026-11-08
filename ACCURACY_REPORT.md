# ORBIT accuracy investigation — November 8, 2026

## Starting point

The active model at the start of this investigation was the 400-image, 15-epoch custom CNN, not the 100-image model described in the old README. Its saved test result was 130/200 correct (65%). The original 100-image baseline scored 52%.

| Category | Previous test recall | Correct / total |
| --- | ---: | ---: |
| Plastic | 84% | 42 / 50 |
| Metal | 38% | 19 / 50 |
| Paper | 80% | 40 / 50 |
| Trash | 58% | 29 / 50 |

## What the evidence suggests

- **Limited training data:** The dataset contains 2,706 unique images. The fixed split reserves 100 for validation and 200 for testing, leaving 2,406 available for training. The active model used only 400 (100 per category). The original CLI default used only 100 total. The recorded 52% and 65% results are consistent with more data helping, though two runs do not isolate every source of variation.
- **Limited training budget:** The old run ended at 15 epochs with 72.25% training accuracy. Its lowest validation loss was at epoch 14. This motivated testing a longer run, rather than assuming the network had converged.
- **Metal confusion:** Of 50 metal images, 13 were predicted as plastic and 16 as trash. That accounts for 29 of the old model's 70 errors. Images show appearance, rather than directly measuring material; the exact visual features responsible have not been isolated.
- **Model limitations:** ORBIT learns a small three-block CNN from scratch on 128-by-128 images. Resizing discards fine detail. Whether a pretrained model or larger inputs would help this dataset requires a separate measured comparison; this change retains the existing research architecture.
- **Confidence is not accuracy:** Eight of the previous model's incorrect test predictions had confidence above 80%. No probability calibration has been performed.
- **Documentation and environment:** The README still advertised the older 52% result. The Python environment was also missing `torch_cpu.dll` and `torch_python.dll`, preventing PyTorch import. Reinstalling the pinned CPU PyTorch package restored inference/training. This installation problem prevents execution; it does not explain the old measured 65% accuracy.

## Changes tested

1. Train on 1,600 images, balanced at 400 per category, from the original nested training pool.
2. Train for 30 epochs, with a validation-loss-based learning-rate schedule: halve the rate after four epochs without improvement, down to 0.000001.
3. Select the checkpoint with minimum validation loss, then evaluate the unchanged 200-image test set once.
4. Change CLI defaults to 1,600 images and 30 epochs. Constant learning rate remains the default for compatibility; the improved command explicitly chooses `plateau`.
5. Preserve the original model, latest-run metadata, and source files in `artifacts/before_accuracy_improvement/`.
6. Add `tests/test_workflow.py`, which uses temporary synthetic images to verify scheduled training, checkpoint reload, inference, probability normalization, and training/holdout separation. Its synthetic accuracy is not an ORBIT benchmark.

Reproduce the improved run:

```powershell
.\.venv\Scripts\python.exe train.py --train-size 1600 --epochs 30 --lr-schedule plateau
```

## Measured result

The improved default model scored **170/200 correct (85%)**, versus **130/200 (65%)** for the previous model: a gain of **20 percentage points**, with 40 fewer errors. Every category improved.

| Category | Previous test recall | Improved test recall |
| --- | ---: | ---: |
| Plastic | 84% | 88% |
| Metal | 38% | 84% |
| Paper | 80% | 94% |
| Trash | 58% | 74% |

Validation loss selected epoch 28 (validation accuracy 85%, validation loss 0.3702); test loss was 0.4563. The run used learning rates 0.001 and 0.0005. Training took about 34 minutes on this CPU. The selected checkpoint now lives at `artifacts/orbit_cnn.pt` and is used by `predict.py` automatically.

The new default checkpoint is byte-identical to the selected run's `best.pt`. Dataset and split fingerprints match the previous run. The 200 saved test predictions reproduce the reported 170 correct answers, and the prediction CLI reproduces the saved prediction for `plastic/Plastic_263.jpg`. The temporary-data workflow integration test passed, as did `pip check`. Verification details are in `artifacts/accuracy_improvement.json`; the older `artifacts/verification.json` remains a historical baseline record.

The main remaining weakness is Trash (74%, 13 errors out of 50). This run did not investigate label quality or evaluate a pretrained alternative. Real camera accuracy is still unmeasured.

Run artifacts: `artifacts/runs/n1600_seed42_20261108T140412751639Z/` (checkpoint, training history, metrics, predictions, and training-image manifest).

## Measurement limits

This is one training seed on an existing held-out partition, not a completed repeated science-fair experiment. More images, more epochs, and the schedule change together; this run cannot attribute gains to any one of those changes. Use a fixed configuration across all sizes and repeated seeds for the dataset-size experiment. The dataset benchmark does not establish accuracy on a robot camera. Collect separately labeled camera images and evaluate them before making deployment claims.
