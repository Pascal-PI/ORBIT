# ORBIT transfer-learning comparison

## Protocol

The starting deployment model is the custom ORBIT CNN trained on 1,600 images, with 85% accuracy (170/200) on the existing fixed test partition. The custom model and its original training script remain available for the dataset-size science-fair experiment.

This comparison tests **MobileNetV3-Large**, initialized with Torchvision's **IMAGENET1K_V2** weights. It is a deployment experiment with external ImageNet pretraining, not another from-scratch CNN data-size result.

- Dataset: the unchanged 2,706 source images and original split fingerprint.
- Training: all 2,406 training-pool images (846 Plastic, 715 Metal, 425 Paper, 420 Trash); inverse-frequency class-weighted training loss.
- Validation: the same 100 images; unweighted cross-entropy for checkpoint selection and comparison.
- Test: the same 200 images, evaluated only after training and validation checkpoint selection finish. This test partition has been used in previous development rounds; it is a reused benchmark, not a fresh independent evaluation.
- Input: pretrained V2 evaluation transform (resize the shorter side to 232, center crop 224, ImageNet mean/std normalization). Fine-tuning uses random crops at scale 0.8–1.0, horizontal flips, and mild color jitter.
- Stage 1: freeze the feature extractor and train the final four-way linear layer for 60 epochs on cached original and horizontally mirrored features.
- Stage 2: fine-tune the last three feature blocks and classifier for eight epochs. Preserve BatchNorm running statistics. Use AdamW with initial feature/classifier learning rates 0.0001/0.0003 and validation-based reductions.
- Choose the checkpoint with the lowest validation loss across both stages. Promote it only if its validation loss beats the current model on the same split, independently of the new test score.
- Training seed: 42. CPU threads: 4. Batch size: 32 (128 for the cached linear-head stage).

## Measured result

The selected model scored **184/200 correct (92%)**, up from **170/200 (85%)** for the previous custom CNN: **7 percentage points**, 14 more correct predictions, and 16 errors instead of 30. Every category improved in this benchmark.

| Category | Previous test recall | New test recall |
| --- | ---: | ---: |
| Plastic | 88% | 90% |
| Metal | 84% | 90% |
| Paper | 94% | 100% |
| Trash | 74% | 88% |

Validation loss selected **fine-tuning epoch 1**, with validation accuracy **91%** and loss **0.2093** (previous loss 0.3702). The peak validation accuracy across all epochs was 94%; it was not the selection criterion. Test loss was 0.2205. Training and feature extraction took about 31 minutes on CPU, excluding initial weight download and dataset scanning.

The model was promoted by the predeclared lower-validation-loss rule, independently of its test score. It is now the default `artifacts/orbit_cnn.pt`; the legacy filename is retained for compatibility. Inference reconstructs MobileNet and loads every weight from that checkpoint without downloading ImageNet weights. The default checkpoint is about 16.2 MiB, versus 0.6 MiB for the previous CNN; Raspberry Pi latency has not been measured.

Among the same test images, 22 previous mistakes were corrected and 8 previously correct predictions became mistakes. The saved predictions reproduce all 184 correct answers. The selected checkpoint matches its run copy byte-for-byte, the split/dataset fingerprints match the baseline, and all 2,406 selected training images are disjoint from validation/test images. The prediction CLI reproduces the saved prediction for `plastic/Plastic_263.jpg`. All five compatibility/workflow/guard tests and `pip check` passed.

Run records: `artifacts/transfer/transfer_20261108T230433885034Z/`. Verification: `artifacts/transfer_improvement.json`.

To compare the preserved custom CNN:

```powershell
.\.venv\Scripts\python.exe predict.py plastic\Plastic_263.jpg --checkpoint artifacts\runs\n1600_seed42_20261108T140412751639Z\best.pt
```

Normal `train.py` runs now preserve a better default on the same validation split. Every candidate remains saved; `--replace-default` explicitly overrides this guard.

## Reproduce

```powershell
.\.venv\Scripts\python.exe train_transfer.py --promote
```

The first run downloads the official pretrained weights. Inference from a saved ORBIT checkpoint does not download weights or call an API. The old custom-CNN checkpoint remains at `artifacts/runs/n1600_seed42_20261108T140412751639Z/best.pt`.

## Sources

- [Torchvision MobileNetV3-Large model and pretrained transforms](https://docs.pytorch.org/vision/0.22/models/generated/torchvision.models.mobilenet_v3_large.html)
- [PyTorch transfer-learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

## Limits

Using all training images, a different architecture, higher input resolution, and external pretraining changes several factors together. The result cannot isolate their individual contributions. The validation set has only 100 images, and this is one training seed. New camera images and a genuinely new held-out dataset are needed to estimate robot deployment accuracy. Confidence values remain uncalibrated softmax scores.
