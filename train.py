"""Train ORBIT-CNN or repeat the dataset-size experiment with fixed held-out sets."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import statistics
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model import (
    ARCHITECTURE, CLASS_FOLDERS, CLASS_NAMES, IMAGE_SIZE, PROJECT_DIR,
    ORBITCNN, image_tensor, load_checkpoint, select_device,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def scan_images(data_dir):
    """Check images and remove exact duplicates before creating any split."""
    records, seen, duplicates = [], {}, 0
    for label, folder in enumerate(CLASS_FOLDERS):
        directory = data_dir / folder
        if not directory.is_dir():
            raise ValueError(f"Missing category folder: {directory}")
        paths = sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if not paths:
            raise ValueError(f"No images in {directory}")
        for path in paths:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in seen:
                if seen[digest]["label"] != label:
                    raise ValueError(f"Identical images have conflicting labels: {path} and {seen[digest]['path']}")
                duplicates += 1
                continue
            with Image.open(path) as image:
                image.verify()
            record = {"path": path.relative_to(data_dir).as_posix(), "label": label, "sha256": digest}
            seen[digest] = record
            records.append(record)
    return records, duplicates


def validate_splits(splits, records):
    expected = {record["path"]: record for record in records}
    visited_paths, visited_hashes = set(), set()
    for name in ("train_pool", "validation", "test"):
        for record in splits[name]:
            if expected.get(record["path"]) != record:
                raise ValueError("The saved split contains an unknown or modified image.")
            if record["path"] in visited_paths or record["sha256"] in visited_hashes:
                raise ValueError("The saved split contains duplicate images or overlap between sets.")
            visited_paths.add(record["path"])
            visited_hashes.add(record["sha256"])
        for label in range(len(CLASS_NAMES)):
            count = sum(record["label"] == label for record in splits[name])
            requested = {"validation": splits["validation_per_class"], "test": splits["test_per_class"]}.get(name)
            if (requested is not None and count != requested) or count == 0:
                raise ValueError(f"Invalid {name} count for {CLASS_NAMES[label]}.")
    if visited_paths != set(expected):
        raise ValueError("The saved split does not cover the current dataset.")


def prepare_splits(data_dir, split_path, seed=1729, validation_per_class=25, test_per_class=50):
    records, duplicates = scan_images(data_dir)
    settings = {
        "version": 1, "dataset_fingerprint": fingerprint(records),
        "class_names": list(CLASS_NAMES), "split_seed": seed,
        "validation_per_class": validation_per_class, "test_per_class": test_per_class,
    }
    if split_path.exists():
        splits = json.loads(split_path.read_text(encoding="utf-8"))
        if any(splits.get(key) != value for key, value in settings.items()):
            raise ValueError(
                "The dataset or split settings changed. Keep the existing split for its experiment; "
                "use a different --split-file for a new experiment."
            )
    else:
        splits = {**settings, "duplicates_excluded": duplicates, "train_pool": [], "validation": [], "test": []}
        for label, name in enumerate(CLASS_NAMES):
            category = [record for record in records if record["label"] == label]
            if len(category) <= validation_per_class + test_per_class:
                raise ValueError(f"Not enough unique {name} images after reserving validation and test images.")
            random.Random(seed + label).shuffle(category)
            end = test_per_class + validation_per_class
            splits["test"].extend(category[:test_per_class])
            splits["validation"].extend(category[test_per_class:end])
            splits["train_pool"].extend(category[end:])
        validate_splits(splits, records)
        save_json(split_path, splits)
    validate_splits(splits, records)
    print(f"Dataset: {len(records)} unique images; {duplicates} exact duplicates excluded", flush=True)
    print(f"Fixed holdouts: {len(splits['validation'])} validation, {len(splits['test'])} test", flush=True)
    return splits


def balanced_subset(splits, size):
    if size < len(CLASS_NAMES) or size % len(CLASS_NAMES):
        raise ValueError("Training size must be a positive multiple of 4 (total images across all categories).")
    selected = []
    for label, name in enumerate(CLASS_NAMES):
        available = [record for record in splits["train_pool"] if record["label"] == label]
        if len(available) < size // len(CLASS_NAMES):
            maximum = min(sum(r["label"] == i for r in splits["train_pool"]) for i in range(4)) * 4
            raise ValueError(f"Not enough {name} training images for {size}; maximum balanced training size is {maximum}.")
        selected.extend(available[:size // len(CLASS_NAMES)])
    return selected


class WasteImages(Dataset):
    def __init__(self, data_dir, records, augment=False):
        self.data_dir, self.records, self.augment = data_dir, records, augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        sample = self.records[index]
        return image_tensor(self.data_dir / sample["path"], self.augment), sample["label"], sample["path"]


def seed_worker(_worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def make_loader(args, records, seed, training=False):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        WasteImages(args.data_dir, records, augment=training),
        batch_size=args.batch_size, shuffle=training, num_workers=args.workers,
        generator=generator, worker_init_fn=seed_worker,
        pin_memory=args.device.type == "cuda",
    )


def evaluate(network, loader, device, include_predictions=False):
    network.eval()
    total_loss, correct, total = 0.0, 0, 0
    confusion = torch.zeros((4, 4), dtype=torch.int64)
    predictions = []
    with torch.inference_mode():
        for images, labels, paths in loader:
            images, labels = images.to(device), labels.to(device)
            logits = network(images)
            loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite evaluation loss. Try a lower learning rate.")
            predicted = logits.argmax(dim=1)
            total_loss += float(loss) * len(labels)
            correct += int((predicted == labels).sum())
            total += len(labels)
            truths, guesses = labels.cpu().tolist(), predicted.cpu().tolist()
            for truth, guess in zip(truths, guesses):
                confusion[truth, guess] += 1
            if include_predictions:
                confidence = logits.softmax(dim=1).max(dim=1).values.cpu().tolist()
                predictions.extend({
                    "image": path, "actual": CLASS_NAMES[truth],
                    "predicted": CLASS_NAMES[guess], "confidence": score,
                    "correct": truth == guess,
                } for path, truth, guess, score in zip(paths, truths, guesses, confidence))
    result = {
        "loss": total_loss / total, "accuracy": correct / total, "images": total,
        "confusion_matrix": confusion.tolist(),
        "per_class_accuracy": {
            name: float(confusion[i, i]) / int(confusion[i].sum())
            for i, name in enumerate(CLASS_NAMES)
        },
    }
    if include_predictions:
        result["predictions"] = predictions
    return result


def run_training(args, splits, size, seed, run_dir):
    selected = balanced_subset(splits, size)
    run_dir.mkdir(parents=True, exist_ok=False)
    save_json(run_dir / "train_images.json", selected)
    set_seed(seed)
    train_loader = make_loader(args, selected, seed, training=True)
    validation_loader = make_loader(args, splits["validation"], seed)
    test_loader = make_loader(args, splits["test"], seed)
    network = ORBITCNN().to(args.device)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, min_lr=1e-6)
        if args.lr_schedule == "plateau" else None
    )
    best_loss, best_epoch = float("inf"), 0
    history = []
    checkpoint_path = run_dir / "best.pt"
    started = time.perf_counter()
    print(f"\nTraining {size} images ({size // 4}/class), seed {seed}, device {args.device}", flush=True)
    for epoch in range(1, args.epochs + 1):
        network.train()
        training_loss, correct, total = 0.0, 0, 0
        for images, labels, _paths in train_loader:
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            logits = network(images)
            loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss. Try a lower learning rate.")
            loss.backward()
            optimizer.step()
            training_loss += float(loss.detach()) * len(labels)
            correct += int((logits.detach().argmax(dim=1) == labels).sum())
            total += len(labels)
        validation = evaluate(network, validation_loader, args.device)
        row = {
            "epoch": epoch, "train_loss": training_loss / total,
            "train_accuracy": correct / total, "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        write_csv(run_dir / "history.csv", history)
        if validation["loss"] < best_loss:
            best_loss, best_epoch = validation["loss"], epoch
            torch.save({
                "architecture": ARCHITECTURE, "class_names": list(CLASS_NAMES),
                "image_size": IMAGE_SIZE, "normalization": "RGB / 127.5 - 1",
                "model_state_dict": {k: v.detach().cpu().clone() for k, v in network.state_dict().items()},
                "train_size": size, "seed": seed, "epoch": epoch,
                "dataset_fingerprint": splits["dataset_fingerprint"],
                "split_fingerprint": fingerprint(splits),
                "validation_loss": best_loss, "validation_accuracy": validation["accuracy"],
            }, checkpoint_path)
        if scheduler is not None:
            scheduler.step(validation["loss"])
        print(
            f"Epoch {epoch:02d}/{args.epochs}: loss {row['train_loss']:.4f}, "
            f"train {row['train_accuracy']:.1%}, validation {row['validation_accuracy']:.1%}",
            flush=True,
        )
    # The held-out test set is evaluated only after validation has chosen the checkpoint.
    network, checkpoint = load_checkpoint(checkpoint_path, args.device)
    test = evaluate(network, test_loader, args.device, include_predictions=True)
    write_csv(run_dir / "test_predictions.csv", test.pop("predictions"))
    metrics = {
        "train_size": size, "seed": seed, "epochs": args.epochs,
        "best_epoch": best_epoch, "best_validation_loss": best_loss,
        "best_validation_accuracy": checkpoint["validation_accuracy"],
        "test": test, "classes": list(CLASS_NAMES),
        "dataset_fingerprint": splits["dataset_fingerprint"], "split_fingerprint": fingerprint(splits),
        "split_file": str(args.split_file), "device": str(args.device),
        "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "lr_schedule": args.lr_schedule,
        "weight_decay": args.weight_decay, "elapsed_seconds": time.perf_counter() - started,
        "torch_version": str(torch.__version__),
    }
    save_json(run_dir / "metrics.json", metrics)
    print(f"Best epoch {best_epoch}; held-out test accuracy: {test['accuracy']:.1%} ({test['images']} images)", flush=True)
    print(f"Saved: {run_dir}", flush=True)
    return metrics


def run_experiment(args, splits, experiment_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for size in args.sizes:
        balanced_subset(splits, size)  # Check all sizes before starting a long experiment.
    experiment_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for size in args.sizes:
        for repeat in range(args.repeats):
            seed = args.seed + repeat
            run_dir = experiment_dir / f"n{size}_seed{seed}"
            metrics = run_training(args, splits, size, seed, run_dir)
            rows.append({
                "train_size": size, "seed": seed,
                "test_accuracy": metrics["test"]["accuracy"], "test_loss": metrics["test"]["loss"],
                "best_epoch": metrics["best_epoch"],
                "validation_accuracy": metrics["best_validation_accuracy"],
            })
            write_csv(experiment_dir / "runs.csv", rows)
    summary = []
    for size in args.sizes:
        accuracies = [row["test_accuracy"] for row in rows if row["train_size"] == size]
        summary.append({
            "train_size": size, "repeats": len(accuracies),
            "mean_accuracy": statistics.mean(accuracies),
            "std_accuracy": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
        })
    write_csv(experiment_dir / "summary.csv", summary)
    save_json(experiment_dir / "summary.json", {"runs": rows, "summary": summary, "split_file": str(args.split_file)})
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.errorbar(
        [row["train_size"] for row in summary], [100 * row["mean_accuracy"] for row in summary],
        yerr=[100 * row["std_accuracy"] for row in summary], fmt="o-", capsize=5,
        label="Mean test accuracy +/- 1 sample standard deviation",
    )
    axis.axhline(25, color="gray", linestyle="--", label="Balanced random baseline (25%)")
    axis.set(xlabel="Total training images (balanced across four classes)", ylabel="Test accuracy (%)", ylim=(0, 100))
    axis.set_title("ORBIT-CNN: accuracy versus training dataset size")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(experiment_dir / "accuracy_vs_dataset_size.png", dpi=180)
    plt.close(figure)
    print(f"\nExperiment CSVs and graph saved: {experiment_dir}", flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "artifacts")
    parser.add_argument("--split-file", type=Path, help="Reuse the same saved split for every compared run")
    parser.add_argument("--train-size", type=int, default=1600, help="Total images, divided equally across four categories")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--lr-schedule", choices=("constant", "plateau"), default="constant",
                        help="Optionally reduce the learning rate when validation loss stalls")
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42, help="Training initialization, augmentation, and shuffle seed")
    parser.add_argument("--split-seed", type=int, default=1729, help="Fixed dataset partition and nested subset seed")
    parser.add_argument("--validation-per-class", type=int, default=25)
    parser.add_argument("--test-per-class", type=int, default=50)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--workers", type=int, default=0, help="0 is simplest on Windows; larger values use subprocesses")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--experiment", action="store_true")
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 200, 400, 800, 1600])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    for name in ("epochs", "batch_size", "threads", "validation_per_class", "test_per_class", "repeats"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.workers < 0 or not 0 <= args.seed < 2 ** 32 - args.repeats or not 0 <= args.split_seed < 2 ** 32:
        parser.error("Workers must be nonnegative and seeds must fit in unsigned 32-bit integers")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0 or not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("Learning rate must be finite and positive; weight decay must be finite and nonnegative")
    args.data_dir, args.output_dir = args.data_dir.resolve(), args.output_dir.resolve()
    args.split_file = (args.split_file or args.output_dir / "splits.json").resolve()
    args.sizes = sorted(set(args.sizes))
    try:
        args.device = select_device(args.device)
        torch.set_num_threads(args.threads)
        splits = prepare_splits(args.data_dir, args.split_file, args.split_seed, args.validation_per_class, args.test_per_class)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        if args.experiment:
            return run_experiment(args, splits, args.output_dir / "experiments" / stamp)
        run_dir = args.output_dir / "runs" / f"n{args.train_size}_seed{args.seed}_{stamp}"
        metrics = run_training(args, splits, args.train_size, args.seed, run_dir)
        shutil.copy2(run_dir / "best.pt", args.output_dir / "orbit_cnn.pt")
        save_json(args.output_dir / "latest_run.json", {"run_dir": str(run_dir), "metrics": metrics})
        print(f"Default prediction checkpoint updated: {args.output_dir / 'orbit_cnn.pt'}", flush=True)
        return metrics
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, f"Training failed: {error}\n")


if __name__ == "__main__":
    main()
