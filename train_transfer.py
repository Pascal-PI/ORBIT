"""Train a pretrained deployment candidate without changing the custom-CNN experiment."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import shutil
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from model import (CLASS_NAMES, DEFAULT_CHECKPOINT, PROJECT_DIR, TRANSFER_ARCHITECTURE,
                   image_tensor, load_checkpoint, make_network, select_device)
from train import balanced_subset, evaluate, fingerprint, prepare_splits, save_json, set_seed, write_csv


class TransferImages(Dataset):
    def __init__(self, root, records, augment=False):
        self.root, self.records, self.augment = root, records, augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        return (image_tensor(self.root / record['path'], self.augment, TRANSFER_ARCHITECTURE),
                record['label'], record['path'])


def extract_embeddings(network, loader, device, mirror=False):
    network.eval()
    features, targets = [], []
    with torch.inference_mode():
        for index, (images, labels, _) in enumerate(loader, 1):
            images = images.to(device)
            views = (images, images.flip(-1)) if mirror else (images,)
            for view in views:
                values = network.features(view)
                values = network.avgpool(values).flatten(1)
                values = network.classifier[:-1](values)
                features.append(values.cpu())
                targets.append(labels)
            if index % 20 == 0 or index == len(loader):
                print(f'Embedding batches {index}/{len(loader)}', flush=True)
    return torch.cat(features), torch.cat(targets)


def fit(args):
    set_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = select_device(args.device)
    splits = prepare_splits(args.data_dir, args.split_file)
    selected = splits['train_pool'] if args.train_size == 0 else balanced_subset(splits, args.train_size)
    if not selected:
        raise ValueError('No training images.')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run_dir = args.output_dir / f'transfer_{stamp}'
    run_dir.mkdir(parents=True, exist_ok=False)
    save_json(run_dir / 'train_images.json', selected)
    save_json(run_dir / 'config.json', {**{k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                                      'pretrained_weights': 'MobileNet_V3_Large_Weights.IMAGENET1K_V2'})
    counts = Counter(r['label'] for r in selected)
    weights = torch.tensor([len(selected) / (4 * counts[i]) for i in range(4)], device=device)
    network = make_network(TRANSFER_ARCHITECTURE, pretrained=True).to(device)
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    for parameter in network.classifier[-1].parameters():
        parameter.requires_grad_(True)
    def loader(records, augment=False):
        return DataLoader(TransferImages(args.data_dir, records, augment), batch_size=args.batch_size,
                          shuffle=augment, num_workers=0,
                          generator=torch.Generator().manual_seed(args.seed))
    started = time.perf_counter()
    print(f'Training pool: {len(selected)} images; counts {dict(counts)}; device {device}', flush=True)
    print('Extracting frozen training features (original + mirrored views)...', flush=True)
    train_x, train_y = extract_embeddings(network, loader(selected), device, mirror=True)
    print('Extracting validation features...', flush=True)
    val_x, val_y = extract_embeddings(network, loader(splits['validation']), device)
    train_x, train_y, val_x, val_y = [v.to(device) for v in (train_x, train_y, val_x, val_y)]
    features_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=128, shuffle=True,
                                generator=torch.Generator().manual_seed(args.seed))
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(network.classifier[-1].parameters(), lr=0.003, weight_decay=0.001)
    best_loss, best_accuracy, best_state, best_stage, best_epoch = float('inf'), 0, None, None, 0
    history = []

    def consider(loss, accuracy, stage, epoch, train_loss):
        nonlocal best_loss, best_accuracy, best_state, best_stage, best_epoch
        if not torch.isfinite(torch.tensor(loss)):
            raise ValueError('Non-finite validation loss.')
        history.append({'stage': stage, 'epoch': epoch, 'train_loss': train_loss,
                        'validation_loss': loss, 'validation_accuracy': accuracy,
                        'learning_rate': optimizer.param_groups[0]['lr']})
        write_csv(run_dir / 'history.csv', history)
        if loss < best_loss:
            best_loss, best_accuracy, best_stage, best_epoch = loss, accuracy, stage, epoch
            best_state = {k: v.detach().cpu().clone() for k,v in network.state_dict().items()}
        print(f'{stage} {epoch:02d}: train loss {train_loss:.4f}, validation loss {loss:.4f}, '
              f'accuracy {accuracy:.1%}; best {best_accuracy:.1%}', flush=True)

    network.eval()
    for epoch in range(1, args.head_epochs + 1):
        total_loss, total = 0., 0
        for values, labels in features_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(network.classifier[-1](values), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(labels)
            total += len(labels)
        with torch.inference_mode():
            logits = network.classifier[-1](val_x)
            loss = float(nn.functional.cross_entropy(logits, val_y))
            accuracy = float((logits.argmax(1) == val_y).float().mean())
        consider(loss, accuracy, 'head', epoch, total_loss / total)
    network.load_state_dict(best_state)
    del train_x, train_y, val_x, val_y, features_loader
    for block in network.features[-3:]:
        for parameter in block.parameters():
            parameter.requires_grad_(True)
    for parameter in network.classifier.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW([
        {'params': [p for p in network.features.parameters() if p.requires_grad], 'lr': 0.0001},
        {'params': network.classifier.parameters(), 'lr': 0.0003},
    ], weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=0.5)
    training_loader, validation_loader = loader(selected, True), loader(splits['validation'])
    for epoch in range(1, args.fine_epochs + 1):
        network.train()
        # Preserve pretrained running statistics: small task batches should not rewrite them.
        for module in network.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        total_loss, total = 0., 0
        for images, labels, _ in training_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(network(images), labels)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss.')
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(labels)
            total += len(labels)
        result = evaluate(network, validation_loader, device)
        consider(result['loss'], result['accuracy'], 'fine', epoch, total_loss / total)
        scheduler.step(result['loss'])
    checkpoint = {
        'architecture': TRANSFER_ARCHITECTURE, 'class_names': list(CLASS_NAMES), 'image_size': 224,
        'normalization': 'ImageNet mean/std; resize 232 then center crop 224',
        'pretrained_weights': 'MobileNet_V3_Large_Weights.IMAGENET1K_V2',
        'model_state_dict': best_state, 'train_size': len(selected), 'seed': args.seed,
        'stage': best_stage, 'epoch': best_epoch, 'validation_loss': best_loss,
        'validation_accuracy': best_accuracy, 'dataset_fingerprint': splits['dataset_fingerprint'],
        'split_fingerprint': fingerprint(splits),
    }
    torch.save(checkpoint, run_dir / 'best.pt')
    network, _ = load_checkpoint(run_dir / 'best.pt', device)
    # The test set is touched only after all head/fine-tuning choices are complete.
    test = evaluate(network, loader(splits['test']), device, include_predictions=True)
    write_csv(run_dir / 'test_predictions.csv', test.pop('predictions'))
    baseline = None
    if DEFAULT_CHECKPOINT.exists():
        _, baseline_checkpoint = load_checkpoint(DEFAULT_CHECKPOINT)
        if baseline_checkpoint.get('split_fingerprint') == checkpoint['split_fingerprint']:
            baseline = {'validation_loss': baseline_checkpoint['validation_loss'],
                        'validation_accuracy': baseline_checkpoint['validation_accuracy'],
                        'architecture': baseline_checkpoint['architecture']}
    promote = bool(args.promote and (not DEFAULT_CHECKPOINT.exists() or
                   (baseline is not None and best_loss < baseline['validation_loss'])))
    metrics = {k:v for k,v in checkpoint.items() if k != 'model_state_dict'}
    metrics.update({'test': test, 'head_epochs': args.head_epochs, 'fine_epochs': args.fine_epochs,
                    'class_counts': dict(counts), 'class_weights': weights.cpu().tolist(),
                    'elapsed_seconds': time.perf_counter() - started, 'device': str(device),
                    'torch_version': str(torch.__version__), 'baseline': baseline,
                    'promoted': promote, 'promotion_rule': 'lower validation loss on the same fixed split'})
    save_json(run_dir / 'metrics.json', metrics)
    if promote:
        # The old checkpoint is also versioned in its original run folder and Git history.
        shutil.copy2(run_dir / 'best.pt', DEFAULT_CHECKPOINT)
        save_json(PROJECT_DIR / 'artifacts/latest_run.json', {'run_dir': str(run_dir), 'metrics': metrics})
    print(f'Final test accuracy: {test["accuracy"]:.1%}; per class {test["per_class_accuracy"]}', flush=True)
    print(f'Promoted by validation: {promote}; run: {run_dir}', flush=True)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=PROJECT_DIR)
    parser.add_argument('--split-file', type=Path, default=PROJECT_DIR / 'artifacts/splits.json')
    parser.add_argument('--output-dir', type=Path, default=PROJECT_DIR / 'artifacts/transfer')
    parser.add_argument('--train-size', type=int, default=0, help='0 uses the entire fixed training pool')
    parser.add_argument('--head-epochs', type=int, default=60)
    parser.add_argument('--fine-epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=('auto','cpu','cuda'), default='auto')
    parser.add_argument('--promote', action='store_true', help='Use this model if validation loss beats the default')
    args = parser.parse_args(argv)
    if min(args.head_epochs, args.batch_size, args.threads) < 1 or args.fine_epochs < 0:
        parser.error('Head epochs, batch size and threads must be positive; fine epochs nonnegative')
    if args.train_size < 0 or not 0 <= args.seed < 2**32:
        parser.error('Training size must be nonnegative and seed must fit unsigned 32 bits')
    args.data_dir, args.split_file, args.output_dir = [p.resolve() for p in (args.data_dir,args.split_file,args.output_dir)]
    try:
        return fit(args)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, f'Transfer training failed: {error}\n')


if __name__ == '__main__':
    main()
