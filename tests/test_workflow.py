"""End-to-end checks using temporary images; never modify production artifacts."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import CLASS_FOLDERS, CLASS_NAMES, load_checkpoint
from predict import predict
from train import main


class TrainingWorkflowTest(unittest.TestCase):
    def test_scheduled_training_checkpoint_and_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(10)
            for folder in CLASS_FOLDERS:
                category = root / folder
                category.mkdir()
                for index in range(4):
                    Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(category / f'{index}.png')
            output = root / 'output'
            metrics = main(['--data-dir', str(root), '--output-dir', str(output),
                            '--train-size', '8', '--epochs', '2', '--threads', '1',
                            '--validation-per-class', '1', '--test-per-class', '1',
                            '--lr-schedule', 'plateau'])
            self.assertEqual(metrics['lr_schedule'], 'plateau')
            self.assertEqual(metrics['test']['images'], 4)
            latest = json.loads((output / 'latest_run.json').read_text())
            run = Path(latest['run_dir'])
            with (run / 'history.csv').open() as handle:
                history = list(csv.DictReader(handle))
            self.assertEqual(len(history), 2)
            self.assertGreater(float(history[0]['learning_rate']), 0)
            splits = json.loads((output / 'splits.json').read_text())
            selected = json.loads((run / 'train_images.json').read_text())
            self.assertTrue({r['sha256'] for r in selected}.isdisjoint(
                {r['sha256'] for r in splits['test'] + splits['validation']}))
            network, checkpoint = load_checkpoint(output / 'orbit_cnn.pt')
            self.assertEqual(checkpoint['train_size'], 8)
            result = predict(root / splits['test'][0]['path'], output / 'orbit_cnn.pt', 'cpu')
            self.assertIn(result['category'], CLASS_NAMES)
            self.assertAlmostEqual(sum(result['probabilities'].values()), 1, places=6)
            with torch.inference_mode():
                self.assertEqual(tuple(network(torch.zeros(1, 3, 128, 128)).shape), (1, 4))


if __name__ == '__main__':
    unittest.main()
