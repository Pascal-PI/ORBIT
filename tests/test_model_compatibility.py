"""Compatibility checks for legacy CNN and pretrained-model checkpoints."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import (ARCHITECTURE, CLASS_NAMES, TRANSFER_ARCHITECTURE,
                   image_tensor, load_checkpoint, make_network)
from predict import predict


class ModelCompatibilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_architecture_specific_preprocessing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'white.png'
            Image.new('RGB', (300, 250), 'white').save(path)
            legacy = image_tensor(path)
            transfer = image_tensor(path, architecture=TRANSFER_ARCHITECTURE)
            self.assertEqual(tuple(legacy.shape), (3,128,128))
            self.assertEqual(tuple(transfer.shape), (3,224,224))
            self.assertTrue(torch.all(legacy == 1))
            expected = (torch.ones(3) - torch.tensor([.485,.456,.406])) / torch.tensor([.229,.224,.225])
            torch.testing.assert_close(transfer[:,100,100], expected)

    def test_transfer_checkpoint_predicts_without_pretrained_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / 'sample.png'
            Image.fromarray(np.random.default_rng(7).integers(0,256,(240,280,3),dtype=np.uint8)).save(image)
            network = make_network(TRANSFER_ARCHITECTURE, pretrained=False).eval()
            checkpoint = {'architecture': TRANSFER_ARCHITECTURE, 'class_names':list(CLASS_NAMES),
                          'image_size':224, 'model_state_dict':network.state_dict()}
            path = root / 'model.pt'
            torch.save(checkpoint, path)
            with patch('predict.image_tensor', wraps=image_tensor) as preprocessing:
                result = predict(image, path, 'cpu')
            preprocessing.assert_called_once_with(image, architecture=TRANSFER_ARCHITECTURE)
            with torch.inference_mode():
                expected = network(image_tensor(image, architecture=TRANSFER_ARCHITECTURE).unsqueeze(0)).softmax(1)[0]
            torch.testing.assert_close(torch.tensor(list(result['probabilities'].values())), expected)
            self.assertAlmostEqual(sum(result['probabilities'].values()), 1., places=6)
            checkpoint['image_size'] = 128
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, 'input size'):
                load_checkpoint(path)

    def test_worse_candidate_does_not_replace_default(self):
        from train import promote_default
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            default, candidate = root / 'default.pt', root / 'candidate.pt'
            network = make_network(ARCHITECTURE)
            checkpoint = {'architecture': ARCHITECTURE, 'class_names': list(CLASS_NAMES),
                          'image_size': 128, 'model_state_dict': network.state_dict(),
                          'split_fingerprint': 'fixed-split', 'validation_loss': 0.2}
            torch.save(checkpoint, default)
            candidate.write_bytes(b'candidate snapshot')
            original = default.read_bytes()
            metrics = {'split_fingerprint': 'fixed-split', 'best_validation_loss': 0.3}
            self.assertFalse(promote_default(candidate, default, metrics))
            self.assertEqual(default.read_bytes(), original)
            metrics['best_validation_loss'] = 0.1
            metrics['split_fingerprint'] = 'different-split'
            self.assertFalse(promote_default(candidate, default, metrics))
            self.assertEqual(default.read_bytes(), original)
            metrics['split_fingerprint'] = 'fixed-split'
            self.assertTrue(promote_default(candidate, default, metrics))
            self.assertEqual(default.read_bytes(), candidate.read_bytes())

    def test_legacy_checkpoint_still_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'legacy.pt'
            network = make_network(ARCHITECTURE).eval()
            torch.save({'architecture':ARCHITECTURE, 'class_names':list(CLASS_NAMES),
                        'image_size':128, 'model_state_dict':network.state_dict()}, path)
            loaded, _ = load_checkpoint(path)
            with torch.inference_mode():
                inputs = torch.randn(1,3,128,128)
                torch.testing.assert_close(network(inputs), loaded(inputs))


if __name__ == '__main__':
    unittest.main()
