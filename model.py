"""ORBIT CNN/MobileNet models and architecture-specific image preprocessing."""

from pathlib import Path
import random
from functools import lru_cache

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import torch
from torch import nn


PROJECT_DIR = Path(__file__).resolve().parent
CLASS_NAMES = ("Plastic", "Metal", "Paper", "Trash")
CLASS_FOLDERS = ("plastic", "metal", "paper", "general trash")
IMAGE_SIZE = 128
ARCHITECTURE = "ORBIT-CNN-v0.1"
TRANSFER_ARCHITECTURE = "ORBIT-MobileNetV3-Large-v1"
DEFAULT_CHECKPOINT = PROJECT_DIR / "artifacts" / "orbit_cnn.pt"


class ORBITCNN(nn.Module):
    """Three convolution/pooling blocks followed by a four-class classifier.

    Input: N x 3 x 128 x 128, normalized to [-1, 1].
    Output: N x 4 raw logits. CrossEntropyLoss applies its own log-softmax.
    """

    def __init__(self):
        super().__init__()
        blocks = []
        input_channels = 3
        for output_channels in (16, 32, 64):
            blocks.extend([
                nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ])
            input_channels = output_channels
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, len(CLASS_NAMES)),
        )

    def forward(self, images):
        return self.classifier(self.features(images))


def make_network(architecture=ARCHITECTURE, pretrained=False):
    if architecture == ARCHITECTURE:
        return ORBITCNN()
    if architecture == TRANSFER_ARCHITECTURE:
        from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
        network = mobilenet_v3_large(
            weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        )
        network.classifier[-1] = nn.Linear(network.classifier[-1].in_features, len(CLASS_NAMES))
        return network
    raise ValueError("The checkpoint uses an unsupported model architecture.")


@lru_cache(maxsize=2)
def transfer_transform(augment=False):
    from torchvision import transforms
    from torchvision.models import MobileNet_V3_Large_Weights
    if not augment:
        return MobileNet_V3_Large_Weights.IMAGENET1K_V2.transforms()
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def image_tensor(path, augment=False, architecture=ARCHITECTURE):
    """Load RGB pixels consistently, with random augmentation only for training."""
    try:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            if architecture == TRANSFER_ARCHITECTURE:
                return transfer_transform(augment)(image)
            if architecture != ARCHITECTURE:
                raise ValueError("Unsupported preprocessing architecture.")
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
            if augment:
                if random.random() < 0.5:
                    image = ImageOps.mirror(image)
                image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
                image = ImageEnhance.Contrast(image).enhance(random.uniform(0.9, 1.1))
            pixels = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read image {path}: {error}") from error
    return torch.from_numpy(pixels.transpose(2, 0, 1).copy())


def select_device(requested="auto"):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Use --device cpu or --device auto.")
    if requested not in ("cpu", "cuda"):
        raise ValueError("Device must be auto, cpu, or cuda.")
    return torch.device(requested)


def load_checkpoint(path=DEFAULT_CHECKPOINT, device="cpu"):
    """Load tensor weights and plain metadata, without unpickling a model object."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No trained model at {path}. Run train.py first.")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("This is not an ORBIT checkpoint.")
    architecture = checkpoint.get("architecture")
    if architecture not in (ARCHITECTURE, TRANSFER_ARCHITECTURE):
        raise ValueError("The checkpoint uses an unsupported model architecture.")
    if checkpoint.get("class_names") != list(CLASS_NAMES):
        raise ValueError("The checkpoint's class order does not match ORBIT.")
    expected_size = IMAGE_SIZE if architecture == ARCHITECTURE else 224
    if checkpoint.get("image_size") != expected_size:
        raise ValueError("The checkpoint's input size does not match ORBIT.")
    network = make_network(architecture)
    network.load_state_dict(checkpoint["model_state_dict"], strict=True)
    network.to(device).eval()
    return network, checkpoint


if __name__ == "__main__":
    network = ORBITCNN().eval()
    with torch.inference_mode():
        outputs = network(torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE))
    print(network)
    print(f"Parameters: {sum(p.numel() for p in network.parameters()):,}")
    print(f"Output shape: {tuple(outputs.shape)}; classes: {', '.join(CLASS_NAMES)}")
