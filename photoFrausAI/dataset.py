import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image

class AIVsRealDataset(Dataset):
    """
    Ultra-high-throughput Dataset wrapper.
    Eliminates all CPU math and augmentations by caching pre-resized uint8 tensors in RAM.
    Augmentations and normalizations are offloaded 100% to the GPU in train.py.
    """
    def __init__(self, hf_split):
        self.dataset = hf_split
        self.cache = {}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if idx in self.cache:
            img_tensor, label = self.cache[idx]
        else:
            item = self.dataset[idx]
            image = item["image"]
            label = int(item["binary_label"])

            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Fast bilinear resize once to 224x224
            image = image.resize((224, 224), Image.BILINEAR)
            # Store as uint8 tensor (3, 224, 224) in RAM (only 147 KB)
            img_tensor = TF.pil_to_tensor(image)
            self.cache[idx] = (img_tensor, label)

        # Convert uint8 to float (0.0 to 1.0)
        img_float = img_tensor.float().div(255.0)
        return img_float, torch.tensor(label, dtype=torch.long)


def get_transforms():
    """
    Standard ImageNet normalization and resize for inference and evaluation.
    """
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return val_transform, val_transform
