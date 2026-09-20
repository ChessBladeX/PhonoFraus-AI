import os
import sys
import torch
import torch.nn as nn

# Ensure photoFrausAI and project root are on path for model loading
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
photofraus_dir = os.path.join(root_dir, "photoFrausAI")

for p in (root_dir, photofraus_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from photoFrausAI.model import create_detector_model
from .config import MODEL_PATH, DECISION_THRESHOLD

class InferenceEngine:
    def __init__(self, model_path: str = None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path or MODEL_PATH
        self.model = None
        self.classes = ["AI", "Real"]
        self.arch = "resnet18"
        self._load_model()

    def _load_model(self):
        if not self.model_path or not os.path.exists(self.model_path):
            print(f"[WARN] Checkpoint not found at {self.model_path}. Running with placeholder model.")
            self.model = create_detector_model(backbone="resnet18", num_classes=2, pretrained=False)
            self.model.to(self.device)
            self.model.eval()
            return

        print(f"[*] Loading model on {self.device} from {self.model_path}...")
        checkpoint = torch.load(self.model_path, map_location=self.device)

        self.val_acc = 0.0
        if isinstance(checkpoint, dict):
            # The dataset Parveshiiii/AI-vs-Real was trained with class 0 = AI, class 1 = Real
            self.classes = checkpoint.get("classes", ["AI", "Real"])
            self.arch = checkpoint.get("architecture", checkpoint.get("backbone", "resnet18"))
            self.val_acc = checkpoint.get("val_acc", 0.0)
            print("=" * 65)
            print("  PhotoFraus AI Engine: PyTorch Model Connected")
            print(f"  Checkpoint File : {self.model_path}")
            print(f"  Architecture    : {self.arch}")
            print(f"  Validation Acc  : {self.val_acc:.2f}%")
            print(f"  Inference Device: {self.device}")
            print("=" * 65)

            self.model = create_detector_model(backbone=self.arch, num_classes=len(self.classes), pretrained=False)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            elif "state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            self.model = create_detector_model(backbone="resnet18", num_classes=2, pretrained=False)
            self.model.load_state_dict(checkpoint)

        # Free checkpoint dictionary and optimizer states from memory
        del checkpoint
        import gc
        gc.collect()

        self.model.to(self.device)
        self.model.eval()

    def predict(self, tensor: torch.Tensor):
        tensor = tensor.to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]

        # Dataset training ground truth: 0 -> AI Generated, 1 -> Authentic Real
        ai_idx = 0
        real_idx = 1
        if len(self.classes) > 1:
            for idx, c in enumerate(self.classes):
                if c.lower() in ["ai", "fake", "synthetic"]:
                    ai_idx = idx
                elif c.lower() in ["real", "authentic"]:
                    real_idx = idx

        ai_score = float(probs[ai_idx].item())
        real_score = float(probs[real_idx].item())

        is_ai = ai_score >= DECISION_THRESHOLD

        return {
            "is_ai": is_ai,
            "confidence": round(ai_score, 4),
            "real_score": round(real_score, 4),
            "raw": [
                {"label": "AI Generated", "score": round(ai_score, 4)},
                {"label": "Authentic Real", "score": round(real_score, 4)}
            ]
        }

engine = InferenceEngine()
