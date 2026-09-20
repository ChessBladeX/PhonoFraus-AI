import os
import sys
import torch
from PIL import Image
from torchvision import transforms
from model import create_detector_model

def get_inference_transform():
    """Standard 224x224 ImageNet transform for inference."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def test_inference(image_path=None, model_path="bestModel.pt"):
    # Check default paths if model_path does not exist
    if not os.path.exists(model_path):
        candidates = [
            os.path.join("checkpoint", "bestModel.pt"),
            os.path.join(os.path.dirname(__file__), "checkpoint", "bestModel.pt"),
            os.path.join(os.path.dirname(__file__), "bestModel.pt"),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "bestModel.pt")
        ]
        for c in candidates:
            if os.path.exists(c):
                model_path = c
                break

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(f"PhotoFraus AI - Inference Verification")
    print(f"[*] Compute Device : {device}")
    print(f"[*] Model File     : {model_path}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    val_acc = checkpoint.get('val_acc', 0)
    epoch = checkpoint.get('epoch', '?')
    arch = checkpoint.get("architecture", checkpoint.get("backbone", "resnet18"))
    classes = checkpoint.get("classes", ["AI", "Real"])
    print(f"[*] Architecture   : {arch}")
    print(f"[*] Checkpoint     : Epoch {epoch} (Validation Accuracy: {val_acc:.2f}%)")
    print("=" * 60)

    model = create_detector_model(backbone=arch, num_classes=len(classes), pretrained=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    transform = get_inference_transform()

    # Load provided image or create a sample synthetic image
    if image_path and os.path.exists(image_path):
        print(f"[*] Inspecting target image: {image_path}")
        img = Image.open(image_path).convert("RGB")
    else:
        print("[*] No custom image provided. Using synthetic test image...")
        img = Image.new("RGB", (224, 224), color=(140, 160, 180))

    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = torch.argmax(probs).item()

    pred_label = classes[pred_idx]
    confidence = probs[pred_idx].item() * 100

    print(f"\n[RESULT]")
    print(f">> Prediction   : {pred_label}")
    print(f">> Confidence   : {confidence:.2f}%")
    print(f">> Breakdown    : {classes[0]}: {probs[0].item()*100:.2f}% | {classes[1]}: {probs[1].item()*100:.2f}%")
    print("=" * 60)
    print("Inference test passed successfully!")

if __name__ == "__main__":
    img_arg = sys.argv[1] if len(sys.argv) > 1 else None
    test_inference(image_path=img_arg)
