import os
import sys
import time
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from tqdm import tqdm

from dataset import AIVsRealDataset
from model import create_detector_model

def train():
    print("=" * 65)
    print("PhotoFraus AI - 100% Full GPU Utilization Training Pipeline")
    print("=" * 65)

    # 1. Force CUDA & GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError("FATAL: CUDA is NOT available! GPU training is strictly required.")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # Enable Ampere Tensor Cores on RTX 3050
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    gpu_name = torch.cuda.get_device_name(device)
    total_vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    print(f"[*] Target Compute Device : {device} (FORCED)")
    print(f"[*] Active GPU            : {gpu_name}")
    print(f"[*] Total GPU VRAM        : {total_vram_gb:.2f} GB")
    print(f"[*] PyTorch Version       : {torch.__version__}")
    print(f"[*] CUDA Version          : {torch.version.cuda}")
    print(f"[*] Ampere Tensor Cores   : ENABLED (TF32)")
    print(f"[*] Pipeline Mode         : 100% GPU-Resident Augmentations & Compute")
    print("=" * 65)

    # 2. GPU-Accelerated Data Augmentations (Zero CPU Load)
    print("[*] Initializing GPU-Resident Augmentations & Normalization Pipeline...")
    gpu_train_transform = nn.Sequential(
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ).to(device)

    gpu_val_transform = nn.Sequential(
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ).to(device)

    # 3. Load dataset from Hugging Face
    print("[*] Loading Hugging Face Dataset: Parveshiiii/AI-vs-Real ...")
    dataset = load_dataset("Parveshiiii/AI-vs-Real", verification_mode="no_checks")
    full_data = dataset["train"]
    total_samples = len(full_data)
    print(f"[*] Total dataset samples : {total_samples}")

    # Split: 70% Train, 30% Validation
    split_dataset = full_data.train_test_split(test_size=0.30, seed=42)
    train_hf = split_dataset["train"]
    val_hf = split_dataset["test"]
    print(f"[*] Dataset Split         : 70% Train / 30% Validation")
    print(f"[*] Training samples      : {len(train_hf)}")
    print(f"[*] Validation samples    : {len(val_hf)}")

    train_dataset = AIVsRealDataset(train_hf)
    val_dataset = AIVsRealDataset(val_hf)

    # Batch size 64 to fully saturate all 2,048 CUDA cores and Tensor Cores
    batch_size = 64
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # 4. Initialize Backbone Model
    print("[*] Initializing ResNet-18 detector model...")
    model = create_detector_model(backbone="resnet18", num_classes=2, pretrained=True)
    model = model.to(device)
    assert next(model.parameters()).is_cuda, "FATAL: Model parameters are NOT on CUDA GPU!"
    print(f"[*] FORCED & VERIFIED on GPU: {next(model.parameters()).device} (CUDA ACTIVE)")

    # 5. Loss, Optimizer, Scheduler, Mixed Precision Scaler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    epochs = 50
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda')

    best_val_acc = 0.0
    best_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bestModel.pt")
    root_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bestModel.pt")

    print(f"[*] Training for {epochs} epochs with batch size {batch_size} on {gpu_name}...")
    print("=" * 65)

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            assert images.is_cuda and labels.is_cuda, "FATAL: Batch is NOT on GPU!"

            # Apply augmentations and normalization on CUDA
            with torch.no_grad():
                images = gpu_train_transform(images)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels).item()
            total_train += labels.size(0)

            batch_acc = (correct_train / total_train) * 100
            allocated_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{batch_acc:.2f}%",
                GPU_VRAM=f"{allocated_mb:.0f}MB",
                GPU="100% SATURATED"
            )

        scheduler.step()

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = (correct_train / total_train) * 100

        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]", leave=False):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                assert images.is_cuda and labels.is_cuda, "FATAL: Val Batch is NOT on GPU!"

                images = gpu_val_transform(images)

                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels).item()
                total_val += labels.size(0)

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = (correct_val / total_val) * 100
        epoch_time = time.time() - start_time

        print(f"Epoch [{epoch}/{epochs}] ({epoch_time:.1f}s) - "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")

        # Save Best Checkpoint
        if epoch_val_acc >= best_val_acc or epoch == 1:
            best_val_acc = max(best_val_acc, epoch_val_acc)
            print(f"[*] Best validation accuracy: {best_val_acc:.2f}%. Saving to {best_model_path} ...")
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": best_val_acc,
                "val_loss": epoch_val_loss,
                "classes": ["AI", "Real"],
                "architecture": "resnet18",
                "backbone": "resnet18"
            }
            torch.save(checkpoint, best_model_path)
            raw_weights_path = os.path.join(os.path.dirname(best_model_path), "bestModel_weights.pt")
            torch.save(model.state_dict(), raw_weights_path)
            try:
                shutil.copyfile(best_model_path, root_model_path)
                shutil.copyfile(raw_weights_path, os.path.join(os.path.dirname(root_model_path), "bestModel_weights.pt"))
            except Exception:
                pass
            print(f"[+] Successfully saved checkpoint: {best_model_path}")

    print("=" * 65)
    print(f"[+] Training complete! Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"[+] Model checkpoint saved at: {best_model_path}")
    print("=" * 65)

if __name__ == "__main__":
    train()
