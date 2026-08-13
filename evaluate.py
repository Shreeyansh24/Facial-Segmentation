"""
Facial Segmentation — Model Evaluation
=======================================
Evaluates a trained model checkpoint using proper imbalanced segmentation metrics:
  - Per-class Dice scores
  - Macro-Large Dice (Classes 0-2)
  - Macro-Small Dice (Classes 3-8)
  - Frequency-Weighted Dice

Uses multi-scale Test-Time Augmentation (TTA) for maximum accuracy.

Usage:
    python evaluate.py
    python evaluate.py --checkpoint path/to/model.pth
"""

import argparse
import torch
import cv2
import os
import numpy as np
from glob import glob
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
IMG_SIZE    = 1024
NUM_CLASSES = 9
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Approximate pixel frequencies from the training set (for FW-Dice)
CLASS_FREQS = np.array([0.4937, 0.3965, 0.1012, 0.0018, 0.0017, 0.0008, 0.0008, 0.0032, 0.0002])

BASE     = "data"
VAL_IMG  = os.path.join(BASE, "images", "val")
VAL_MASK = os.path.join(BASE, "annotations", "val")

# ─────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

val_tfm = A.Compose([
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

# ─────────────────────────────────────────────────────────────
# Inference Helpers
# ─────────────────────────────────────────────────────────────
def preprocess_img(img_bgr, target_size):
    """Resize, normalize, and convert to tensor."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (target_size, target_size))
    aug = val_tfm(image=img_resized)['image']
    return aug.unsqueeze(0).to(DEVICE)

@torch.no_grad()
def predict_tta(model, img_bgr, scales=[0.75, 1.0, 1.25]):
    """Multi-scale Test-Time Augmentation with horizontal flip."""
    orig_h, orig_w = img_bgr.shape[:2]
    all_preds = []

    for scale in scales:
        target_size = int(IMG_SIZE * scale)
        target_size = (target_size // 32) * 32  # Must be divisible by 32 for UNet++

        inp1 = preprocess_img(img_bgr, target_size)
        inp2 = preprocess_img(img_bgr[:, ::-1, :].copy(), target_size)

        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        with torch.autocast(device_type=device_type, enabled=torch.cuda.is_available()):
            p1 = model(inp1).softmax(dim=1)
            p2 = model(inp2).softmax(dim=1).flip(dims=[3])

        avg = ((p1 + p2) / 2.0)[0].permute(1, 2, 0).cpu().numpy()
        avg_full = cv2.resize(avg, (orig_w, orig_h))
        all_preds.append(avg_full)

    final_avg = np.mean(all_preds, axis=0)
    return np.argmax(final_avg, axis=-1).astype(np.uint8)

def compute_per_class_dice(y_true, y_pred, num_classes=NUM_CLASSES, smooth=1e-6):
    """Compute Dice score for each class independently."""
    dices = []
    for c in range(num_classes):
        gt   = (y_true == c).astype(np.float32)
        pred = (y_pred == c).astype(np.float32)
        inter = (gt * pred).sum()
        denom = gt.sum() + pred.sum()
        dices.append((2 * inter + smooth) / (denom + smooth))
    return np.array(dices)

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate facial segmentation model")
    parser.add_argument("--checkpoint", type=str, default="best_model.pth",
                        help="Path to model checkpoint (default: best_model.pth)")
    args = parser.parse_args()

    # Load validation pairs
    imgs  = sorted(glob(os.path.join(VAL_IMG,  "*.jpg")) + glob(os.path.join(VAL_IMG,  "*.png")))
    masks = sorted(glob(os.path.join(VAL_MASK, "*.jpg")) + glob(os.path.join(VAL_MASK, "*.png")))

    if len(imgs) != len(masks):
        img_dict  = {os.path.splitext(os.path.basename(p))[0]: p for p in imgs}
        mask_dict = {os.path.splitext(os.path.basename(p))[0]: p for p in masks}
        common = sorted(list(set(img_dict.keys()) & set(mask_dict.keys())))
        val_pairs = [(img_dict[k], mask_dict[k]) for k in common]
    else:
        val_pairs = list(zip(imgs, masks))

    # Build and load model
    print(f"Loading model from {args.checkpoint} for evaluation on {len(val_pairs)} images...")
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b3",
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,
        decoder_attention_type="scse",
    )

    if not os.path.exists(args.checkpoint):
        print(f"Error: {args.checkpoint} not found.")
        print("Download weights from the GitHub Releases page, or train the model first.")
        exit(1)

    checkpoint = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'] if isinstance(checkpoint, dict) else checkpoint)
    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully.")

    # Run evaluation
    all_dices = []
    for img_p, mask_p in tqdm(val_pairs, desc="Evaluating (Multi-Scale TTA)"):
        img_bgr   = cv2.imread(img_p)
        true_mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        true_mask[true_mask >= NUM_CLASSES] = 0

        pred  = predict_tta(model, img_bgr)
        dices = compute_per_class_dice(true_mask, pred)
        all_dices.append(dices)

    all_dices = np.array(all_dices)
    mean_per_class = all_dices.mean(axis=0)

    # Compute aggregate metrics
    macro_dice  = mean_per_class.mean()
    macro_large = mean_per_class[[0, 1, 2]].mean()
    macro_small = mean_per_class[3:].mean()
    fw_dice     = np.sum(mean_per_class * CLASS_FREQS)

    # Print report
    print("\n" + "="*60)
    print("EVALUATION REPORT")
    print("="*60)
    print(f"Macro-Average Dice:        {macro_dice:.5f}")
    print(f"Frequency-Weighted Dice:   {fw_dice:.5f}")
    print(f"Macro-Large (Classes 0-2): {macro_large:.5f}")
    print(f"Macro-Small (Classes 3-8): {macro_small:.5f}")
    print("-" * 60)
    print("Per-Class Breakdown:")
    for i, score in enumerate(mean_per_class):
        label = "Large" if i <= 2 else "Small"
        print(f"  Class {i}: {score:.4f}  ({label})")
    print("=" * 60)
