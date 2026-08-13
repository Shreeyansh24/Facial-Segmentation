"""
Facial Segmentation — Visualization & Inference
=================================================
Generates side-by-side comparison images:
    Input Face | Ground Truth Mask | Model Prediction

Saves output images to the results/ directory.

Usage:
    python visualize.py                           # Visualize 5 random validation images
    python visualize.py --num 10                  # Visualize 10 random validation images
    python visualize.py --checkpoint model.pth    # Use a specific checkpoint
    python visualize.py --image path/to/face.jpg  # Run on a single image (no ground truth)
"""

import argparse
import os
import random
import torch
import cv2
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

# Color palette for each class (BGR format for OpenCV)
CLASS_COLORS = [
    (0,   0,   0),    # Class 0 — Black
    (180, 200, 230),  # Class 1 — Light Beige
    (60,  60,  180),  # Class 2 — Dark Red
    (0,   200, 255),  # Class 3 — Yellow
    (0,   165, 255),  # Class 4 — Orange
    (255, 200, 0),    # Class 5 — Cyan
    (255, 150, 0),    # Class 6 — Blue
    (0,   255, 0),    # Class 7 — Green
    (130, 0,   220),  # Class 8 — Purple
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

val_tfm = A.Compose([
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

# ─────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────
def preprocess_img(img_bgr, target_size):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (target_size, target_size))
    aug = val_tfm(image=img_resized)['image']
    return aug.unsqueeze(0).to(DEVICE)

@torch.no_grad()
def predict_tta(model, img_bgr, scales=[0.75, 1.0, 1.25]):
    """Multi-scale TTA with horizontal flip for best accuracy."""
    orig_h, orig_w = img_bgr.shape[:2]
    all_preds = []

    for scale in scales:
        target_size = int(IMG_SIZE * scale)
        target_size = (target_size // 32) * 32

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

# ─────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────
def mask_to_color(mask):
    """Convert a class-index mask to a color image."""
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, bgr in enumerate(CLASS_COLORS):
        color[mask == class_id] = bgr
    return color

def create_overlay(img_bgr, mask, alpha=0.45):
    """Blend the original image with a color-coded mask overlay."""
    color_mask = mask_to_color(mask)
    overlay = cv2.addWeighted(img_bgr, 1 - alpha, color_mask, alpha, 0)
    return overlay

def add_title(img, title, font_scale=0.7, thickness=2):
    """Add a centered title bar above the image."""
    h, w = img.shape[:2]
    bar_h = 35
    bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
    text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    x = (w - text_size[0]) // 2
    cv2.putText(bar, title, (x, 25), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (220, 220, 220), thickness, cv2.LINE_AA)
    return np.vstack([bar, img])

def visualize_prediction(img_bgr, pred_mask, gt_mask=None, display_size=512):
    """Create a side-by-side comparison image."""
    img_resized  = cv2.resize(img_bgr, (display_size, display_size))
    pred_colored = mask_to_color(cv2.resize(pred_mask, (display_size, display_size),
                                            interpolation=cv2.INTER_NEAREST))
    pred_overlay = create_overlay(img_resized, cv2.resize(pred_mask, (display_size, display_size),
                                                          interpolation=cv2.INTER_NEAREST))

    panels = [
        add_title(img_resized, "Input"),
        add_title(pred_overlay, "Prediction Overlay"),
        add_title(pred_colored, "Prediction Mask"),
    ]

    if gt_mask is not None:
        gt_colored = mask_to_color(cv2.resize(gt_mask, (display_size, display_size),
                                              interpolation=cv2.INTER_NEAREST))
        panels.insert(2, add_title(gt_colored, "Ground Truth"))

    combined = np.hstack(panels)

    return combined

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize facial segmentation predictions")
    parser.add_argument("--checkpoint", type=str, default="best_model.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--num", type=int, default=5,
                        help="Number of random validation images to visualize")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a single image for inference (no ground truth)")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory for visualization images")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load model
    print(f"Loading model from {args.checkpoint}...")
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
    print("Model loaded successfully.\n")

    if args.image:
        # Single image inference
        print(f"Running inference on: {args.image}")
        img_bgr = cv2.imread(args.image)
        if img_bgr is None:
            print(f"Error: Could not read image {args.image}")
            exit(1)

        pred = predict_tta(model, img_bgr)
        result = visualize_prediction(img_bgr, pred)

        out_path = os.path.join(args.output, "prediction.png")
        cv2.imwrite(out_path, result)
        print(f"Saved: {out_path}")
    else:
        # Validation set visualization
        val_img_dir  = os.path.join("data", "images", "val")
        val_mask_dir = os.path.join("data", "annotations", "val")

        imgs  = sorted(glob(os.path.join(val_img_dir,  "*.jpg")) +
                       glob(os.path.join(val_img_dir,  "*.png")))
        masks = sorted(glob(os.path.join(val_mask_dir, "*.jpg")) +
                       glob(os.path.join(val_mask_dir, "*.png")))

        if len(imgs) != len(masks):
            img_dict  = {os.path.splitext(os.path.basename(p))[0]: p for p in imgs}
            mask_dict = {os.path.splitext(os.path.basename(p))[0]: p for p in masks}
            common = sorted(list(set(img_dict.keys()) & set(mask_dict.keys())))
            val_pairs = [(img_dict[k], mask_dict[k]) for k in common]
        else:
            val_pairs = list(zip(imgs, masks))

        # Select random samples
        num = min(args.num, len(val_pairs))
        samples = random.sample(val_pairs, num)

        print(f"Generating visualizations for {num} images...")
        for i, (img_p, mask_p) in enumerate(tqdm(samples, desc="Visualizing")):
            img_bgr   = cv2.imread(img_p)
            true_mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
            true_mask[true_mask >= NUM_CLASSES] = 0

            pred = predict_tta(model, img_bgr)
            result = visualize_prediction(img_bgr, pred, gt_mask=true_mask)

            name = os.path.splitext(os.path.basename(img_p))[0]
            out_path = os.path.join(args.output, f"{name}.png")
            cv2.imwrite(out_path, result)

        print(f"\nSaved {num} visualizations to {args.output}/")
        print("Use these images in your README to showcase the model's predictions.")
