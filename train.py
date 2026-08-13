"""
Facial Segmentation — Training Pipeline
========================================
Two-phase transfer learning for multi-class facial segmentation.

Architecture: UNet++ with EfficientNet-B3 encoder and scSE attention
Resolution:   1024x1024
Classes:      9 (3 large regions + 6 small facial features)

Phase 1: Frozen encoder — trains decoder from scratch
Phase 2: Full fine-tuning with differential learning rates

Usage:
    python train.py
"""

import os, sys, random, warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast

import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark     = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(f"VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
class CFG:
    img_size        = 1024
    num_classes     = 9
    batch_size      = 2
    accum_steps     = 16        # Effective batch size = 32
    epochs_frozen   = 10        # Phase 1: decoder only
    epochs_finetune = 30        # Phase 2: full network
    lr_frozen       = 5e-4
    lr_encoder      = 1e-5      # Differential LR for Phase 2
    lr_decoder      = 1e-4      # Differential LR for Phase 2
    weight_decay    = 1e-4
    num_workers     = 2
    checkpoint      = "best_model.pth"
    mixed_precision = True

# ─────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────
BASE = "data"

TRAIN_IMG  = os.path.join(BASE, "images", "train")
TRAIN_MASK = os.path.join(BASE, "annotations", "train")
VAL_IMG    = os.path.join(BASE, "images", "val")
VAL_MASK   = os.path.join(BASE, "annotations", "val")

def get_pairs(img_dir, mask_dir):
    """Match image files to their corresponding mask files."""
    imgs  = sorted(glob(os.path.join(img_dir,  "*.jpg")) +
                   glob(os.path.join(img_dir,  "*.png")))
    masks = sorted(glob(os.path.join(mask_dir, "*.jpg")) +
                   glob(os.path.join(mask_dir, "*.png")))
    if len(imgs) != len(masks):
        print(f"Warning: {len(imgs)} images vs {len(masks)} masks. Aligning by filename.")
        img_dict = {os.path.splitext(os.path.basename(p))[0]: p for p in imgs}
        mask_dict = {os.path.splitext(os.path.basename(p))[0]: p for p in masks}
        common = sorted(list(set(img_dict.keys()) & set(mask_dict.keys())))
        return [(img_dict[k], mask_dict[k]) for k in common]
    return list(zip(imgs, masks))

train_pairs = get_pairs(TRAIN_IMG, TRAIN_MASK)
val_pairs   = get_pairs(VAL_IMG,   VAL_MASK)
print(f"Train: {len(train_pairs):,}   Val: {len(val_pairs):,}")

# ─────────────────────────────────────────────────────────────
# Augmentation Pipeline
# ─────────────────────────────────────────────────────────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

train_tfm = A.Compose([
    A.RandomResizedCrop(size=(CFG.img_size, CFG.img_size), scale=(0.7, 1.0), ratio=(0.8, 1.2), p=1.0),
    A.HorizontalFlip(p=0.5),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8)),
    ], p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
    A.OneOf([
        A.GaussNoise(var_limit=(10.0, 50.0)),
        A.GaussianBlur(blur_limit=(3, 5)),
    ], p=0.2),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_REFLECT, p=0.5),
    A.OneOf([
        A.ElasticTransform(alpha=50, sigma=5, border_mode=cv2.BORDER_REFLECT),
        A.GridDistortion(num_steps=5, distort_limit=0.3, border_mode=cv2.BORDER_REFLECT),
    ], p=0.2),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

val_tfm = A.Compose([
    A.Resize(height=CFG.img_size, width=CFG.img_size),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────
class FaceSegDataset(Dataset):
    def __init__(self, pairs, transform):
        self.pairs     = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img  = cv2.cvtColor(cv2.imread(img_path),  cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask[mask >= CFG.num_classes] = 0

        aug  = self.transform(image=img, mask=mask)
        img_t  = aug['image'].float()
        mask_t = aug['mask'].long()
        return img_t, mask_t

train_ds = FaceSegDataset(train_pairs, train_tfm)
val_ds   = FaceSegDataset(val_pairs,   val_tfm)

# ─────────────────────────────────────────────────────────────
# Model: UNet++ with EfficientNet-B3 backbone
# ─────────────────────────────────────────────────────────────
print("\nBuilding model...")

model = smp.UnetPlusPlus(
    encoder_name="efficientnet-b3",
    encoder_weights="imagenet",
    in_channels=3,
    classes=CFG.num_classes,
    activation=None,
    decoder_attention_type="scse",
)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params:     {total_params:,}")
print(f"Trainable params: {trainable_params:,}")

# ─────────────────────────────────────────────────────────────
# Loss Function
# ─────────────────────────────────────────────────────────────
class CombinedLoss(nn.Module):
    """Combines weighted Cross-Entropy, Focal Loss, and Dice Loss.

    Uses aggressive class weights to counteract the extreme pixel-level
    imbalance between large regions (background, skin, hair) and small
    facial features (eyebrows, eyes, nose, mouth).
    """
    def __init__(self, num_classes):
        super().__init__()
        weights = torch.tensor([0.1, 0.5, 1.0, 50.0, 50.0, 100.0, 100.0, 50.0, 100.0]).float()
        self.ce = nn.CrossEntropyLoss(weight=weights)
        self.focal = smp.losses.FocalLoss(mode='multiclass', gamma=2.0)
        self.dice = smp.losses.DiceLoss(mode='multiclass')

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        focal_loss = self.focal(logits, targets)
        dice_loss = self.dice(logits, targets)
        return 0.4 * ce_loss + 0.4 * focal_loss + 0.2 * dice_loss

# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def compute_dice(logits, targets, num_classes=CFG.num_classes, smooth=1e-6):
    """Compute macro-averaged Dice score across all classes."""
    pred = logits.argmax(dim=1)
    dice = []
    for c in range(num_classes):
        gt     = (targets == c).float()
        pred_c = (pred == c).float()
        inter  = (gt * pred_c).sum()
        denom  = gt.sum() + pred_c.sum()
        dice.append(((2 * inter + smooth) / (denom + smooth)).item())
    return np.mean(dice)

# ─────────────────────────────────────────────────────────────
# Training Helpers
# ─────────────────────────────────────────────────────────────
criterion = CombinedLoss(CFG.num_classes).to(DEVICE)
scaler    = GradScaler(enabled=CFG.mixed_precision)

def freeze_encoder(model):
    """Freeze encoder weights for Phase 1 training."""
    for p in model.encoder.parameters():
        p.requires_grad = False

def unfreeze_encoder(model):
    """Unfreeze encoder weights for Phase 2 fine-tuning."""
    for p in model.encoder.parameters():
        p.requires_grad = True

def make_optimizer_phase1(model, lr):
    """Create optimizer for Phase 1 (decoder parameters only)."""
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=CFG.weight_decay)

def make_optimizer_phase2(model, lr_encoder, lr_decoder):
    """Create optimizer for Phase 2 with differential learning rates."""
    encoder_params = []
    decoder_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "encoder" in name:
            encoder_params.append(p)
        else:
            decoder_params.append(p)

    return torch.optim.AdamW([
        {'params': encoder_params, 'lr': lr_encoder},
        {'params': decoder_params, 'lr': lr_decoder},
    ], weight_decay=CFG.weight_decay)

def train_one_epoch(model, loader, optimizer, device, epoch):
    """Run one training epoch with gradient accumulation and mixed precision."""
    model.train()
    total_loss = 0.0
    total_dice = 0.0

    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"  Train Ep {epoch}", leave=False)

    for i, (imgs, masks) in enumerate(pbar):
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with autocast('cuda', enabled=CFG.mixed_precision):
            logits = model(imgs)
            loss   = criterion(logits, masks)
            loss   = loss / CFG.accum_steps

        scaler.scale(loss).backward()

        if (i + 1) % CFG.accum_steps == 0 or (i + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        dice = compute_dice(logits.detach(), masks.detach())
        total_loss += loss.item() * CFG.accum_steps
        total_dice += dice
        pbar.set_postfix(loss=f"{loss.item()*CFG.accum_steps:.4f}", dice=f"{dice:.4f}")

    n = len(loader)
    return total_loss / n, total_dice / n

@torch.no_grad()
def validate(model, loader, device, epoch):
    """Run validation and return average loss and Dice score."""
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    for imgs, masks in tqdm(loader, desc=f"  Val Ep {epoch}", leave=False):
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with autocast('cuda', enabled=CFG.mixed_precision):
            logits = model(imgs)
            loss   = criterion(logits, masks)
        total_loss += loss.item()
        total_dice += compute_dice(logits, masks)
    n = len(loader)
    return total_loss / n, total_dice / n

# ─────────────────────────────────────────────────────────────
# Main Training Loop
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size,
                              shuffle=True,  num_workers=CFG.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG.batch_size,
                              shuffle=False, num_workers=CFG.num_workers,
                              pin_memory=True)

    print(f"\nTrain batches: {len(train_loader)}   Val batches: {len(val_loader)}")

    model.to(DEVICE)

    # ── Checkpoint Resumption ────────────────────────────────
    start_phase = 1
    start_epoch = 1
    best_dice = 0.0
    checkpoint_data = None

    if os.path.exists(CFG.checkpoint):
        print(f"Loading checkpoint {CFG.checkpoint}...")
        checkpoint_data = torch.load(CFG.checkpoint, map_location=DEVICE, weights_only=False)
        if isinstance(checkpoint_data, dict) and 'model_state_dict' in checkpoint_data:
            model.load_state_dict(checkpoint_data['model_state_dict'])
            start_phase = checkpoint_data.get('phase', 1)
            start_epoch = checkpoint_data.get('epoch', 0) + 1
            best_dice = checkpoint_data.get('best_dice', 0.0)
            print(f"Resuming from Phase {start_phase}, Epoch {start_epoch-1} with Best Dice: {best_dice:.5f}")
        else:
            model.load_state_dict(checkpoint_data)
            checkpoint_data = None

    log_rows = []

    # ── Phase 1: Frozen Encoder ──────────────────────────────
    if start_phase == 1:
        freeze_encoder(model)
        print("\n" + "="*60)
        print("PHASE 1: Frozen encoder — training decoder only")
        print("="*60)

        optimizer = make_optimizer_phase1(model, CFG.lr_frozen)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=3, factor=0.5, min_lr=1e-6)

        if checkpoint_data and start_phase == 1:
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])

        patience_count = 0
        PATIENCE = 6

        for epoch in range(start_epoch, CFG.epochs_frozen + 1):
            print(f"\nPhase 1 Epoch {epoch}/{CFG.epochs_frozen}")
            tr_loss, tr_dice = train_one_epoch(model, train_loader, optimizer, DEVICE, epoch)
            vl_loss, vl_dice = validate(model, val_loader, DEVICE, epoch)
            scheduler.step(vl_dice)

            print(f"  Train — loss: {tr_loss:.4f}  dice: {tr_dice:.4f}")
            print(f"  Val   — loss: {vl_loss:.4f}  dice: {vl_dice:.4f}")
            log_rows.append({'phase':1, 'epoch':epoch,
                             'tr_loss':tr_loss, 'tr_dice':tr_dice,
                             'vl_loss':vl_loss, 'vl_dice':vl_dice})

            if vl_dice > best_dice:
                best_dice = vl_dice
                torch.save({
                    'phase': 1,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_dice': best_dice
                }, CFG.checkpoint)
                print(f"  ✓ New best val Dice: {best_dice:.5f} — saved checkpoint")
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= PATIENCE:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        print(f"\nPhase 1 completed. Best val Dice = {best_dice:.5f}")
        start_epoch = 1

    # ── Phase 2: Full Fine-Tuning ────────────────────────────
    if start_phase <= 2:
        print(f"\nLoading best checkpoint before Phase 2...")
        checkpoint = torch.load(CFG.checkpoint, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'] if isinstance(checkpoint, dict) else checkpoint)
        unfreeze_encoder(model)

        print("\n" + "="*60)
        print("PHASE 2: Fine-tuning full network (Differential LR)")
        print("="*60)

        optimizer = make_optimizer_phase2(model, CFG.lr_encoder, CFG.lr_decoder)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=4, factor=0.5, min_lr=1e-7)

        if checkpoint_data and start_phase == 2:
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])

        patience_count = 0
        PATIENCE = 10

        for epoch in range(start_epoch, CFG.epochs_finetune + 1):
            print(f"\nPhase 2 Epoch {epoch}/{CFG.epochs_finetune}")
            tr_loss, tr_dice = train_one_epoch(model, train_loader, optimizer, DEVICE, epoch)
            vl_loss, vl_dice = validate(model, val_loader, DEVICE, epoch)
            scheduler.step(vl_dice)

            print(f"  Train — loss: {tr_loss:.4f}  dice: {tr_dice:.4f}")
            print(f"  Val   — loss: {vl_loss:.4f}  dice: {vl_dice:.4f}")
            log_rows.append({'phase':2, 'epoch':epoch,
                             'tr_loss':tr_loss, 'tr_dice':tr_dice,
                             'vl_loss':vl_loss, 'vl_dice':vl_dice})

            if vl_dice > best_dice:
                best_dice = vl_dice
                torch.save({
                    'phase': 2,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_dice': best_dice
                }, CFG.checkpoint)
                print(f"  ✓ New best val Dice: {best_dice:.5f} — saved checkpoint")
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= PATIENCE:
                    print(f"  Early stopping at epoch {epoch}")
                    break

    # ── Save Training Log ────────────────────────────────────
    if log_rows:
        pd.DataFrame(log_rows).to_csv("training_log.csv", index=False)

    print(f"\nOverall best val Dice: {best_dice:.5f}")

    # ── Reload Best Weights ──────────────────────────────────
    checkpoint = torch.load(CFG.checkpoint, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'] if isinstance(checkpoint, dict) else checkpoint)
    model.eval()
    print(f"Loaded best model ({CFG.checkpoint})")

    # ── Final Validation with Test-Time Augmentation ─────────
    print("\nFinal validation with Test-Time Augmentation...")

    def np_dice(y_true, y_pred, n=CFG.num_classes, smooth=1e-6):
        scores = []
        for c in range(n):
            gt   = (y_true == c).astype(np.float32)
            pred = (y_pred == c).astype(np.float32)
            inter = (gt * pred).sum()
            denom = gt.sum() + pred.sum()
            scores.append((2 * inter + smooth) / (denom + smooth))
        return float(np.mean(scores))

    def preprocess_img(img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        aug     = val_tfm(image=img_rgb)['image']
        return aug.unsqueeze(0).to(DEVICE)

    @torch.no_grad()
    def predict_tta(model, img_bgr):
        orig_h, orig_w = img_bgr.shape[:2]
        inp1 = preprocess_img(img_bgr)
        inp2 = preprocess_img(img_bgr[:, ::-1, :].copy())

        with autocast('cuda', enabled=CFG.mixed_precision):
            p1 = model(inp1).softmax(dim=1)
            p2 = model(inp2).softmax(dim=1).flip(dims=[3])

        avg = ((p1 + p2) / 2.0)[0]
        avg = avg.permute(1, 2, 0).cpu().numpy()
        avg_full = cv2.resize(avg, (orig_w, orig_h))
        return np.argmax(avg_full, axis=-1).astype(np.uint8)

    val_dices = []
    for img_p, mask_p in tqdm(val_pairs, desc="Val TTA"):
        img_bgr   = cv2.imread(img_p)
        true_mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        true_mask[true_mask >= CFG.num_classes] = 0
        pred      = predict_tta(model, img_bgr)
        val_dices.append(np_dice(true_mask, pred))

    print(f"\n{'='*60}")
    print(f"  Val Dice (TTA): {np.mean(val_dices):.5f}")
    print(f"  Min: {np.min(val_dices):.5f}   Max: {np.max(val_dices):.5f}")
    print(f"{'='*60}")
    print("\n[DONE] Training complete.")
