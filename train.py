# train.py
# Full training loop for SOD Model with checkpoint save/resume
#
# PROJECT: Xponian Program Cohort IV - Homework #3
# DESCRIPTION: Training pipeline with automatic comparison
#
# FEATURES:
# - Full training loop with forward/backward passes
# - Validation after each epoch
# - Early stopping with patience
# - Checkpoint save/resume (BONUS)
# - Trains both baseline and improved models
# - Automatic comparison table generation
#
# USAGE:
#   Set TRAIN_BASELINE and TRAIN_IMPROVED flags (lines 312-313)
#   Run: python train.py
#
# OUTPUT:
#   - best_model_baseline.pth (baseline model)
#   - best_model_improved.pth (improved model)
#   - Comparison table showing improvements

import os
import sys
import time
import torch
import torch.optim as optim
from sod_model import SODModel, bce_iou_loss, compute_iou, compute_dice
from data_loader import create_data_splits


# ========================================
# CHECKPOINT FUNCTIONS (BONUS)
# ========================================

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, val_iou, 
                   checkpoint_path="checkpoint.pth", best_path="best_model.pth", 
                   is_best=False):
    """
    Save model checkpoint with all training state
    
    BONUS: Saves model weights, optimizer state, and current epoch
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_iou': val_iou
    }
    
    # Save regular checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"[CHECKPOINT] Saved at epoch {epoch} -> {checkpoint_path}")
    
    # Save best model
    if is_best:
        torch.save(checkpoint, best_path)
        print(f"[BEST MODEL] Saved at epoch {epoch} with Val IoU: {val_iou:.4f}")


def load_checkpoint(model, optimizer, checkpoint_path="checkpoint.pth"):
    """
    Load checkpoint and resume training
    
    BONUS: Automatically loads last checkpoint and continues training
    """
    if os.path.exists(checkpoint_path):
        print(f"\n[RESUME] Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        
        print(f"[RESUME] Checkpoint loaded successfully!")
        print(f"[RESUME] Resuming from epoch {start_epoch}")
        print(f"[RESUME] Previous - Train Loss: {checkpoint['train_loss']:.4f}, "
              f"Val Loss: {checkpoint['val_loss']:.4f}, Val IoU: {checkpoint['val_iou']:.4f}")
        
        return start_epoch
    else:
        print(f"[INFO] No checkpoint found at {checkpoint_path}")
        print(f"[INFO] Starting training from scratch")
        return 0


# ========================================
# TRAINING LOOP
# ========================================

def train_one_epoch(model, train_loader, optimizer, device):
    """
    Train for one epoch
    
    Includes:
    - Forward pass
    - Backward pass
    - Loss and metrics logging
    """
    model.train()
    
    epoch_loss = 0.0
    epoch_iou = 0.0
    epoch_dice = 0.0
    num_batches = len(train_loader)
    
    for batch_idx, (imgs, masks) in enumerate(train_loader):
        # Move to device
        imgs = imgs.to(device)
        masks = masks.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(imgs)
        
        # Compute loss
        loss = bce_iou_loss(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Compute metrics
        iou = compute_iou(outputs, masks)
        dice = compute_dice(outputs, masks)
        
        # Accumulate
        epoch_loss += loss.item()
        epoch_iou += iou
        epoch_dice += dice
        
        # Print progress every 50 batches
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch [{batch_idx+1}/{num_batches}] - "
                  f"Loss: {loss.item():.4f}, IoU: {iou:.4f}, Dice: {dice:.4f}")
    
    # Average metrics
    epoch_loss /= num_batches
    epoch_iou /= num_batches
    epoch_dice /= num_batches
    
    return epoch_loss, epoch_iou, epoch_dice


def validate(model, val_loader, device):
    """
    Validate model on validation set
    
    Includes:
    - Forward pass (no gradients)
    - Loss and metrics computation
    """
    model.eval()
    
    val_loss = 0.0
    val_iou = 0.0
    val_dice = 0.0
    num_batches = len(val_loader)
    
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(imgs)
            
            # Compute loss and metrics
            loss = bce_iou_loss(outputs, masks)
            iou = compute_iou(outputs, masks)
            dice = compute_dice(outputs, masks)
            
            val_loss += loss.item()
            val_iou += iou
            val_dice += dice
    
    # Average metrics
    val_loss /= num_batches
    val_iou /= num_batches
    val_dice /= num_batches
    
    return val_loss, val_iou, val_dice


def train_model(model, train_loader, val_loader, num_epochs=20, lr=1e-3, 
                patience=5, device='cpu', checkpoint_path="checkpoint.pth",
                best_path="best_model.pth", resume=True, model_name="baseline"):
    """
    Full training loop with:
    - Forward and backward passes
    - Logging of loss and metrics
    - Validation after each epoch
    - Saving best model
    - Early stopping
    - BONUS: Checkpoint save/resume
    
    Args:
        model: SODModel instance
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Maximum number of epochs (15-25 recommended)
        lr: Learning rate (1e-3 recommended)
        patience: Early stopping patience
        device: 'cpu' or 'cuda'
        checkpoint_path: Path to save checkpoints
        best_path: Path to save best model
        resume: Whether to resume from checkpoint
        model_name: Name of model variant (for logging) - "baseline" or "improved"
    """
    
    # Optimizer: Adam with lr=1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Move model to device
    model = model.to(device)
    
    # BONUS: Load checkpoint if exists and resume=True
    start_epoch = 0
    if resume:
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
    
    # Early stopping variables
    best_val_iou = 0.0
    patience_counter = 0
    
    # Training history
    history = {
        'train_loss': [],
        'train_iou': [],
        'train_dice': [],
        'val_loss': [],
        'val_iou': [],
        'val_dice': []
    }
    
    print("\n" + "="*70)
    print(f"STARTING TRAINING - {model_name.upper()} MODEL")
    print("="*70)
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {lr}")
    print(f"Patience: {patience}")
    print(f"Optimizer: Adam")
    print(f"Loss: BCE + 0.5 * (1 - IoU)")
    print(f"Model variant: {model_name}")
    print("="*70 + "\n")
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print("-" * 70)
        
        # Train one epoch
        train_loss, train_iou, train_dice = train_one_epoch(
            model, train_loader, optimizer, device
        )
        
        # Validate
        val_loss, val_iou, val_dice = validate(model, val_loader, device)
        
        # Log metrics
        history['train_loss'].append(train_loss)
        history['train_iou'].append(train_iou)
        history['train_dice'].append(train_dice)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        history['val_dice'].append(val_dice)
        
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch summary
        print(f"\n[EPOCH {epoch+1} SUMMARY]")
        print(f"  Train - Loss: {train_loss:.4f}, IoU: {train_iou:.4f}, Dice: {train_dice:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Dice: {val_dice:.4f}")
        print(f"  Time: {epoch_time:.2f}s")
        
        # Check if best model
        is_best = val_iou > best_val_iou
        if is_best:
            best_val_iou = val_iou
            patience_counter = 0
        else:
            patience_counter += 1
        
        # BONUS: Save checkpoint after each epoch
        save_checkpoint(
            model, optimizer, epoch, train_loss, val_loss, val_iou,
            checkpoint_path, best_path, is_best
        )
        
        # Early stopping check
        if patience_counter >= patience:
            print(f"\n[EARLY STOPPING] No improvement for {patience} epochs")
            print(f"[EARLY STOPPING] Best Val IoU: {best_val_iou:.4f}")
            break
        
        print("=" * 70 + "\n")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best Val IoU: {best_val_iou:.4f}")
    print(f"Best model saved at: {best_path}")
    print("="*70)
    
    return history


# ========================================
# MAIN EXECUTION
# ========================================

if __name__ == "__main__":
    # Configuration
    IMAGES_PATH = "images/"
    MASKS_PATH = "ground_truth_mask/"
    BATCH_SIZE = 8
    NUM_EPOCHS = 30  # Increased for better convergence (use 30-40 for large datasets)
    LEARNING_RATE = 1e-3  # As required
    PATIENCE = 10  # Patience for early stopping
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ========================================
    # IMPROVEMENT: Train both baseline and improved models
    # ========================================
    TRAIN_BASELINE = True   # Train baseline first
    TRAIN_IMPROVED = True   # Then train improved model
    
    print("="*70)
    print("SOD MODEL TRAINING - BASELINE vs IMPROVED")
    print("="*70)
    print(f"Train Baseline: {TRAIN_BASELINE}")
    print(f"Train Improved: {TRAIN_IMPROVED}")
    print("="*70)
    
    results = {}
    
    # ========================================
    # BASELINE MODEL TRAINING
    # ========================================
    if TRAIN_BASELINE:
        print("\n" + "="*70)
        print("TRAINING BASELINE MODEL")
        print("="*70)
        
        # Load data (baseline augmentations)
        print("\n[1/4] Loading dataset (baseline augmentations)...")
        train_loader, val_loader, test_loader = create_data_splits(
            IMAGES_PATH, MASKS_PATH, batch_size=BATCH_SIZE, use_improvements=False
        )
        print("[OK] Dataset loaded")
        
        # Create baseline model
        print("\n[2/4] Creating baseline model...")
        model_baseline = SODModel(use_improvements=False)
        print(f"[OK] Baseline model created with {model_baseline.count_parameters():,} parameters")
        
        # Train baseline model
        print("\n[3/4] Training baseline model...")
        history_baseline = train_model(
            model=model_baseline,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=NUM_EPOCHS,
            lr=LEARNING_RATE,
            patience=PATIENCE,
            device=DEVICE,
            checkpoint_path="checkpoint_baseline.pth",
            best_path="best_model_baseline.pth",
            resume=True,
            model_name="baseline"
        )
        
        # Evaluate baseline on test set
        print("\n[4/4] Evaluating baseline on test set...")
        model_baseline.load_state_dict(torch.load("best_model_baseline.pth")['model_state_dict'])
        test_loss_b, test_iou_b, test_dice_b = validate(model_baseline, test_loader, DEVICE)
        
        results['baseline'] = {
            'test_loss': test_loss_b,
            'test_iou': test_iou_b,
            'test_dice': test_dice_b,
            'history': history_baseline
        }
        
        print("\n" + "="*70)
        print("BASELINE MODEL - FINAL TEST RESULTS")
        print("="*70)
        print(f"Test Loss: {test_loss_b:.4f}")
        print(f"Test IoU:  {test_iou_b:.4f}")
        print(f"Test Dice: {test_dice_b:.4f}")
        print("="*70)
    
    # ========================================
    # IMPROVED MODEL TRAINING
    # ========================================
    if TRAIN_IMPROVED:
        print("\n" + "="*70)
        print("TRAINING IMPROVED MODEL")
        print("="*70)
        print("IMPROVEMENTS:")
        print("  - Batch Normalization")
        print("  - Dropout (p=0.3)")
        print("  - Deeper layers (3 conv per block)")
        print("  - Skip connections (U-Net style)")
        print("  - Enhanced data augmentations")
        print("="*70)
        
        # Load data (improved augmentations)
        print("\n[1/4] Loading dataset (improved augmentations)...")
        train_loader_imp, val_loader_imp, test_loader_imp = create_data_splits(
            IMAGES_PATH, MASKS_PATH, batch_size=BATCH_SIZE, use_improvements=True
        )
        print("[OK] Dataset loaded")
        
        # Create improved model
        print("\n[2/4] Creating improved model...")
        print("IMPROVEMENTS:")
        print("  1. Batch Normalization - Stabilizes training")
        print("  2. Dropout (0.2) - Prevents overfitting")
        print("  3. Deeper layers (3 conv per block) - Better feature extraction")
        print("  4. Skip connections (U-Net style) - Preserves spatial info")
        model_improved = SODModel(use_improvements=True, dropout_rate=0.2)
        print(f"[OK] Improved model created with {model_improved.count_parameters():,} parameters")
        
        # Train improved model
        print("\n[3/4] Training improved model...")
        history_improved = train_model(
            model=model_improved,
            train_loader=train_loader_imp,
            val_loader=val_loader_imp,
            num_epochs=NUM_EPOCHS,
            lr=LEARNING_RATE,
            patience=PATIENCE,
            device=DEVICE,
            checkpoint_path="checkpoint_improved_v2.pth",
            best_path="best_model_improved_v2.pth",
            resume=True,  # Resume from checkpoint if exists
            model_name="improved"
        )
        
        # Evaluate improved on test set
        print("\n[4/4] Evaluating improved on test set...")
        model_improved.load_state_dict(torch.load("best_model_improved.pth")['model_state_dict'])
        test_loss_i, test_iou_i, test_dice_i = validate(model_improved, test_loader_imp, DEVICE)
        
        results['improved'] = {
            'test_loss': test_loss_i,
            'test_iou': test_iou_i,
            'test_dice': test_dice_i,
            'history': history_improved
        }
        
        print("\n" + "="*70)
        print("IMPROVED MODEL - FINAL TEST RESULTS")
        print("="*70)
        print(f"Test Loss: {test_loss_i:.4f}")
        print(f"Test IoU:  {test_iou_i:.4f}")
        print(f"Test Dice: {test_dice_i:.4f}")
        print("="*70)
    
    # ========================================
    # COMPARISON TABLE
    # ========================================
    if TRAIN_BASELINE and TRAIN_IMPROVED:
        print("\n" + "="*70)
        print("BASELINE vs IMPROVED - COMPARISON")
        print("="*70)
        print(f"{'Metric':<20} {'Baseline':<15} {'Improved':<15} {'Change':<15}")
        print("-"*70)
        
        baseline_iou = results['baseline']['test_iou']
        improved_iou = results['improved']['test_iou']
        iou_change = ((improved_iou - baseline_iou) / baseline_iou) * 100
        
        baseline_dice = results['baseline']['test_dice']
        improved_dice = results['improved']['test_dice']
        dice_change = ((improved_dice - baseline_dice) / baseline_dice) * 100
        
        baseline_loss = results['baseline']['test_loss']
        improved_loss = results['improved']['test_loss']
        loss_change = ((improved_loss - baseline_loss) / baseline_loss) * 100
        
        print(f"{'Test IoU':<20} {baseline_iou:<15.4f} {improved_iou:<15.4f} {iou_change:+.2f}%")
        print(f"{'Test Dice':<20} {baseline_dice:<15.4f} {improved_dice:<15.4f} {dice_change:+.2f}%")
        print(f"{'Test Loss':<20} {baseline_loss:<15.4f} {improved_loss:<15.4f} {loss_change:+.2f}%")
        print("="*70)
        
        print("\nIMPROVEMENTS APPLIED:")
        print("  1. Batch Normalization - Stabilizes training")
        print("  2. Dropout (0.3) - Prevents overfitting")
        print("  3. Deeper layers - Better feature extraction")
        print("  4. Skip connections - Preserves spatial info")
        print("  5. Vertical flip - Additional augmentation")
        print("  6. Color jittering - More robust to color variations")
        print("  7. Stronger augmentations - Better generalization")
    
    print("\n[SUCCESS] Training complete!")
    print("\nFiles saved:")
    if TRAIN_BASELINE:
        print("  - checkpoint_baseline.pth (baseline checkpoint)")
        print("  - best_model_baseline.pth (best baseline model)")
    if TRAIN_IMPROVED:
        print("  - checkpoint_improved_v2.pth (improved checkpoint)")
        print("  - best_model_improved_v2.pth (best improved model)")
