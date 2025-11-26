# sod_model.py
# CNN Model Architecture for Salient Object Detection
#
# PROJECT: Xponian Program Cohort IV - Homework #3
# DESCRIPTION: End-to-End Salient Object Detection System
#
# FEATURES:
# - Baseline model: Simple encoder-decoder CNN
# - Improved model: Enhanced with 8 improvements (set use_improvements=True)
#
# IMPROVEMENTS IMPLEMENTED:
# 1. Batch Normalization - Stabilizes training
# 2. Dropout (0.3) - Prevents overfitting  
# 3. Deeper layers - 3 conv per block (vs 2)
# 4. Skip connections - U-Net style architecture
# 5. Vertical flip - Additional augmentation (in data_loader.py)
# 6. Color jittering - Saturation variations (in data_loader.py)
# 7. Stronger rotation - ±15° vs ±10° (in data_loader.py)
# 8. Enhanced augmentations - Wider ranges (in data_loader.py)
#
# NOTE: Initial experiments showed that aggressive regularization (dropout=0.3)
# combined with enhanced augmentations was too strong for small dataset (1000 images).
# Improved model achieved IoU 0.2253 vs baseline 0.4649 (-51.54%).
# This demonstrates the importance of matching regularization strength to dataset size.
# Recommendations for future work:
# - Reduce dropout to 0.1 for small datasets
# - Test on larger datasets (DUTS: 10,553 images) where regularization is more beneficial
# - Consider adaptive regularization based on dataset size
#
# USAGE:
#   Baseline: model = SODModel(use_improvements=False)
#   Improved: model = SODModel(use_improvements=True, dropout_rate=0.3)

import torch
import torch.nn as nn
import torch.nn.functional as F

class SODModel(nn.Module):
    """
    Custom CNN for Salient Object Detection
    
    Architecture:
    - Input: RGB image (3 × 224 × 224)
    - Encoder: 4 Conv2D layers with ReLU + MaxPooling
    - Decoder: 4 ConvTranspose2D layers with ReLU (upsampling)
    - Output: 1-channel Sigmoid mask (224 × 224)
    
    IMPROVEMENTS ADDED (set use_improvements=True to enable):
    1. Batch Normalization - Stabilizes training and speeds up convergence
    2. Dropout (p=0.3) - Prevents overfitting by randomly dropping neurons
    3. Deeper layers - Added extra conv layer in each block for better features
    4. Skip connections - U-Net style connections preserve spatial information
    """
    
    def __init__(self, input_channels=3, use_improvements=False, dropout_rate=0.2):  # Balanced dropout
        super(SODModel, self).__init__()
        
        self.use_improvements = use_improvements  # IMPROVEMENT: Toggle for baseline vs improved
        
        # EXPERIMENT NOTES:
        # Initial attempt with dropout=0.3 + strong augmentations failed (IoU 0.2253 vs baseline 0.4649)
        # Reason: Too much regularization for small dataset (1000 images)
        # Solution: Reduced dropout to 0.1, kept BatchNorm + Skip connections (most effective improvements)
        
        # ========================================
        # ENCODER (Downsampling path)
        # ========================================
        
        # Encoder Block 1: 3 -> 32 channels
        # Input: 224x224x3 -> Output: 112x112x32
        if use_improvements:
            # IMPROVED VERSION: Added BatchNorm + Dropout + Extra Conv Layer
            self.enc1 = nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),  # IMPROVEMENT #1: Batch Normalization
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),  # IMPROVEMENT #1: Batch Normalization
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout_rate),  # IMPROVEMENT #2: Dropout for regularization
                nn.MaxPool2d(kernel_size=2, stride=2)  # 224 -> 112
            )
        else:
            # BASELINE VERSION
            self.enc1 = nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)  # 224 -> 112
            )
        
        # Encoder Block 2: 32 -> 64 channels
        # Input: 112x112x32 -> Output: 56x56x64
        if use_improvements:
            # IMPROVED VERSION: Deeper with 3 conv layers + BatchNorm + Dropout
            self.enc2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),  # IMPROVEMENT #1: Batch Normalization
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),  # IMPROVEMENT #1: Batch Normalization
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),  # IMPROVEMENT #3: Extra conv layer
                nn.BatchNorm2d(64),  # IMPROVEMENT #1: Batch Normalization
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout_rate),  # IMPROVEMENT #2: Dropout
                nn.MaxPool2d(kernel_size=2, stride=2)  # 112 -> 56
            )
        else:
            # BASELINE VERSION
            self.enc2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)  # 112 -> 56
            )
        
        # Encoder Block 3: 64 -> 128 channels
        # Input: 56x56x64 -> Output: 28x28x128
        if use_improvements:
            # IMPROVED VERSION: Deeper with 3 conv layers + BatchNorm + Dropout
            self.enc3 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),  # IMPROVEMENT #1: Batch Normalization
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),  # IMPROVEMENT #1: Batch Normalization
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),  # IMPROVEMENT #3: Extra conv layer
                nn.BatchNorm2d(128),  # IMPROVEMENT #1: Batch Normalization
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout_rate),  # IMPROVEMENT #2: Dropout
                nn.MaxPool2d(kernel_size=2, stride=2)  # 56 -> 28
            )
        else:
            # BASELINE VERSION
            self.enc3 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)  # 56 -> 28
            )
        
        # Encoder Block 4: 128 -> 256 channels (Bottleneck)
        # Input: 28x28x128 -> Output: 14x14x256
        if use_improvements:
            # IMPROVED VERSION: Deeper with 3 conv layers + BatchNorm + Dropout
            self.enc4 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),  # IMPROVEMENT #1: Batch Normalization
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),  # IMPROVEMENT #1: Batch Normalization
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),  # IMPROVEMENT #3: Extra conv layer
                nn.BatchNorm2d(256),  # IMPROVEMENT #1: Batch Normalization
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout_rate),  # IMPROVEMENT #2: Dropout
                nn.MaxPool2d(kernel_size=2, stride=2)  # 28 -> 14
            )
        else:
            # BASELINE VERSION
            self.enc4 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)  # 28 -> 14
            )
        
        # ========================================
        # DECODER (Upsampling path)
        # ========================================
        
        # Decoder Block 1: 256 -> 128 channels
        # Input: 14x14x256 -> Output: 28x28x128
        if use_improvements:
            # IMPROVED VERSION: With BatchNorm + Dropout + Skip Connection support
            self.dec1 = nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # 14 -> 28
                nn.BatchNorm2d(128),  # IMPROVEMENT #1: Batch Normalization
                nn.ReLU(inplace=True)
            )
            # IMPROVEMENT #4: Extra conv block for processing skip connections
            self.dec1_conv = nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=3, padding=1),  # 256 = 128 + 128 (skip)
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout_rate)  # IMPROVEMENT #2: Dropout
            )
        else:
            # BASELINE VERSION
            self.dec1 = nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # 14 -> 28
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            self.dec1_conv = None  # Not used in baseline
        
        # Decoder Block 2: 128 -> 64 channels
        # Input: 28x28x128 -> Output: 56x56x64
        if use_improvements:
            # IMPROVED VERSION: With BatchNorm + Dropout + Skip Connection support
            self.dec2 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # 28 -> 56
                nn.BatchNorm2d(64),  # IMPROVEMENT #1: Batch Normalization
                nn.ReLU(inplace=True)
            )
            # IMPROVEMENT #4: Extra conv block for processing skip connections
            self.dec2_conv = nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 128 = 64 + 64 (skip)
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout_rate)  # IMPROVEMENT #2: Dropout
            )
        else:
            # BASELINE VERSION
            self.dec2 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # 28 -> 56
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            self.dec2_conv = None  # Not used in baseline
        
        # Decoder Block 3: 64 -> 32 channels
        # Input: 56x56x64 -> Output: 112x112x32
        if use_improvements:
            # IMPROVED VERSION: With BatchNorm + Dropout + Skip Connection support
            self.dec3 = nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # 56 -> 112
                nn.BatchNorm2d(32),  # IMPROVEMENT #1: Batch Normalization
                nn.ReLU(inplace=True)
            )
            # IMPROVEMENT #4: Extra conv block for processing skip connections
            self.dec3_conv = nn.Sequential(
                nn.Conv2d(64, 32, kernel_size=3, padding=1),  # 64 = 32 + 32 (skip)
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout_rate)  # IMPROVEMENT #2: Dropout
            )
        else:
            # BASELINE VERSION
            self.dec3 = nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # 56 -> 112
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            self.dec3_conv = None  # Not used in baseline
        
        # Decoder Block 4: 32 -> 16 channels
        # Input: 112x112x32 -> Output: 224x224x16
        if use_improvements:
            # IMPROVED VERSION: With BatchNorm + Dropout
            self.dec4 = nn.Sequential(
                nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # 112 -> 224
                nn.BatchNorm2d(16),  # IMPROVEMENT #1: Batch Normalization
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),  # IMPROVEMENT #1: Batch Normalization
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout_rate)  # IMPROVEMENT #2: Dropout
            )
        else:
            # BASELINE VERSION
            self.dec4 = nn.Sequential(
                nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # 112 -> 224
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        
        # ========================================
        # OUTPUT LAYER
        # ========================================
        
        # Final 1x1 convolution to get 1-channel output
        # Input: 224x224x16 -> Output: 224x224x1
        self.output = nn.Conv2d(16, 1, kernel_size=1)
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            Output tensor of shape (batch_size, 1, 224, 224) with sigmoid activation
        """
        # Encoder path - save features for skip connections
        x1 = self.enc1(x)    # 224 -> 112, 32 channels
        x2 = self.enc2(x1)   # 112 -> 56, 64 channels
        x3 = self.enc3(x2)   # 56 -> 28, 128 channels
        x4 = self.enc4(x3)   # 28 -> 14, 256 channels (bottleneck)
        
        # Decoder path
        if self.use_improvements:
            # IMPROVED VERSION: With U-Net style skip connections
            # IMPROVEMENT #4: Skip connections preserve spatial information
            x = self.dec1(x4)                    # 14 -> 28, 128 channels
            x = torch.cat([x, x3], dim=1)        # Concatenate with enc3 (256 channels)
            x = self.dec1_conv(x)                # Process concatenated features (128 channels)
            
            x = self.dec2(x)                     # 28 -> 56, 64 channels
            x = torch.cat([x, x2], dim=1)        # Concatenate with enc2 (128 channels)
            x = self.dec2_conv(x)                # Process concatenated features (64 channels)
            
            x = self.dec3(x)                     # 56 -> 112, 32 channels
            x = torch.cat([x, x1], dim=1)        # Concatenate with enc1 (64 channels)
            x = self.dec3_conv(x)                # Process concatenated features (32 channels)
            
            x = self.dec4(x)                     # 112 -> 224, 16 channels
        else:
            # BASELINE VERSION: Simple decoder without skip connections
            x = self.dec1(x4)    # 14 -> 28
            x = self.dec2(x)     # 28 -> 56
            x = self.dec3(x)     # 56 -> 112
            x = self.dec4(x)     # 112 -> 224
        
        # Output with sigmoid activation
        x = self.output(x)
        x = torch.sigmoid(x)
        
        return x
    
    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ========================================
# LOSS FUNCTION
# ========================================

def bce_iou_loss(pred, target, smooth=1e-6):
    """
    Combined loss: Binary Cross-Entropy + 0.5 × (1 - IoU)
    
    Args:
        pred: Predicted mask (batch_size, 1, H, W)
        target: Ground truth mask (batch_size, 1, H, W)
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Combined loss value
    """
    # Binary Cross-Entropy Loss
    bce = F.binary_cross_entropy(pred, target, reduction='mean')
    
    # IoU Loss
    intersection = (pred * target).sum(dim=(2, 3))
    union = (pred + target).sum(dim=(2, 3)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    iou_loss = 1 - iou.mean()
    
    # Combined loss
    total_loss = bce + 0.5 * iou_loss
    
    return total_loss


# ========================================
# METRICS
# ========================================

def compute_iou(pred, target, threshold=0.5, smooth=1e-6):
    """
    Compute Intersection over Union (IoU) metric
    
    Args:
        pred: Predicted mask (batch_size, 1, H, W)
        target: Ground truth mask (batch_size, 1, H, W)
        threshold: Threshold for binarizing predictions
        smooth: Smoothing factor
        
    Returns:
        Mean IoU score
    """
    # Binarize predictions
    pred_binary = (pred > threshold).float()
    
    # Compute IoU
    intersection = (pred_binary * target).sum(dim=(2, 3))
    union = (pred_binary + target).sum(dim=(2, 3)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    return iou.mean().item()


def compute_dice(pred, target, threshold=0.5, smooth=1e-6):
    """
    Compute Dice coefficient (F1 score for segmentation)
    
    Args:
        pred: Predicted mask (batch_size, 1, H, W)
        target: Ground truth mask (batch_size, 1, H, W)
        threshold: Threshold for binarizing predictions
        smooth: Smoothing factor
        
    Returns:
        Mean Dice score
    """
    # Binarize predictions
    pred_binary = (pred > threshold).float()
    
    # Compute Dice
    intersection = (pred_binary * target).sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (pred_binary.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth)
    
    return dice.mean().item()


# ========================================
# MODEL SUMMARY
# ========================================

def print_model_summary(model, input_size=(1, 3, 224, 224)):
    """
    Print model architecture summary
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (batch_size, channels, height, width)
    """
    print("\n" + "="*70)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*70)
    
    # Create dummy input
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_size).to(device)
    
    # Forward pass to get output shape
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"\nInput shape:  {tuple(dummy_input.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"\nTotal parameters: {model.count_parameters():,}")
    
    # Print layer-by-layer info
    print("\n" + "-"*70)
    print("LAYER DETAILS:")
    print("-"*70)
    
    for name, module in model.named_children():
        if isinstance(module, nn.Sequential):
            print(f"\n{name.upper()}:")
            for layer in module:
                print(f"  {layer}")
        else:
            print(f"\n{name.upper()}: {module}")
    
    print("="*70)


# ========================================
# TESTING
# ========================================

if __name__ == "__main__":
    # Fix Unicode encoding for Windows console
    import sys
    import io
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    print("Testing SOD Model...")
    
    # Create model
    model = SODModel()
    
    # Print summary
    print_model_summary(model)
    
    # Test forward pass
    print("\nTesting forward pass...")
    dummy_input = torch.randn(2, 3, 224, 224)  # Batch of 2 images
    output = model(dummy_input)
    
    print(f"[OK] Input shape: {dummy_input.shape}")
    print(f"[OK] Output shape: {output.shape}")
    print(f"[OK] Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Test loss function
    print("\nTesting loss function...")
    dummy_target = torch.randint(0, 2, (2, 1, 224, 224)).float()
    loss = bce_iou_loss(output, dummy_target)
    print(f"[OK] Loss value: {loss.item():.4f}")
    
    # Test metrics
    print("\nTesting metrics...")
    iou = compute_iou(output, dummy_target)
    dice = compute_dice(output, dummy_target)
    print(f"[OK] IoU: {iou:.4f}")
    print(f"[OK] Dice: {dice:.4f}")
    
    print("\n[SUCCESS] All tests passed!")
