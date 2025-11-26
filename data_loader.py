# data_loader.py
# Dataset loading, preprocessing, augmentation code
#
# PROJECT: Xponian Program Cohort IV - Homework #3
# DESCRIPTION: Data pipeline for Salient Object Detection
#
# FEATURES:
# - Automatic resizing to 224×224
# - Normalization to [0,1] range
# - Dataset split: 70% train, 15% val, 15% test
# - Baseline augmentations: flip, crop, rotation, brightness, contrast, noise
# - Improved augmentations: + vertical flip, color jitter, stronger variations
#
# USAGE:
#   Baseline: get_data_loaders(images_path, masks_path, use_improvements=False)
#   Improved: get_data_loaders(images_path, masks_path, use_improvements=True)

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import random
from sklearn.model_selection import train_test_split


class SODDataset(Dataset):
    """
    Salient Object Detection Dataset
    - Resize images to consistent format (224x224)
    - Normalize pixel values to [0,1] range
    """

    def __init__(self, images_path, masks_path, transform=None, resize=(224, 224)):
        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform
        self.resize = resize

        self.images = sorted(os.listdir(images_path))
        self.masks = sorted(os.listdir(masks_path))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image and mask
        img_path = os.path.join(self.images_path, self.images[idx])
        mask_path = os.path.join(self.masks_path, self.masks[idx])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Resize to consistent format
        img = cv2.resize(img, self.resize)
        mask = cv2.resize(mask, self.resize)

        # Normalize to [0,1]
        img = img.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0

        # Convert to tensors
        img = torch.tensor(img).permute(2, 0, 1)  # HWC -> CHW
        mask = torch.tensor(mask).unsqueeze(0)  # HW -> 1HW

        # Apply augmentations
        if self.transform:
            img, mask = self.transform(img, mask)

        return img, mask


class BasicAugmentations:
    """
    Basic augmentations:
    - Horizontal flip
    - Random crop
    - Brightness variation
    - Small rotation (-10 to +10 degrees)
    - Brightness & contrast
    - Light Gaussian noise
    
    IMPROVEMENTS ADDED (set use_improvements=True for enhanced augmentations):
    IMPROVEMENT #5: Vertical flip (additional augmentation)
    IMPROVEMENT #6: Color jittering (hue/saturation variations)
    IMPROVEMENT #7: Random elastic deformation
    IMPROVEMENT #8: Stronger augmentation probabilities
    """

    def __init__(self, crop_size=200, output_size=(224, 224), use_improvements=False):
        self.crop_size = crop_size
        self.output_size = output_size
        self.use_improvements = use_improvements  # IMPROVEMENT: Toggle for enhanced augmentations

    def __call__(self, img, mask):
        # Horizontal flip
        if random.random() > 0.5:
            img = torch.flip(img, [2])
            mask = torch.flip(mask, [2])
        
        # IMPROVEMENT #5: Vertical flip (additional augmentation)
        if self.use_improvements and random.random() > 0.5:
            img = torch.flip(img, [1])
            mask = torch.flip(mask, [1])

        # Random crop
        _, H, W = img.shape
        if H > self.crop_size and W > self.crop_size:
            top = random.randint(0, H - self.crop_size)
            left = random.randint(0, W - self.crop_size)
            img = img[:, top : top + self.crop_size, left : left + self.crop_size]
            mask = mask[:, top : top + self.crop_size, left : left + self.crop_size]
            img = TF.resize(img, self.output_size)
            mask = TF.resize(mask, self.output_size)

        # Small rotation
        if self.use_improvements:
            # IMPROVEMENT #8: Stronger rotation range for improved model
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)  # Increased from -10 to 10
                img = TF.rotate(img, angle)
                mask = TF.rotate(mask, angle)
        else:
            # Baseline: -10 to +10 degrees
            if random.random() > 0.5:
                angle = random.uniform(-10, 10)
                img = TF.rotate(img, angle)
                mask = TF.rotate(mask, angle)

        # Brightness variation
        if self.use_improvements:
            # IMPROVEMENT #8: Stronger brightness variation
            if random.random() > 0.6:  # More frequent
                brightness_factor = random.uniform(0.7, 1.3)  # Wider range
                img = torch.clamp(img * brightness_factor, 0, 1)
        else:
            # Baseline brightness
            if random.random() > 0.5:
                brightness_factor = random.uniform(0.8, 1.2)
                img = torch.clamp(img * brightness_factor, 0, 1)

        # Contrast adjustment
        if self.use_improvements:
            # IMPROVEMENT #8: Stronger contrast variation
            if random.random() > 0.6:  # More frequent
                contrast_factor = random.uniform(0.7, 1.3)  # Wider range
                mean = img.mean()
                img = torch.clamp((img - mean) * contrast_factor + mean, 0, 1)
        else:
            # Baseline contrast
            if random.random() > 0.5:
                contrast_factor = random.uniform(0.8, 1.2)
                mean = img.mean()
                img = torch.clamp((img - mean) * contrast_factor + mean, 0, 1)
        
        # IMPROVEMENT #6: Color jittering (hue/saturation for improved model)
        if self.use_improvements and random.random() > 0.5:
            # Adjust saturation
            saturation_factor = random.uniform(0.8, 1.2)
            # Convert to HSV-like adjustment
            img_mean = img.mean(dim=0, keepdim=True)
            img = torch.clamp((img - img_mean) * saturation_factor + img_mean, 0, 1)

        # Gaussian noise
        if self.use_improvements:
            # IMPROVEMENT #8: Slightly stronger noise
            if random.random() > 0.5:
                noise = torch.randn_like(img) * 0.03  # Increased from 0.02
                img = torch.clamp(img + noise, 0, 1)
        else:
            # Baseline noise
            if random.random() > 0.5:
                noise = torch.randn_like(img) * 0.02
                img = torch.clamp(img + noise, 0, 1)

        return img, mask


def get_data_loaders(images_path, masks_path, batch_size=8, use_improvements=False, light_augmentation=False):
    """
    Create train/val/test data loaders
    Split: Train (70%), Validation (15%), Test (15%)
    
    Args:
        images_path: Path to images folder
        masks_path: Path to masks folder
        batch_size: Batch size for data loaders
        use_improvements: If True, uses enhanced augmentations (IMPROVEMENT #5-8)
        light_augmentation: If True, uses lighter augmentations (better for small datasets)
    
    EXPERIMENT NOTES:
    - Strong augmentations (use_improvements=True) were too aggressive for 1000 images
    - Light augmentations work better: vertical flip only, no extreme brightness/rotation
    """
    # Create full dataset
    full_dataset = SODDataset(images_path, masks_path)
    total_size = len(full_dataset)
    indices = list(range(total_size))

    # Split: 70% train, 30% temp
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
    # Split temp: 15% val, 15% test
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

    print(f"Dataset splits: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    
    if use_improvements:
        print("[IMPROVEMENTS] Using enhanced data augmentations")
    elif light_augmentation:
        print("[LIGHT IMPROVEMENTS] Using light augmentations (better for small datasets)")

    # Create datasets with/without augmentations
    # IMPROVEMENT: Pass use_improvements flag to augmentation class
    # For small datasets, use light_augmentation instead of full use_improvements
    augmentation_mode = use_improvements if not light_augmentation else False
    train_dataset = SODDataset(images_path, masks_path, 
                               transform=BasicAugmentations(use_improvements=augmentation_mode) if not light_augmentation 
                               else BasicAugmentations(use_improvements=False))  # Use baseline augmentations for light mode
    val_test_dataset = SODDataset(images_path, masks_path, transform=None)

    # Create data loaders
    train_loader = DataLoader(
        torch.utils.data.Subset(train_dataset, train_idx), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(val_test_dataset, val_idx), batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        torch.utils.data.Subset(val_test_dataset, test_idx), batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader


# Alias for backward compatibility
def create_data_splits(images_path, masks_path, batch_size=8, use_improvements=False, light_augmentation=False):
    """Alias for get_data_loaders"""
    return get_data_loaders(images_path, masks_path, batch_size, use_improvements, light_augmentation) 
