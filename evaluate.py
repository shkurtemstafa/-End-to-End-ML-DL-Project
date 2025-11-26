# evaluate.py
# Evaluation metrics and visualization
#
# PROJECT: Xponian Program Cohort IV - Homework #3
# DESCRIPTION: Complete evaluation pipeline with metrics and visualizations
#
# FEATURES:
# - IoU (Intersection over Union)
# - Precision, Recall, F1-Score
# - Mean Absolute Error (MAE)
# - Confusion Matrix
# - Sample predictions visualization
# - Overlay visualization
# - Automatic comparison table (baseline vs improved)
#
# USAGE:
#   Set EVAL_BASELINE and EVAL_IMPROVED flags (lines 217-218)
#   Run: python evaluate.py
#
# OUTPUT:
#   - evaluation_results.png (sample predictions)
#   - confusion_matrix_baseline.png
#   - confusion_matrix_improved.png
#   - Comparison table with all metrics

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from sod_model import SODModel, compute_iou
from data_loader import get_data_loaders


def compute_metrics(model, test_loader, device='cpu', threshold=0.5):
    """
    Compute evaluation metrics:
    - Test Loss (BCE + IoU Loss)
    - IoU (Intersection over Union)
    - Precision
    - Recall
    - F1-Score
    - Mean Absolute Error (optional)
    """
    from sod_model import bce_iou_loss  # Import loss function
    
    model.eval()
    model.to(device)

    all_ious = []
    all_preds = []
    all_targets = []
    all_maes = []
    all_losses = []

    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            # Predict
            preds = model(imgs)

            # Compute loss
            loss = bce_iou_loss(preds, masks)
            all_losses.append(loss.item())

            # Binarize predictions
            preds_binary = (preds > threshold).float()

            # Compute IoU per batch
            iou = compute_iou(preds, masks, threshold=threshold)
            all_ious.append(iou)

            # Flatten for sklearn metrics - ensure binary values
            preds_flat = preds_binary.cpu().numpy().flatten().astype(int)  # Convert to int (0 or 1)
            masks_flat = masks.cpu().numpy().flatten().astype(int)  # Convert to int (0 or 1)

            all_preds.extend(preds_flat)
            all_targets.extend(masks_flat)

            # Mean Absolute Error
            mae = torch.abs(preds - masks).mean().item()
            all_maes.append(mae)

    # Compute metrics
    mean_loss = np.mean(all_losses)
    mean_iou = np.mean(all_ious)
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    mean_mae = np.mean(all_maes)

    return {
        "Loss": mean_loss,
        "IoU": mean_iou,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "MAE": mean_mae,
        "predictions": all_preds,
        "targets": all_targets,
    }


def visualize_predictions(model, test_loader, device='cpu', num_samples=5):
    """
    Generate sample visualizations:
    - Input image
    - Ground-truth mask
    - Predicted mask
    - Overlay (predicted + input)
    """
    model.eval()
    model.to(device)

    # Get one batch
    imgs, masks = next(iter(test_loader))
    imgs = imgs.to(device)
    masks = masks.to(device)

    # Predict
    with torch.no_grad():
        preds = model(imgs)

    # Move to CPU for visualization
    imgs = imgs.cpu()
    masks = masks.cpu()
    preds = preds.cpu()

    # Plot
    num_samples = min(num_samples, len(imgs))
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, num_samples * 3))

    for i in range(num_samples):
        # Input image
        img_np = imgs[i].permute(1, 2, 0).numpy()
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title("Input Image")
        axes[i, 0].axis("off")

        # Ground truth
        mask_np = masks[i].squeeze().numpy()
        axes[i, 1].imshow(mask_np, cmap="gray")
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")

        # Prediction
        pred_np = preds[i].squeeze().numpy()
        axes[i, 2].imshow(pred_np, cmap="gray")
        axes[i, 2].set_title("Predicted Mask")
        axes[i, 2].axis("off")

        # Overlay
        overlay = img_np.copy()
        mask_colored = np.zeros_like(overlay)
        mask_colored[:, :, 0] = pred_np  # Red channel
        overlay = np.clip(overlay * 0.7 + mask_colored * 0.3, 0, 1)
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title("Overlay")
        axes[i, 3].axis("off")

    plt.tight_layout()
    plt.savefig("evaluation_results.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("Visualization saved as 'evaluation_results.png'")


def plot_confusion_matrix(predictions, targets, save_path="confusion_matrix.png"):
    """
    Plot confusion matrix for binary segmentation
    
    Args:
        predictions: Flattened binary predictions
        targets: Flattened binary ground truth
        save_path: Path to save the plot
    """
    # Compute confusion matrix
    cm = confusion_matrix(targets, predictions)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=['Background', 'Salient'], 
                yticklabels=['Background', 'Salient'])
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Ground Truth', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Add percentages
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            percentage = (cm[i, j] / total) * 100
            ax.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                   ha='center', va='center', fontsize=10, color='gray')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Confusion matrix saved as '{save_path}'")
    
    # Print confusion matrix values
    print("\nConfusion Matrix:")
    print(f"  True Negatives (TN):  {cm[0,0]:,}")
    print(f"  False Positives (FP): {cm[0,1]:,}")
    print(f"  False Negatives (FN): {cm[1,0]:,}")
    print(f"  True Positives (TP):  {cm[1,1]:,}")
    
    return cm


def evaluate_model(model_path, images_path, masks_path, batch_size=8, device='cpu', 
                  use_improvements=False, model_name="baseline"):
    """
    Full evaluation pipeline
    
    Args:
        model_path: Path to saved model checkpoint
        images_path: Path to images folder
        masks_path: Path to masks folder
        batch_size: Batch size for evaluation
        device: 'cpu' or 'cuda'
        use_improvements: Whether model uses improvements
        model_name: Name for saving outputs
    """
    print("=" * 70)
    print(f"MODEL EVALUATION - {model_name.upper()}")
    print("=" * 70)

    # Load data
    print("\n[1/3] Loading test data...")
    _, _, test_loader = get_data_loaders(images_path, masks_path, batch_size=batch_size, 
                                         use_improvements=use_improvements)
    print(f"Test set size: {len(test_loader.dataset)}")

    # Load model
    print("\n[2/3] Loading model...")
    model = SODModel(use_improvements=use_improvements)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    print(f"Model loaded from {model_path}")
    print(f"Model parameters: {model.count_parameters():,}")

    # Compute metrics
    print("\n[3/3] Computing metrics...")
    metrics = compute_metrics(model, test_loader, device=device)

    print("\n" + "=" * 70)
    print(f"EVALUATION RESULTS - {model_name.upper()}")
    print("=" * 70)
    print(f"Test Loss:  {metrics['Loss']:.4f}")
    print(f"IoU:        {metrics['IoU']:.4f}")
    print(f"Precision:  {metrics['Precision']:.4f}")
    print(f"Recall:     {metrics['Recall']:.4f}")
    print(f"F1-Score:   {metrics['F1-Score']:.4f}")
    print(f"MAE:        {metrics['MAE']:.4f}")
    print("=" * 70)

    # Visualize
    print("\nGenerating visualizations...")
    visualize_predictions(model, test_loader, device=device, num_samples=5)
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    cm = plot_confusion_matrix(metrics['predictions'], metrics['targets'], 
                              save_path=f"confusion_matrix_{model_name}.png")

    return metrics, cm


if __name__ == "__main__":
    # Configuration
    IMAGES_PATH = "images/"
    MASKS_PATH = "ground_truth_mask/"
    BATCH_SIZE = 8
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ========================================
    # IMPROVEMENT: Evaluate both baseline and improved models
    # ========================================
    EVAL_BASELINE = True   # Set to True to evaluate baseline model
    EVAL_IMPROVED = True   # Set to True to evaluate improved model
    
    # Model paths
    BASELINE_MODEL = "best_model_baseline.pth"
    IMPROVED_MODEL = "best_model_improved.pth"  # Trained model from Colab
    
    print("="*70)
    print("SOD MODEL EVALUATION - BASELINE vs IMPROVED")
    print("="*70)
    
    results = {}
    
    # Evaluate baseline model
    if EVAL_BASELINE:
        print("\n" + "="*70)
        print("EVALUATING BASELINE MODEL")
        print("="*70)
        
        metrics_baseline, cm_baseline = evaluate_model(
            model_path=BASELINE_MODEL,
            images_path=IMAGES_PATH,
            masks_path=MASKS_PATH,
            batch_size=BATCH_SIZE,
            device=DEVICE,
            use_improvements=False,
            model_name="baseline"
        )
        results['baseline'] = metrics_baseline
    
    # Evaluate improved model
    if EVAL_IMPROVED:
        print("\n" + "="*70)
        print("EVALUATING IMPROVED MODEL")
        print("="*70)
        
        metrics_improved, cm_improved = evaluate_model(
            model_path=IMPROVED_MODEL,
            images_path=IMAGES_PATH,
            masks_path=MASKS_PATH,
            batch_size=BATCH_SIZE,
            device=DEVICE,
            use_improvements=True,
            model_name="improved"
        )
        results['improved'] = metrics_improved
    
    # ========================================
    # COMPARISON TABLE
    # ========================================
    if EVAL_BASELINE and EVAL_IMPROVED:
        print("\n" + "="*70)
        print("BASELINE vs IMPROVED - DETAILED COMPARISON")
        print("="*70)
        print(f"{'Metric':<20} {'Baseline':<15} {'Improved':<15} {'Change':<15}")
        print("-"*70)
        
        for metric_name in ['Loss', 'IoU', 'Precision', 'Recall', 'F1-Score', 'MAE']:
            baseline_val = results['baseline'][metric_name]
            improved_val = results['improved'][metric_name]
            
            if baseline_val != 0:
                change = ((improved_val - baseline_val) / baseline_val) * 100
            else:
                change = 0
            
            print(f"{metric_name:<20} {baseline_val:<15.4f} {improved_val:<15.4f} {change:+.2f}%")
        
        print("="*70)
        
        print("\nIMPROVEMENTS APPLIED (v2 - Optimized for small dataset):")
        print("  1. Batch Normalization - Stabilizes training")
        print("  2. Dropout (0.05) - Minimal regularization for small dataset")
        print("  3. Deeper layers (3 conv per block) - Better feature extraction")
        print("  4. Skip connections (U-Net style) - Preserves spatial information")
        print("  5. Light augmentations - Balanced for 1000 images")
        print("\nKEY INSIGHT:")
        print("  Dataset size (1000 images) requires careful regularization tuning.")
        print("  Heavy dropout (0.3) + strong augmentations caused underfitting.")
        print("  Reduced dropout (0.05) + light augmentations achieved better balance.")

    print("\n[SUCCESS] Evaluation complete!")
