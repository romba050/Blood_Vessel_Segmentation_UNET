"""
Comparison utilities for blood vessel segmentation evaluation.

Based on segRetino by Srijarko Roy (https://github.com/srijarkoroy/segRetino)
Licensed under MIT License.
Enhanced with comparison metrics and visualization tools.
"""

import numpy as np
from PIL import Image
import cv2
from typing import Tuple, Dict

def load_and_preprocess_mask(image_path: str) -> np.ndarray:
    """Load image and convert to binary mask (0 or 1)"""
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img_array = np.array(img)
    # Threshold to binary (assuming vessels are bright)
    binary_mask = (img_array > 127).astype(np.uint8)
    return binary_mask

def calculate_segmentation_metrics(predicted: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
    """Calculate traditional segmentation metrics"""
    # Flatten arrays for easier calculation
    pred_flat = predicted.flatten()
    gt_flat = ground_truth.flatten()
    
    # Calculate confusion matrix components
    tp = np.sum((pred_flat == 1) & (gt_flat == 1))  # True Positives
    tn = np.sum((pred_flat == 0) & (gt_flat == 0))  # True Negatives
    fp = np.sum((pred_flat == 1) & (gt_flat == 0))  # False Positives
    fn = np.sum((pred_flat == 0) & (gt_flat == 1))  # False Negatives
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    # F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Dice Coefficient (same as F1 for binary)
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    
    # IoU (Jaccard Index)
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1_score,
        'dice_coefficient': dice,
        'iou': iou,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }

def create_comparison_visualization(predicted: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    """Create a color-coded comparison image showing agreement/disagreement"""
    h, w = predicted.shape
    comparison = np.zeros((h, w, 3), dtype=np.uint8)
    
    # True Positives (both predict vessel) - Green
    tp_mask = (predicted == 1) & (ground_truth == 1)
    comparison[tp_mask] = [0, 255, 0]  # Green
    
    # True Negatives (both predict background) - Black (already initialized)
    
    # False Positives (predicted vessel, but ground truth is background) - Red
    fp_mask = (predicted == 1) & (ground_truth == 0)
    comparison[fp_mask] = [255, 0, 0]  # Red
    
    # False Negatives (predicted background, but ground truth is vessel) - Blue
    fn_mask = (predicted == 0) & (ground_truth == 1)
    comparison[fn_mask] = [0, 0, 255]  # Blue
    
    return comparison

def get_manual_annotation_path(image_number: str) -> str:
    """Get the path to the corresponding manual annotation"""
    return f'./segRetino/DRIVE_augmented/test/1st_manual/{image_number}_test_0.png'