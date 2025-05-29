import os
import cv2
import numpy as np
from skimage.measure import shannon_entropy
from utils.metrics import compute_ssim, compute_psnr

def evaluate_augmentations(original_dir, augmented_dir):
    """
    Compute average SSIM, PSNR, entropy, brightness mean and std between
    original and augmented image folders.
    """
    ssim_scores = []
    psnr_scores = []
    entropy_scores = []
    brightness_means = []
    brightness_stds = []

    for filename in os.listdir(original_dir):
        orig_path = os.path.join(original_dir, filename)
        aug_path = os.path.join(augmented_dir, filename)

        if not os.path.exists(aug_path):
            continue

        orig = cv2.imread(orig_path)
        aug = cv2.imread(aug_path)

        # Skip non-readable or dimension-mismatched images
        if orig is None or aug is None or orig.shape != aug.shape:
            continue

        # Metric computations
        ssim_scores.append(compute_ssim(orig, aug))
        psnr_scores.append(compute_psnr(orig, aug))

        # Convert to grayscale for entropy
        gray_aug = cv2.cvtColor(aug, cv2.COLOR_BGR2GRAY)
        entropy_scores.append(shannon_entropy(gray_aug))

        # Brightness from V channel in HSV
        hsv = cv2.cvtColor(aug, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        brightness_means.append(np.mean(v_channel))
        brightness_stds.append(np.std(v_channel))

    return {
        "avg_ssim": sum(ssim_scores) / len(ssim_scores) if ssim_scores else 0,
        "avg_psnr": sum(psnr_scores) / len(psnr_scores) if psnr_scores else 0,
        "avg_entropy": sum(entropy_scores) / len(entropy_scores) if entropy_scores else 0,
        "avg_brightness_mean": sum(brightness_means) / len(brightness_means) if brightness_means else 0,
        "avg_brightness_std": sum(brightness_stds) / len(brightness_stds) if brightness_stds else 0
    }
