import os
import cv2
from utils.metrics import compute_ssim, compute_psnr

def evaluate_augmentations(original_dir, augmented_dir):
    """
    Compute average SSIM and PSNR between original and augmented image folders.
    """
    ssim_scores = []
    psnr_scores = []

    for filename in os.listdir(original_dir):
        orig_path = os.path.join(original_dir, filename)
        aug_path = os.path.join(augmented_dir, f"fogged_{filename}")
        if not os.path.exists(aug_path):
            continue

        orig = cv2.imread(orig_path)
        aug = cv2.imread(aug_path)

        ssim_scores.append(compute_ssim(orig, aug))
        psnr_scores.append(compute_psnr(orig, aug))

    return {
        "avg_ssim": sum(ssim_scores) / len(ssim_scores) if ssim_scores else 0,
        "avg_psnr": sum(psnr_scores) / len(psnr_scores) if psnr_scores else 0
    }
