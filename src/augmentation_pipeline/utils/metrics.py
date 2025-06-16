import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

def compute_ssim(img1, img2):
    """
    Compute SSIM between two images.
    """
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return ssim(img1_gray, img2_gray)

def compute_psnr(img1, img2):
    """
    Compute PSNR between two images.
    """
    return cv2.PSNR(img1, img2)
