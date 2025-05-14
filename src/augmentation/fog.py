"""
fog.py
--------
Applies synthetic fog effect to images using OpenCV.

Usage:
    python fog.py --input path/to/image.jpg --output path/to/output.jpg --intensity 0.6

Dependencies:
    - OpenCV (cv2)
    - NumPy
"""

import cv2
import numpy as np
import argparse
import os

def add_fog(image, intensity=0.5):
    """
    Applies a fog effect to an image.

    Args:
        image (np.array): Input image in BGR format.
        intensity (float): Fog intensity, between 0 (none) and 1 (dense fog).

    Returns:
        np.array: Fog-augmented image.
    """
    if intensity < 0 or intensity > 1:
        raise ValueError("Intensity must be between 0 and 1")

    h, w = image.shape[:2]
    
    # Create a white haze
    haze = np.full((h, w, 3), 255, dtype=np.uint8)
    
    # Blend the image with the white haze
    foggy_image = cv2.addWeighted(image, 1 - intensity, haze, intensity, 0)
    
    # Optional: Add Gaussian blur to simulate scattering
    blur_amount = int(15 * intensity) | 1  # Ensure it's odd
    foggy_image = cv2.GaussianBlur(foggy_image, (blur_amount, blur_amount), 0)
    
    return foggy_image

def main():
    parser = argparse.ArgumentParser(description="Apply fog effect to an image.")
    parser.add_argument('--input', type=str, required=True, help="Path to input image")
    parser.add_argument('--output', type=str, required=True, help="Path to save augmented image")
    parser.add_argument('--intensity', type=float, default=0.5, help="Fog intensity (0 to 1)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file {args.input} not found.")

    image = cv2.imread(args.input)
    foggy = add_fog(image, intensity=args.intensity)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cv2.imwrite(args.output, foggy)
    print(f"Saved foggy image to {args.output}")

if __name__ == "__main__":
    main()
