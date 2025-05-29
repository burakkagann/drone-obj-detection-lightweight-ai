import sys
import os
import yaml
import cv2

# Setup project path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fog import add_fog
from night import simulate_night
from sensor_distortions import add_sensor_effects
from utils.visualization import plot_histograms
from utils.metrics import compute_ssim, compute_psnr
from evaluation.compare_metrics import evaluate_augmentations

# Directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "visdrone")
AUG_DIR = os.path.join(BASE_DIR, "data", "augmented")
CONFIG_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "config", "augmentation_config.yaml"))

# Load config
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def process_image(filename):
    print(f"\n🖼️ Processing {filename}")
    path = os.path.join(RAW_DIR, filename)
    image = cv2.imread(path)

    fogged = add_fog(image, config["fog"])
    night = simulate_night(image, config["night"])
    distorted = add_sensor_effects(image, config["sensor"])

    ensure_dir(AUG_DIR)
    cv2.imwrite(os.path.join(AUG_DIR, f"fogged_{filename}"), fogged)
    cv2.imwrite(os.path.join(AUG_DIR, f"night_{filename}"), night)
    cv2.imwrite(os.path.join(AUG_DIR, f"distorted_{filename}"), distorted)

    # Evaluate each
    print(" - Fog SSIM:", compute_ssim(image, fogged), "PSNR:", compute_psnr(image, fogged))
    print(" - Night SSIM:", compute_ssim(image, night), "PSNR:", compute_psnr(image, night))
    print(" - Distort SSIM:", compute_ssim(image, distorted), "PSNR:", compute_psnr(image, distorted))

    # Visual Check
    plot_histograms(image, fogged)
    plot_histograms(image, night)
    plot_histograms(image, distorted)

def batch_test():
    print("🚀 Starting batch test on folder:", RAW_DIR)
    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".jpg")]
    for f in files:
        process_image(f)

    # Folder-level evaluation (if originals + augmented have same names)
    print("\n📊 Folder-level SSIM/PSNR comparison:")
    result = evaluate_augmentations(RAW_DIR, AUG_DIR)
    print(result)

if __name__ == "__main__":
    batch_test()
