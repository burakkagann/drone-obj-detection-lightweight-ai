import sys
import os
import shutil
import yaml
import cv2
import csv
from datetime import datetime
import numpy as np
from skimage.measure import shannon_entropy

# Setup paths
SCRIPT_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))
sys.path.append(BASE_DIR)

# 🔥 Clean up all __pycache__ folders from known locations
def remove_pycache():
    print("🚹 Cleaning up __pycache__ folders...")
    targets = [
        os.path.join(BASE_DIR, "augment_scripts"),
        os.path.join(BASE_DIR, "evaluation"),
        os.path.join(BASE_DIR, "utils")
    ]
    for root_dir in targets:
        for root, dirs, files in os.walk(root_dir):
            for d in dirs:
                if d == "__pycache__":
                    pycache_path = os.path.join(root, d)
                    try:
                        shutil.rmtree(pycache_path)
                        print(f"  🔥 Removed: {pycache_path}")
                    except PermissionError:
                        print(f"  ⚠️ Skipped (in use): {pycache_path}")
                    except Exception as e:
                        print(f"  ⚠️ Error removing {pycache_path}: {e}")

remove_pycache()

# Local imports
from fog import add_fog
from night import simulate_night
from sensor_distortions import add_sensor_effects
from utils.visualization import save_histogram
from utils.metrics import compute_ssim, compute_psnr
from evaluation.compare_metrics import evaluate_augmentations

# Directories
RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "visdrone")
AUG_DIR_FOG = os.path.join(BASE_DIR, "data", "augmented", "fog")
AUG_DIR_NIGHT = os.path.join(BASE_DIR, "data", "augmented", "night")
AUG_DIR_SENSOR = os.path.join(BASE_DIR, "data", "augmented", "sensor_distortions")
HISTOGRAM_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "augmentation_histograms")
CSV_LOG_PATH = os.path.join(HISTOGRAM_DIR, "augmentation_eval_log.csv")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "augmentation_config.yaml")

# Load config
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def calc_entropy(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(shannon_entropy(gray))

def brightness_stats(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = hsv[..., 2]
    return float(np.mean(brightness)), float(np.std(brightness))

def process_image(filename):
    print(f"\n🖼️ Processing {filename}")
    path = os.path.join(RAW_DIR, filename)
    image = cv2.imread(path)

    fogged = add_fog(image, config["fog"])
    night = simulate_night(image, config["night"])
    distorted = add_sensor_effects(image, config["sensor"])

    ensure_dir(AUG_DIR_FOG)
    ensure_dir(AUG_DIR_NIGHT)
    ensure_dir(AUG_DIR_SENSOR)
    ensure_dir(HISTOGRAM_DIR)

    cv2.imwrite(os.path.join(AUG_DIR_FOG, filename), fogged)
    cv2.imwrite(os.path.join(AUG_DIR_NIGHT, filename), night)
    cv2.imwrite(os.path.join(AUG_DIR_SENSOR, filename), distorted)

    print(" - Fog SSIM:", compute_ssim(image, fogged), "PSNR:", compute_psnr(image, fogged))
    print(" - Night SSIM:", compute_ssim(image, night), "PSNR:", compute_psnr(image, night))
    print(" - Distort SSIM:", compute_ssim(image, distorted), "PSNR:", compute_psnr(image, distorted))

    save_histogram(image, fogged, filename, "fog", HISTOGRAM_DIR)
    save_histogram(image, night, filename, "night", HISTOGRAM_DIR)
    save_histogram(image, distorted, filename, "sensor", HISTOGRAM_DIR)

def log_results_to_csv(fog_result, night_result, sensor_result):
    row = {
        "timestamp": datetime.now().isoformat(timespec='seconds'),
        "fog_ssim": float(fog_result["avg_ssim"]),
        "fog_psnr": float(fog_result["avg_psnr"]),
        "night_ssim": float(night_result["avg_ssim"]),
        "night_psnr": float(night_result["avg_psnr"]),
        "sensor_ssim": float(sensor_result["avg_ssim"]),
        "sensor_psnr": float(sensor_result["avg_psnr"]),
        "fog_entropy": fog_result.get("avg_entropy", 0),
        "night_entropy": night_result.get("avg_entropy", 0),
        "sensor_entropy": sensor_result.get("avg_entropy", 0),
        "fog_brightness_mean": fog_result.get("avg_brightness_mean", 0),
        "fog_brightness_std": fog_result.get("avg_brightness_std", 0),
        "night_brightness_mean": night_result.get("avg_brightness_mean", 0),
        "night_brightness_std": night_result.get("avg_brightness_std", 0),
        "sensor_brightness_mean": sensor_result.get("avg_brightness_mean", 0),
        "sensor_brightness_std": sensor_result.get("avg_brightness_std", 0),
        "fog_density": config["fog"].get("density", ""),
        "night_darkness": config["night"].get("darkness", ""),
        "sensor_blur_strength": config["sensor"].get("blur_strength", "")
    }
    ensure_dir(os.path.dirname(CSV_LOG_PATH))
    file_exists = os.path.isfile(CSV_LOG_PATH)
    with open(CSV_LOG_PATH, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def batch_test():
    print("🚀 Starting batch test on folder:", RAW_DIR)
    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".jpg")]
    for f in files:
        process_image(f)

    print("\n📊 Folder-level SSIM/PSNR comparison:")
    fog_result = evaluate_augmentations(RAW_DIR, AUG_DIR_FOG)
    night_result = evaluate_augmentations(RAW_DIR, AUG_DIR_NIGHT)
    sensor_result = evaluate_augmentations(RAW_DIR, AUG_DIR_SENSOR)

    print("Fog:", fog_result)
    print("Night:", night_result)
    print("Sensor Distortion:", sensor_result)

    log_results_to_csv(fog_result, night_result, sensor_result)

if __name__ == "__main__":
    batch_test()
