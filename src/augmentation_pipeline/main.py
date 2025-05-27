import cv2
from augment.fog import add_fog
from augment.night import simulate_night
from augment.sensor_distortions import add_sensor_effects
from utils.visualization import plot_histograms
from utils.metrics import compute_ssim, compute_psnr

def run_pipeline(image_path):
    original = cv2.imread(image_path)

    fogged = add_fog(original)
    night = simulate_night(original)
    distorted = add_sensor_effects(original)

    for aug, name in zip([fogged, night, distorted], ["Fog", "Night", "Sensor"]):
        print(f"{name} SSIM:", compute_ssim(original, aug))
        print(f"{name} PSNR:", compute_psnr(original, aug))
        plot_histograms(original, aug)

if __name__ == "__main__":
    run_pipeline("data/raw/sample.jpg")  # Replace with your test image path
