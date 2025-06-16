import numpy as np
import cv2

def simulate_night(image, config):
    gamma = config.get("gamma", 2.5)
    brightness_reduction = config.get("brightness_reduction", 0.6)
    desaturate = config.get("desaturate", True)

    # Gamma correction
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
    dark = cv2.LUT(image, table)

    # Brightness reduction
    dark = cv2.convertScaleAbs(dark, alpha=brightness_reduction, beta=0)

    # Desaturate
    if desaturate:
        hsv = cv2.cvtColor(dark, cv2.COLOR_BGR2HSV)
        hsv[..., 1] = hsv[..., 1] * 0.4
        dark = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return dark
