import numpy as np
import cv2

def simulate_night(image, gamma=1.8):
    """
    Simulate nighttime conditions using gamma correction.
    """
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
    night_image = cv2.LUT(image, table)
    return night_image
