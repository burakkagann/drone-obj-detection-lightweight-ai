import numpy as np
import cv2

def add_sensor_effects(image, config):
    blur_ksize = config.get("blur_ksize", 3)
    noise_std = config.get("noise_std", 5)
    shift_pixels = config.get("shift_pixels", 1)

    # Motion blur
    kernel = np.zeros((blur_ksize, blur_ksize))
    kernel[int((blur_ksize - 1) / 2), :] = np.ones(blur_ksize)
    kernel /= blur_ksize
    blurred = cv2.filter2D(image, -1, kernel)

    # Gaussian noise
    noise = np.random.normal(0, noise_std, image.shape).astype(np.int16)
    noisy = np.clip(blurred.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Chromatic aberration
    B, G, R = cv2.split(noisy)
    R = np.roll(R, shift_pixels, axis=1)
    B = np.roll(B, -shift_pixels, axis=0)
    distorted = cv2.merge([B, G, R])

    return distorted
