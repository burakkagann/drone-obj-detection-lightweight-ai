import numpy as np
import cv2

def add_sensor_effects(image):
    """
    Simulate sensor distortions: motion blur, Gaussian noise, chromatic aberration.
    """
    # Motion blur
    ksize = 9
    kernel = np.zeros((ksize, ksize))
    kernel[int((ksize-1)/2), :] = np.ones(ksize)
    kernel /= ksize
    blurred = cv2.filter2D(image, -1, kernel)

    # Gaussian noise
    noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
    noisy = cv2.add(blurred, noise)

    # Chromatic aberration
    B, G, R = cv2.split(noisy)
    R = np.roll(R, 1, axis=1)
    B = np.roll(B, -1, axis=0)
    distorted = cv2.merge([B, G, R])
    
    return distorted
