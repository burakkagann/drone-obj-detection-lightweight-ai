import numpy as np
import cv2

def add_fog(image, config):
    fog_intensity = config.get("intensity", 0.5)
    depth_blend = config.get("depth_blend", 0.3)
    
    h, w = image.shape[:2]
    fog_layer = np.full_like(image, 255, dtype=np.uint8)
    depth = np.tile(np.linspace(0, 1, h).reshape(h, 1), (1, w))
    alpha = fog_intensity * np.exp(-depth_blend * depth)
    alpha = alpha[..., np.newaxis]
    
    fogged = cv2.convertScaleAbs(image * (1 - alpha) + fog_layer * alpha)
    return fogged
