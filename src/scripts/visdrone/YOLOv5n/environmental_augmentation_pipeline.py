#!/usr/bin/env python3
"""
Environmental Augmentation Pipeline for YOLOv5n + VisDrone
Implements comprehensive synthetic data augmentation according to methodology framework.

This pipeline supports:
- Fog/Haze: Light, Medium, Heavy (visibility-based)
- Low Light/Night: Dusk, Urban Night, Minimal Light
- Motion Blur: Light, Medium, Heavy (camera shake simulation)
- Weather: Light Rain, Heavy Rain, Snow

Distribution Strategy:
- Original: 40% of training data
- Light conditions: 20% of training data
- Medium conditions: 25% of training data
- Heavy conditions: 15% of training data
"""

import cv2
import numpy as np
import random
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class AugmentationType(Enum):
    FOG = "fog"
    NIGHT = "night"
    MOTION_BLUR = "motion_blur"
    RAIN = "rain"
    SNOW = "snow"

class IntensityLevel(Enum):
    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"

@dataclass
class AugmentationConfig:
    """Configuration for augmentation parameters"""
    aug_type: AugmentationType
    intensity: IntensityLevel
    parameters: Dict

class EnvironmentalAugmentator:
    """Main class for environmental augmentation pipeline"""
    
    def __init__(self, seed: int = 42):
        """Initialize the augmentator with reproducible random seed"""
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
        # Initialize augmentation configurations
        self.fog_configs = self._init_fog_configs()
        self.night_configs = self._init_night_configs()
        self.motion_blur_configs = self._init_motion_blur_configs()
        self.rain_configs = self._init_rain_configs()
        self.snow_configs = self._init_snow_configs()
        
    def _init_fog_configs(self) -> Dict[IntensityLevel, AugmentationConfig]:
        """Initialize fog augmentation configurations based on visibility ranges"""
        return {
            IntensityLevel.LIGHT: AugmentationConfig(
                aug_type=AugmentationType.FOG,
                intensity=IntensityLevel.LIGHT,
                parameters={
                    "fog_intensity": 0.3,  # Light fog (50-100m visibility)
                    "depth_blend": 0.2,
                    "atmospheric_perspective": 0.1,
                    "color_shift": [10, 10, 15]  # Slight blue-white shift
                }
            ),
            IntensityLevel.MEDIUM: AugmentationConfig(
                aug_type=AugmentationType.FOG,
                intensity=IntensityLevel.MEDIUM,
                parameters={
                    "fog_intensity": 0.5,  # Medium fog (25-50m visibility)
                    "depth_blend": 0.4,
                    "atmospheric_perspective": 0.2,
                    "color_shift": [15, 15, 25]
                }
            ),
            IntensityLevel.HEAVY: AugmentationConfig(
                aug_type=AugmentationType.FOG,
                intensity=IntensityLevel.HEAVY,
                parameters={
                    "fog_intensity": 0.7,  # Heavy fog (10-25m visibility)
                    "depth_blend": 0.6,
                    "atmospheric_perspective": 0.3,
                    "color_shift": [20, 20, 35]
                }
            )
        }
    
    def _init_night_configs(self) -> Dict[IntensityLevel, AugmentationConfig]:
        """Initialize night/low-light augmentation configurations"""
        return {
            IntensityLevel.LIGHT: AugmentationConfig(
                aug_type=AugmentationType.NIGHT,
                intensity=IntensityLevel.LIGHT,
                parameters={
                    "gamma": 1.8,  # Dusk/golden hour
                    "brightness_reduction": 0.7,
                    "desaturate_factor": 0.8,
                    "color_temperature": 3000,  # Warm light
                    "noise_factor": 0.02
                }
            ),
            IntensityLevel.MEDIUM: AugmentationConfig(
                aug_type=AugmentationType.NIGHT,
                intensity=IntensityLevel.MEDIUM,
                parameters={
                    "gamma": 2.2,  # Urban night lighting
                    "brightness_reduction": 0.5,
                    "desaturate_factor": 0.6,
                    "color_temperature": 4000,  # Cool white
                    "noise_factor": 0.05
                }
            ),
            IntensityLevel.HEAVY: AugmentationConfig(
                aug_type=AugmentationType.NIGHT,
                intensity=IntensityLevel.HEAVY,
                parameters={
                    "gamma": 2.8,  # Minimal lighting
                    "brightness_reduction": 0.3,
                    "desaturate_factor": 0.4,
                    "color_temperature": 5000,  # Moonlight
                    "noise_factor": 0.08
                }
            )
        }
    
    def _init_motion_blur_configs(self) -> Dict[IntensityLevel, AugmentationConfig]:
        """Initialize motion blur augmentation configurations"""
        return {
            IntensityLevel.LIGHT: AugmentationConfig(
                aug_type=AugmentationType.MOTION_BLUR,
                intensity=IntensityLevel.LIGHT,
                parameters={
                    "kernel_size": 9,  # Light camera shake
                    "angle_range": (-15, 15),
                    "motion_length": 5,
                    "blur_probability": 0.8
                }
            ),
            IntensityLevel.MEDIUM: AugmentationConfig(
                aug_type=AugmentationType.MOTION_BLUR,
                intensity=IntensityLevel.MEDIUM,
                parameters={
                    "kernel_size": 15,  # Moderate motion
                    "angle_range": (-30, 30),
                    "motion_length": 10,
                    "blur_probability": 0.9
                }
            ),
            IntensityLevel.HEAVY: AugmentationConfig(
                aug_type=AugmentationType.MOTION_BLUR,
                intensity=IntensityLevel.HEAVY,
                parameters={
                    "kernel_size": 21,  # Significant blur
                    "angle_range": (-45, 45),
                    "motion_length": 15,
                    "blur_probability": 1.0
                }
            )
        }
    
    def _init_rain_configs(self) -> Dict[IntensityLevel, AugmentationConfig]:
        """Initialize rain augmentation configurations"""
        return {
            IntensityLevel.LIGHT: AugmentationConfig(
                aug_type=AugmentationType.RAIN,
                intensity=IntensityLevel.LIGHT,
                parameters={
                    "rain_intensity": 0.3,
                    "drop_length": 10,
                    "drop_width": 1,
                    "drop_count": 500,
                    "atmospheric_effect": 0.1
                }
            ),
            IntensityLevel.MEDIUM: AugmentationConfig(
                aug_type=AugmentationType.RAIN,
                intensity=IntensityLevel.MEDIUM,
                parameters={
                    "rain_intensity": 0.5,
                    "drop_length": 15,
                    "drop_width": 2,
                    "drop_count": 1000,
                    "atmospheric_effect": 0.2
                }
            ),
            IntensityLevel.HEAVY: AugmentationConfig(
                aug_type=AugmentationType.RAIN,
                intensity=IntensityLevel.HEAVY,
                parameters={
                    "rain_intensity": 0.7,
                    "drop_length": 20,
                    "drop_width": 3,
                    "drop_count": 1500,
                    "atmospheric_effect": 0.3
                }
            )
        }
    
    def _init_snow_configs(self) -> Dict[IntensityLevel, AugmentationConfig]:
        """Initialize snow augmentation configurations"""
        return {
            IntensityLevel.LIGHT: AugmentationConfig(
                aug_type=AugmentationType.SNOW,
                intensity=IntensityLevel.LIGHT,
                parameters={
                    "snow_intensity": 0.2,
                    "flake_size_range": (1, 3),
                    "flake_count": 300,
                    "atmospheric_effect": 0.1,
                    "brightness_increase": 0.1
                }
            ),
            IntensityLevel.MEDIUM: AugmentationConfig(
                aug_type=AugmentationType.SNOW,
                intensity=IntensityLevel.MEDIUM,
                parameters={
                    "snow_intensity": 0.4,
                    "flake_size_range": (2, 5),
                    "flake_count": 600,
                    "atmospheric_effect": 0.2,
                    "brightness_increase": 0.15
                }
            ),
            IntensityLevel.HEAVY: AugmentationConfig(
                aug_type=AugmentationType.SNOW,
                intensity=IntensityLevel.HEAVY,
                parameters={
                    "snow_intensity": 0.6,
                    "flake_size_range": (3, 7),
                    "flake_count": 900,
                    "atmospheric_effect": 0.3,
                    "brightness_increase": 0.2
                }
            )
        }
    
    def apply_fog(self, image: np.ndarray, config: AugmentationConfig) -> np.ndarray:
        """Apply fog augmentation based on visibility and atmospheric perspective"""
        params = config.parameters
        h, w = image.shape[:2]
        
        # Create depth-based fog layer
        fog_layer = np.full_like(image, 255, dtype=np.uint8)
        
        # Apply color shift to fog (bluish-white)
        for i, shift in enumerate(params["color_shift"]):
            fog_layer[:, :, i] = np.clip(fog_layer[:, :, i] - shift, 0, 255)
        
        # Create depth gradient (objects further away get more fog)
        depth = np.tile(np.linspace(0, 1, h).reshape(h, 1), (1, w))
        
        # Apply atmospheric perspective
        fog_intensity = params["fog_intensity"]
        depth_blend = params["depth_blend"]
        atm_perspective = params["atmospheric_perspective"]
        
        # Calculate fog alpha with atmospheric perspective
        alpha = fog_intensity * (1 - np.exp(-depth_blend * depth))
        alpha = alpha * (1 + atm_perspective * depth)
        alpha = np.clip(alpha, 0, 0.8)  # Prevent complete white-out
        alpha = alpha[..., np.newaxis]
        
        # Apply fog effect
        fogged = cv2.convertScaleAbs(image * (1 - alpha) + fog_layer * alpha)
        
        return fogged
    
    def apply_night(self, image: np.ndarray, config: AugmentationConfig) -> np.ndarray:
        """Apply night/low-light augmentation with proper color temperature"""
        params = config.parameters
        
        # Apply gamma correction for low light
        gamma = params["gamma"]
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
        night_image = cv2.LUT(image, table)
        
        # Apply brightness reduction
        brightness_reduction = params["brightness_reduction"]
        night_image = cv2.convertScaleAbs(night_image, alpha=brightness_reduction, beta=0)
        
        # Apply desaturation
        desaturate_factor = params["desaturate_factor"]
        hsv = cv2.cvtColor(night_image, cv2.COLOR_BGR2HSV)
        hsv[..., 1] = hsv[..., 1] * desaturate_factor
        night_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Apply color temperature adjustment
        color_temp = params["color_temperature"]
        night_image = self._adjust_color_temperature(night_image, color_temp)
        
        # Add noise for realistic low-light conditions
        noise_factor = params["noise_factor"]
        noise = np.random.normal(0, noise_factor * 255, night_image.shape).astype(np.int16)
        night_image = np.clip(night_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return night_image
    
    def apply_motion_blur(self, image: np.ndarray, config: AugmentationConfig) -> np.ndarray:
        """Apply motion blur to simulate camera shake or movement"""
        params = config.parameters
        
        if np.random.random() > params["blur_probability"]:
            return image
        
        # Create motion blur kernel
        kernel_size = params["kernel_size"]
        angle = np.random.uniform(*params["angle_range"])
        motion_length = params["motion_length"]
        
        # Generate motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        
        # Calculate kernel center
        center = kernel_size // 2
        
        # Create line kernel for motion blur
        angle_rad = np.radians(angle)
        for i in range(motion_length):
            x = int(center + i * np.cos(angle_rad))
            y = int(center + i * np.sin(angle_rad))
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1
        
        # Normalize kernel
        kernel = kernel / np.sum(kernel)
        
        # Apply motion blur
        blurred = cv2.filter2D(image, -1, kernel)
        
        return blurred
    
    def apply_rain(self, image: np.ndarray, config: AugmentationConfig) -> np.ndarray:
        """Apply rain effect with atmospheric scattering"""
        params = config.parameters
        h, w = image.shape[:2]
        
        # Create rain layer
        rain_layer = np.zeros_like(image)
        
        # Generate rain drops
        drop_count = params["drop_count"]
        drop_length = params["drop_length"]
        drop_width = params["drop_width"]
        
        for _ in range(drop_count):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h - drop_length)
            
            # Draw rain drop
            cv2.line(rain_layer, (x, y), (x, y + drop_length), 
                    (200, 200, 200), drop_width)
        
        # Apply atmospheric effect
        atmospheric_effect = params["atmospheric_effect"]
        rain_intensity = params["rain_intensity"]
        
        # Blend rain with image
        alpha = rain_intensity
        rainy_image = cv2.addWeighted(image, 1 - alpha, rain_layer, alpha, 0)
        
        # Apply atmospheric scattering
        if atmospheric_effect > 0:
            scatter_layer = np.full_like(image, 180, dtype=np.uint8)
            rainy_image = cv2.addWeighted(rainy_image, 1 - atmospheric_effect, 
                                        scatter_layer, atmospheric_effect, 0)
        
        return rainy_image
    
    def apply_snow(self, image: np.ndarray, config: AugmentationConfig) -> np.ndarray:
        """Apply snow effect with brightness adjustment"""
        params = config.parameters
        h, w = image.shape[:2]
        
        # Create snow layer
        snow_layer = np.zeros_like(image)
        
        # Generate snow flakes
        flake_count = params["flake_count"]
        flake_size_range = params["flake_size_range"]
        
        for _ in range(flake_count):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            size = np.random.randint(*flake_size_range)
            
            # Draw snow flake
            cv2.circle(snow_layer, (x, y), size, (255, 255, 255), -1)
        
        # Apply snow effect
        snow_intensity = params["snow_intensity"]
        snowy_image = cv2.addWeighted(image, 1 - snow_intensity, snow_layer, snow_intensity, 0)
        
        # Apply brightness increase (snow reflects light)
        brightness_increase = params["brightness_increase"]
        snowy_image = cv2.convertScaleAbs(snowy_image, alpha=1 + brightness_increase, beta=0)
        
        # Apply atmospheric effect
        atmospheric_effect = params["atmospheric_effect"]
        if atmospheric_effect > 0:
            scatter_layer = np.full_like(image, 240, dtype=np.uint8)
            snowy_image = cv2.addWeighted(snowy_image, 1 - atmospheric_effect, 
                                        scatter_layer, atmospheric_effect, 0)
        
        return snowy_image
    
    def _adjust_color_temperature(self, image: np.ndarray, temperature: int) -> np.ndarray:
        """Adjust color temperature of image"""
        # Color temperature adjustment matrix
        if temperature < 3000:  # Warm light
            matrix = np.array([[1.0, 0.0, 0.0],
                             [0.0, 0.9, 0.0],
                             [0.0, 0.0, 0.7]])
        elif temperature < 4000:  # Neutral warm
            matrix = np.array([[1.0, 0.0, 0.0],
                             [0.0, 0.95, 0.0],
                             [0.0, 0.0, 0.8]])
        elif temperature < 5000:  # Cool white
            matrix = np.array([[0.9, 0.0, 0.0],
                             [0.0, 1.0, 0.0],
                             [0.0, 0.0, 1.0]])
        else:  # Cool/moonlight
            matrix = np.array([[0.8, 0.0, 0.0],
                             [0.0, 0.9, 0.0],
                             [0.0, 0.0, 1.0]])
        
        # Apply color temperature adjustment
        adjusted = cv2.transform(image, matrix)
        return np.clip(adjusted, 0, 255).astype(np.uint8)
    
    def augment_image(self, image: np.ndarray, aug_type: AugmentationType, 
                     intensity: IntensityLevel) -> Tuple[np.ndarray, AugmentationConfig]:
        """Apply specific augmentation to image"""
        config_dict = {
            AugmentationType.FOG: self.fog_configs,
            AugmentationType.NIGHT: self.night_configs,
            AugmentationType.MOTION_BLUR: self.motion_blur_configs,
            AugmentationType.RAIN: self.rain_configs,
            AugmentationType.SNOW: self.snow_configs
        }
        
        config = config_dict[aug_type][intensity]
        
        augment_func = {
            AugmentationType.FOG: self.apply_fog,
            AugmentationType.NIGHT: self.apply_night,
            AugmentationType.MOTION_BLUR: self.apply_motion_blur,
            AugmentationType.RAIN: self.apply_rain,
            AugmentationType.SNOW: self.apply_snow
        }
        
        augmented_image = augment_func[aug_type](image, config)
        
        return augmented_image, config
    
    def get_augmentation_metadata(self, config: AugmentationConfig) -> Dict:
        """Get metadata for augmentation for tracking and analysis"""
        return {
            "augmentation_type": config.aug_type.value,
            "intensity_level": config.intensity.value,
            "parameters": config.parameters,
            "seed": self.seed
        }
    
    def validate_augmentation_quality(self, original: np.ndarray, 
                                    augmented: np.ndarray) -> Dict[str, float]:
        """Validate augmentation quality using image metrics"""
        from skimage.metrics import structural_similarity as ssim
        from skimage.metrics import peak_signal_noise_ratio as psnr
        
        # Convert to grayscale for SSIM calculation
        gray_orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        gray_aug = cv2.cvtColor(augmented, cv2.COLOR_BGR2GRAY)
        
        # Calculate quality metrics
        ssim_score = ssim(gray_orig, gray_aug)
        psnr_score = psnr(gray_orig, gray_aug)
        
        # Calculate histogram difference
        hist_orig = cv2.calcHist([original], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
        hist_aug = cv2.calcHist([augmented], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
        hist_correlation = cv2.compareHist(hist_orig, hist_aug, cv2.HISTCMP_CORREL)
        
        return {
            "ssim": ssim_score,
            "psnr": psnr_score,
            "histogram_correlation": hist_correlation,
            "quality_score": (ssim_score + hist_correlation) / 2  # Combined quality metric
        } 