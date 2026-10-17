
"""""
Run:
    python grain_boundary_pipeline.py --input ./images --output ./grain_dataset

Image extensions: png, jpg, jpeg, tif, tiff, bmp
"""

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def read_image(path):
    # IMREAD_UNCHANGED supports grayscale and many TIFF inputs.
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype != np.uint8:
        # Scale 16-bit or floating-point microscopy images to 8-bit for CV.
        # For quantitative microscopy, review this scaling choice for your data.
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return image


def xdog(gray, sigma=1.4, k_sigma=4.3, epsilon=0.02, phi=30, gamma=0.96):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray).astype(np.float32) / 255.0
    g1 = cv2.GaussianBlur(contrast, (0, 0), float(sigma))
    g2 = cv2.GaussianBlur(contrast, (0, 0), float(sigma) * float(k_sigma))
    dog = g1 - float(gamma) * g2
    out = np.ones_like(dog)
    idx = dog < float(epsilon)
    out[idx] = 1.0 + np.tanh(float(phi) * (dog[idx] - float(epsilon)))
    return (out * 255).clip(0, 255).astype(np.uint8)


def remove_small_components(binary, min_area=50):
    # Convention: boundary pixels are black (0), background is white (255).
    foreground = (binary == 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
    cleaned = np.full_like(binary, 255)
    for label in range(1, n):
        if stats[label, cv2.CC_STAT_AREA] >= int(min_area):
            cleaned[labels == label] = 0
    return cleaned


def make_mask(gray, params):
    # Keep bilateral smoothing from the original cv.py, with tunable strength.
    smooth = cv2.bilateralFilter(
        gray, d=9, sigmaColor=int(params["bilateral_color"]),
        sigmaSpace=int(params["bilateral_space"])
    )
    edge = xdog(
        smooth, sigma=params["sigma"], k_sigma=params["k_sigma"],
        epsilon=params["epsilon"], phi=params["phi"], gamma=params["gamma"]
    )
    _, binary = cv2.threshold(edge, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cleaned = remove_small_components(binary, params["min_area"])
    return edge, cleaned


def default_params():
    return {
        "sigma": 1.4, "k_sigma": 4.3, "epsilon": 0.02,
        "phi": 30, "gamma": 0.96, "min_area": 50,
        "bilateral_color": 75, "bilateral_space": 75
    }

