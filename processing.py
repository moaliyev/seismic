"""Gaussian smoothing and sharpening for a single 2D slice.

OpenCV does the filtering: it is already a dependency, it works directly on the
float32 arrays the store hands out, and it is fast enough to refilter on every
slider move.
"""

import cv2
import numpy as np

NONE = "None"
GAUSSIAN = "Gaussian smoothing"
SHARPEN = "Sharpening (unsharp mask)"
FILTERS = (NONE, GAUSSIAN, SHARPEN)


def gaussian_smooth(image, sigma):
    """Blur with an isotropic Gaussian; sigma is in samples."""
    image = np.ascontiguousarray(image, dtype=np.float32)
    if sigma <= 0:
        return image
    # ksize (0, 0) lets OpenCV size the kernel from sigma, and reflecting at the
    # border avoids the dark rim a zero-padded edge would leave on the slice
    return cv2.GaussianBlur(
        image, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma),
        borderType=cv2.BORDER_REFLECT,
    )


def sharpen(image, sigma, amount):
    """Unsharp mask: add back a scaled copy of whatever the blur removed."""
    image = np.ascontiguousarray(image, dtype=np.float32)
    if sigma <= 0 or amount <= 0:
        return image
    blurred = gaussian_smooth(image, sigma)
    return cv2.addWeighted(image, 1.0 + float(amount), blurred, -float(amount), 0.0)


def apply_filter(image, name, sigma=1.5, amount=1.0):
    """Dispatch by filter name; an unknown name passes the slice through."""
    if name == GAUSSIAN:
        return gaussian_smooth(image, sigma)
    if name == SHARPEN:
        return sharpen(image, sigma, amount)
    return np.ascontiguousarray(image, dtype=np.float32)


def describe(name, sigma, amount):
    if name == GAUSSIAN:
        return f"{name}, sigma {sigma:g}"
    if name == SHARPEN:
        return f"{name}, sigma {sigma:g}, amount {amount:g}"
    return name
