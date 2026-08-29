import cv2
import numpy as np
import matplotlib.pyplot as plt


def intensity_normalization(image, low_percentile=2, high_percentile=98):

    # Convert image to float
    image_float = image.astype(np.float32)

    # Original intensity range
    original_min = image_float.min()
    original_max = image_float.max()

    # Find percentile intensity values
    low = np.percentile(image_float, low_percentile)
    high = np.percentile(image_float, high_percentile)

    # Count pixels below and above the selected range
    lower_pixels = np.sum(image_float < low)
    upper_pixels = np.sum(image_float > high)

    total_pixels = image_float.size

    lower_percentage = (lower_pixels / total_pixels) * 100
    upper_percentage = (upper_pixels / total_pixels) * 100

    # Avoid division by zero
    if high <= low:
        return image.astype(np.uint8)

    # Clip extreme values
    image_clipped = np.clip(image_float, low, high)

    # Normalize to 0-255
    normalized = (image_clipped - low) / (high - low)
    normalized = (normalized * 255).astype(np.uint8)

    # Display statistics
    print("\nOriginal intensity range:",
          f"{original_min:.2f} - {original_max:.2f}")

    print("Normalization range:",
          f"{low:.2f} - {high:.2f}")

    print("Lower clipped pixels:",
          f"{lower_percentage:.2f}%")

    print("Upper clipped pixels:",
          f"{upper_percentage:.2f}%")

    print("Total extreme pixels:",
          f"{lower_percentage + upper_percentage:.2f}%")

    print("Normalized intensity range: 0 - 255")

    return normalized

