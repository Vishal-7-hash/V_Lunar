"""
Georeferencing Module for Lunar Image Registration (.png images)

Aligns lunar images from different projections to a common coordinate system.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict


def align_images(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Align source image to reference image coordinate system.
    
    Args:
        source: Source image (numpy array, grayscale or color)
        reference: Reference image (numpy array)
        
    Returns:
        aligned_source: Source image aligned to reference dimensions and system
    """
    # Get dimensions
    src_h, src_w = source.shape[:2]
    ref_h, ref_w = reference.shape[:2]
    
    # If same size, return as-is (already aligned)
    if src_h == ref_h and src_w == ref_w:
        return source
    
    # Resize source to match reference dimensions
    aligned = cv2.resize(source, (ref_w, ref_h), interpolation=cv2.INTER_LINEAR)
    
    return aligned


def reproject_by_bounds(image: np.ndarray, 
                       source_bounds: Dict, 
                       target_bounds: Dict,
                       target_shape: Tuple) -> np.ndarray:
    """
    Reproject image based on geographic bounds.
    
    Args:
        image: Input image
        source_bounds: dict with keys 'north', 'south', 'east', 'west' (source coordinates)
        target_bounds: dict with keys 'north', 'south', 'east', 'west' (target coordinates)
        target_shape: Target image shape (height, width)
        
    Returns:
        reprojected_image: Image transformed to target coordinate system
    """
    src_h, src_w = image.shape[:2]
    tgt_h, tgt_w = target_shape
    
    # Create coordinate grids
    # Target grid: where we want to sample from
    tgt_y, tgt_x = np.mgrid[0:tgt_h, 0:tgt_w]
    
    # Normalize target coordinates to [0, 1]
    norm_y = tgt_y / (tgt_h - 1) if tgt_h > 1 else tgt_y
    norm_x = tgt_x / (tgt_w - 1) if tgt_w > 1 else tgt_x
    
    # Map to source image coordinates using bounds
    # Target bounds -> geographic coords -> source pixel coords
    src_lat_min = source_bounds['south']
    src_lat_max = source_bounds['north']
    src_lon_min = source_bounds['west']
    src_lon_max = source_bounds['east']
    
    tgt_lat_min = target_bounds['south']
    tgt_lat_max = target_bounds['north']
    tgt_lon_min = target_bounds['west']
    tgt_lon_max = target_bounds['east']
    
    # Geographic coordinates in target system
    tgt_lats = tgt_lat_max - norm_y * (tgt_lat_max - tgt_lat_min)
    tgt_lons = tgt_lon_min + norm_x * (tgt_lon_max - tgt_lon_min)
    
    # Map to source pixel coordinates
    src_py = (src_lat_max - tgt_lats) / (src_lat_max - src_lat_min) * (src_h - 1)
    src_px = (tgt_lons - src_lon_min) / (src_lon_max - src_lon_min) * (src_w - 1)
    
    # Clip to valid range
    src_px = np.clip(src_px, 0, src_w - 1).astype(np.float32)
    src_py = np.clip(src_py, 0, src_h - 1).astype(np.float32)
    
    # Apply remap with bilinear interpolation
    reprojected = cv2.remap(
        image.astype(np.float32),
        src_px, src_py,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )
    
    return reprojected


def georeferencing(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Align source image to reference image (simple version for .png).
    
    Args:
        source: Source image (moving image)
        reference: Reference image (fixed image)
        
    Returns:
        aligned_source: Source aligned to reference
    """
    return align_images(source, reference)


def visualize_georeferencing(source: np.ndarray, reference: np.ndarray, 
                            aligned_source: np.ndarray, 
                            title: str = "Lunar Image Georeferencing") -> None:
    """
    Visualize source, reference, and georeferenced images side by side.
    
    Args:
        source: Original source image
        reference: Reference image
        aligned_source: Georeferenced source image
        title: Title for the visualization window
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Display source image
    axes[0].imshow(source, cmap='gray')
    axes[0].set_title('Source Image (Moving)', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Display reference image
    axes[1].imshow(reference, cmap='gray')
    axes[1].set_title('Reference Image (Fixed)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Display aligned/georeferenced source
    axes[2].imshow(aligned_source, cmap='gray')
    axes[2].set_title('Georeferenced Source', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def visualize_comparison(reference: np.ndarray, aligned_source: np.ndarray,
                        alpha: float = 0.5, title: str = "Overlay Comparison") -> None:
    """
    Visualize overlay comparison of reference and georeferenced source.
    
    Args:
        reference: Reference image
        aligned_source: Georeferenced source image
        alpha: Alpha blending factor (0-1)
        title: Title for the visualization window
    """
    # Create overlay: blend reference and aligned source
    overlay = cv2.addWeighted(reference, alpha, aligned_source, 1 - alpha, 0)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Display reference
    axes[0].imshow(reference, cmap='gray')
    axes[0].set_title('Reference Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Display overlay
    axes[1].imshow(overlay, cmap='gray')
    axes[1].set_title(f'Overlay (Reference {int(alpha*100)}% + Source {int((1-alpha)*100)}%)', 
                     fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def visualize_single(image: np.ndarray, title: str = "Lunar Image", 
                     figsize: Tuple = (8, 8)) -> None:
    """
    Visualize a single image with title.
    
    Args:
        image: Image to visualize
        title: Title for the image
        figsize: Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    plt.imshow(image, cmap='gray')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

