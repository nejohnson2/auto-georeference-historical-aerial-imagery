"""Image enhancement for improving road visibility in historical aerial imagery."""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

from ..config import PreprocessingConfig


class ImageEnhancer:
    """Apply contrast enhancement and noise reduction for road detection."""

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """Initialize the image enhancer.

        Args:
            config: Preprocessing configuration. Uses defaults if not provided.
        """
        self.config = config or PreprocessingConfig()

    def load_image(self, path: Path) -> np.ndarray:
        """Load an image as grayscale.

        Args:
            path: Path to the image file (TIFF, JPEG, PNG supported).

        Returns:
            Grayscale image as numpy array.

        Raises:
            FileNotFoundError: If image file doesn't exist.
            ValueError: If image cannot be loaded.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image: {path}")

        return image

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization.

        CLAHE improves local contrast, which is essential for detecting roads
        in historical aerial images with varying exposure across the frame.

        Args:
            image: Grayscale input image.

        Returns:
            Contrast-enhanced image.
        """
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_tile_grid_size,
        )
        return clahe.apply(image)

    def apply_bilateral_filter(self, image: np.ndarray) -> np.ndarray:
        """Apply bilateral filtering for edge-preserving noise reduction.

        Bilateral filtering smooths the image while preserving edges,
        which helps reduce film grain noise while keeping road edges sharp.

        Args:
            image: Grayscale input image.

        Returns:
            Filtered image with reduced noise.
        """
        return cv2.bilateralFilter(
            image,
            d=self.config.bilateral_d,
            sigmaColor=self.config.bilateral_sigma_color,
            sigmaSpace=self.config.bilateral_sigma_space,
        )

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """Apply full enhancement pipeline.

        Pipeline:
        1. CLAHE for local contrast enhancement
        2. Bilateral filter for noise reduction

        Args:
            image: Grayscale input image.

        Returns:
            Enhanced image ready for road extraction.
        """
        enhanced = self.apply_clahe(image)
        enhanced = self.apply_bilateral_filter(enhanced)
        return enhanced

    def compute_image_stats(self, image: np.ndarray) -> dict:
        """Compute image statistics for adaptive parameter selection.

        Args:
            image: Grayscale input image.

        Returns:
            Dictionary with image statistics.
        """
        return {
            "mean": float(np.mean(image)),
            "std": float(np.std(image)),
            "min": int(np.min(image)),
            "max": int(np.max(image)),
            "median": float(np.median(image)),
            "shape": image.shape,
        }

    def auto_adjust_parameters(self, image: np.ndarray) -> PreprocessingConfig:
        """Automatically adjust preprocessing parameters based on image characteristics.

        Args:
            image: Grayscale input image.

        Returns:
            Adjusted preprocessing configuration.
        """
        stats = self.compute_image_stats(image)
        config = PreprocessingConfig()

        # Adjust CLAHE clip limit based on contrast
        if stats["std"] < 30:
            # Low contrast image - increase CLAHE effect
            config.clahe_clip_limit = 3.0
        elif stats["std"] > 60:
            # High contrast - reduce CLAHE effect
            config.clahe_clip_limit = 1.5

        # Adjust Canny thresholds based on median intensity
        median = stats["median"]
        sigma = 0.33
        config.canny_low_threshold = int(max(0, (1.0 - sigma) * median))
        config.canny_high_threshold = int(min(255, (1.0 + sigma) * median))

        return config
