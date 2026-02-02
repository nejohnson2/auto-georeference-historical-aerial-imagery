"""Water body detection and shoreline extraction from aerial imagery."""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from scipy import ndimage

from ..config import WaterConfig
from .enhancement import ImageEnhancer


@dataclass
class WaterExtractionResult:
    """Result of water body detection from an image."""

    water_mask: np.ndarray  # Binary mask of detected water
    shoreline_contours: List[np.ndarray]  # List of (N, 2) contour arrays in pixel coords
    enhanced_image: np.ndarray  # Preprocessed image
    stats: dict  # Detection statistics


class WaterExtractor:
    """Extract water bodies and shorelines from aerial imagery.

    Water is detected using a combination of:
    1. Intensity thresholding (water appears darker in aerial imagery)
    2. Texture analysis (water has lower local variance than land)
    """

    def __init__(self, config: Optional[WaterConfig] = None):
        """Initialize the water extractor.

        Args:
            config: Water configuration. Uses defaults if not provided.
        """
        self.config = config or WaterConfig()
        self.enhancer = ImageEnhancer()

    def detect_water_regions(self, image: np.ndarray) -> np.ndarray:
        """Detect water regions using intensity and texture analysis.

        Water typically appears darker and smoother (lower texture variance)
        than land in grayscale aerial imagery.

        Args:
            image: Grayscale aerial image (enhanced).

        Returns:
            Binary water mask (255 for water, 0 for land).
        """
        # Stage 1: Intensity-based detection
        # Water is typically darker than land
        median_intensity = np.median(image)
        std_intensity = np.std(image)
        intensity_threshold = median_intensity + self.config.water_intensity_threshold * std_intensity
        intensity_mask = image < intensity_threshold

        # Stage 2: Texture-based refinement
        # Water has lower local variance (smoother) than land
        window_size = self.config.texture_window_size
        local_mean = ndimage.uniform_filter(image.astype(np.float64), size=window_size)
        local_sq_mean = ndimage.uniform_filter(image.astype(np.float64) ** 2, size=window_size)
        local_variance = local_sq_mean - local_mean ** 2
        texture_mask = local_variance < self.config.texture_variance_threshold

        # Stage 3: Combine masks
        water_mask = (intensity_mask & texture_mask).astype(np.uint8) * 255

        # Stage 4: Morphological cleanup
        # Opening removes small noise specks
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel_open)

        # Closing fills small holes in water bodies
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel_close)

        # Stage 5: Remove small water bodies
        water_mask = self._filter_small_components(water_mask)

        return water_mask

    def _filter_small_components(self, water_mask: np.ndarray) -> np.ndarray:
        """Remove small connected components from water mask.

        Args:
            water_mask: Binary water mask.

        Returns:
            Filtered water mask with small components removed.
        """
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            water_mask, connectivity=8
        )

        # Create output mask
        filtered_mask = np.zeros_like(water_mask)

        # Keep components larger than threshold
        min_area = self.config.min_water_area_px
        for label in range(1, num_labels):  # Skip background (label 0)
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= min_area:
                filtered_mask[labels == label] = 255

        return filtered_mask

    def extract_shorelines(self, water_mask: np.ndarray) -> List[np.ndarray]:
        """Extract shoreline contours from water mask.

        Args:
            water_mask: Binary water mask.

        Returns:
            List of contours, each as (N, 2) array of (x, y) pixel coordinates.
        """
        # Find contours (external only - we want water boundaries)
        contours, _ = cv2.findContours(
            water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        shorelines = []
        min_length = self.config.min_shoreline_length_px

        for contour in contours:
            # Compute arc length
            arc_length = cv2.arcLength(contour, closed=True)
            if arc_length < min_length:
                continue

            # Smooth contour to reduce noise
            epsilon = self.config.contour_smoothing_epsilon * arc_length
            smoothed = cv2.approxPolyDP(contour, epsilon, closed=True)

            # Convert from OpenCV format (N, 1, 2) to simple (N, 2)
            shoreline = smoothed.reshape(-1, 2).astype(np.float64)
            shorelines.append(shoreline)

        return shorelines

    def compute_water_coverage(
        self, water_mask: np.ndarray, image_shape: Tuple[int, int]
    ) -> float:
        """Compute the ratio of water pixels to total image pixels.

        Args:
            water_mask: Binary water mask.
            image_shape: Original image dimensions (height, width).

        Returns:
            Water coverage ratio (0 to 1).
        """
        water_pixels = np.sum(water_mask > 0)
        total_pixels = image_shape[0] * image_shape[1]
        return water_pixels / total_pixels

    def extract(self, image: np.ndarray) -> WaterExtractionResult:
        """Extract water bodies and shorelines from an aerial image.

        Full pipeline:
        1. Enhance image (CLAHE + bilateral filter)
        2. Detect water regions (intensity + texture)
        3. Extract shoreline contours
        4. Compute statistics

        Args:
            image: Grayscale aerial image.

        Returns:
            WaterExtractionResult with water mask, shorelines, and stats.
        """
        # Step 1: Enhance image
        enhanced = self.enhancer.enhance(image)

        # Step 2: Detect water regions
        water_mask = self.detect_water_regions(enhanced)

        # Step 3: Extract shoreline contours
        shoreline_contours = self.extract_shorelines(water_mask)

        # Compute statistics
        water_coverage = self.compute_water_coverage(water_mask, image.shape)
        total_shoreline_length = sum(
            cv2.arcLength(c.reshape(-1, 1, 2).astype(np.float32), closed=True)
            for c in shoreline_contours
        )

        stats = {
            "water_pixels": int(np.sum(water_mask > 0)),
            "water_coverage_ratio": water_coverage,
            "num_water_bodies": len(shoreline_contours),
            "total_shoreline_length_px": total_shoreline_length,
            "image_shape": image.shape,
            "has_significant_water": water_coverage > 0.05,  # >5% coverage
        }

        return WaterExtractionResult(
            water_mask=water_mask,
            shoreline_contours=shoreline_contours,
            enhanced_image=enhanced,
            stats=stats,
        )

    def extract_with_retry(
        self, image: np.ndarray, max_attempts: int = 3
    ) -> WaterExtractionResult:
        """Extract water with automatic parameter adjustment.

        If initial extraction yields too little water, retry with adjusted parameters.

        Args:
            image: Grayscale aerial image.
            max_attempts: Maximum extraction attempts with different parameters.

        Returns:
            Best WaterExtractionResult from all attempts.
        """
        best_result = None
        best_score = 0.0

        # Parameter variations to try (more aggressive water detection)
        param_variations = [
            {},  # Default parameters
            {
                "water_intensity_threshold": -1.0,  # Less strict intensity
                "texture_variance_threshold": 150.0,  # More variance allowed
            },
            {
                "water_intensity_threshold": -2.0,  # Stricter intensity
                "texture_variance_threshold": 50.0,  # Stricter texture
                "min_water_area_px": 200,  # Allow smaller water bodies
            },
        ]

        for i, variations in enumerate(param_variations[:max_attempts]):
            # Create config with variations
            config = WaterConfig()
            for key, value in variations.items():
                setattr(config, key, value)

            # Create extractor with modified config
            extractor = WaterExtractor(config)
            result = extractor.extract(image)

            # Score based on reasonable water coverage (not too little, not too much)
            coverage = result.stats["water_coverage_ratio"]
            # Prefer coverage between 5% and 50%
            if 0.05 <= coverage <= 0.5:
                score = 1.0 - abs(coverage - 0.25) / 0.25  # Peak at 25%
            elif coverage > 0.5:
                score = 0.3  # Too much water detected
            else:
                score = coverage / 0.05 * 0.5  # Scale up to 0.5 for <5%

            # Also consider shoreline count
            if result.stats["num_water_bodies"] > 0:
                score += 0.2

            if score > best_score:
                best_score = score
                best_result = result

        return best_result or result
