"""Road network extraction from aerial imagery using computer vision."""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from skimage import morphology, measure

from ..config import PreprocessingConfig
from .enhancement import ImageEnhancer


@dataclass
class RoadExtractionResult:
    """Result of road extraction from an image."""

    skeleton: np.ndarray  # Single-pixel width road skeleton
    edge_map: np.ndarray  # Raw edge detection result
    binary_roads: np.ndarray  # Binary road mask before skeletonization
    enhanced_image: np.ndarray  # Preprocessed image
    stats: dict  # Extraction statistics


class RoadExtractor:
    """Extract road networks from aerial imagery using edge detection and morphology."""

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """Initialize the road extractor.

        Args:
            config: Preprocessing configuration. Uses defaults if not provided.
        """
        self.config = config or PreprocessingConfig()
        self.enhancer = ImageEnhancer(self.config)

    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """Apply Canny edge detection.

        Args:
            image: Enhanced grayscale image.

        Returns:
            Binary edge map.
        """
        edges = cv2.Canny(
            image,
            threshold1=self.config.canny_low_threshold,
            threshold2=self.config.canny_high_threshold,
        )
        return edges

    def apply_morphological_closing(self, edge_image: np.ndarray) -> np.ndarray:
        """Apply morphological closing to connect nearby road segments.

        Closing (dilation followed by erosion) bridges small gaps in detected
        road edges, creating more continuous road segments.

        Args:
            edge_image: Binary edge map.

        Returns:
            Binary image with connected road segments.
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.config.morph_kernel_size, self.config.morph_kernel_size),
        )
        closed = cv2.morphologyEx(
            edge_image,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=self.config.morph_close_iterations,
        )
        return closed

    def skeletonize(self, binary_image: np.ndarray) -> np.ndarray:
        """Reduce binary road mask to single-pixel width skeleton.

        Args:
            binary_image: Binary road mask.

        Returns:
            Skeletonized image with single-pixel wide roads.
        """
        # Convert to boolean for skimage
        binary_bool = binary_image > 0
        skeleton = morphology.skeletonize(binary_bool)
        return skeleton.astype(np.uint8) * 255

    def filter_small_components(self, skeleton: np.ndarray) -> np.ndarray:
        """Remove small disconnected components (noise).

        Args:
            skeleton: Skeletonized binary image.

        Returns:
            Skeleton with small components removed.
        """
        # Label connected components
        labeled = measure.label(skeleton > 0, connectivity=2)

        # Calculate component sizes
        component_sizes = np.bincount(labeled.ravel())

        # Keep only components larger than threshold
        min_size = self.config.min_road_length_px
        small_components = component_sizes < min_size
        small_components[0] = False  # Don't remove background

        # Remove small components
        mask = small_components[labeled]
        filtered = skeleton.copy()
        filtered[mask] = 0

        return filtered

    def compute_road_coverage(
        self, skeleton: np.ndarray, image_shape: Tuple[int, int]
    ) -> float:
        """Compute the ratio of road pixels to total image pixels.

        Args:
            skeleton: Skeletonized road network.
            image_shape: Original image dimensions (height, width).

        Returns:
            Road coverage ratio (0 to 1).
        """
        road_pixels = np.sum(skeleton > 0)
        total_pixels = image_shape[0] * image_shape[1]
        return road_pixels / total_pixels

    def extract(self, image: np.ndarray) -> RoadExtractionResult:
        """Extract road network from an aerial image.

        Full pipeline:
        1. Enhance image (CLAHE + bilateral filter)
        2. Detect edges (Canny)
        3. Apply morphological closing
        4. Skeletonize
        5. Filter small components

        Args:
            image: Grayscale aerial image.

        Returns:
            RoadExtractionResult with skeleton and intermediate results.
        """
        # Step 1: Enhance image
        enhanced = self.enhancer.enhance(image)

        # Step 2: Detect edges
        edges = self.detect_edges(enhanced)

        # Step 3: Morphological closing
        binary_roads = self.apply_morphological_closing(edges)

        # Step 4: Skeletonize
        skeleton = self.skeletonize(binary_roads)

        # Step 5: Filter noise
        skeleton = self.filter_small_components(skeleton)

        # Compute statistics
        road_coverage = self.compute_road_coverage(skeleton, image.shape)
        road_pixels = int(np.sum(skeleton > 0))

        stats = {
            "road_pixels": road_pixels,
            "road_coverage_ratio": road_coverage,
            "image_shape": image.shape,
            "edge_pixels": int(np.sum(edges > 0)),
            "is_sparse": road_coverage < self.config.min_road_coverage_ratio,
        }

        return RoadExtractionResult(
            skeleton=skeleton,
            edge_map=edges,
            binary_roads=binary_roads,
            enhanced_image=enhanced,
            stats=stats,
        )

    def extract_with_retry(
        self, image: np.ndarray, max_attempts: int = 3
    ) -> RoadExtractionResult:
        """Extract roads with automatic parameter adjustment on failure.

        If initial extraction yields too few roads, retry with adjusted parameters.

        Args:
            image: Grayscale aerial image.
            max_attempts: Maximum extraction attempts with different parameters.

        Returns:
            Best RoadExtractionResult from all attempts.
        """
        best_result = None
        best_coverage = 0.0

        # Parameter variations to try
        param_variations = [
            {},  # Default parameters
            {"clahe_clip_limit": 3.0, "canny_low_threshold": 30, "canny_high_threshold": 100},
            {"clahe_clip_limit": 1.5, "canny_low_threshold": 70, "canny_high_threshold": 180},
        ]

        for i, variations in enumerate(param_variations[:max_attempts]):
            # Update config with variations
            config = PreprocessingConfig()
            for key, value in variations.items():
                setattr(config, key, value)

            # Create extractor with modified config
            extractor = RoadExtractor(config)
            result = extractor.extract(image)

            # Keep best result
            if result.stats["road_coverage_ratio"] > best_coverage:
                best_coverage = result.stats["road_coverage_ratio"]
                best_result = result

            # Stop if we have reasonable coverage
            if best_coverage >= self.config.min_road_coverage_ratio:
                break

        return best_result or result
