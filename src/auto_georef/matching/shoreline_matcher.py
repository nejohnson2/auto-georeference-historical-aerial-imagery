"""Shoreline matching using curvature-based contour alignment."""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
import logging

from ..config import WaterConfig, MatchingConfig

logger = logging.getLogger(__name__)


@dataclass
class ShorelineCorrespondence:
    """A correspondence between image and OSM shoreline points."""

    image_point: Tuple[float, float]  # (x, y) in pixels
    osm_point: Tuple[float, float]  # (x, y) in meters
    distance: float  # Match quality score (lower is better)
    segment_id: int = 0  # Which shoreline segment this came from


class ShorelineMatcher:
    """Match image shorelines to OSM shorelines using curvature-based alignment.

    The algorithm:
    1. Compute curvature signature for each contour
    2. Use cross-correlation to find best alignment between contours
    3. Extract corresponding points at equal arc-length intervals
    """

    def __init__(
        self,
        water_config: Optional[WaterConfig] = None,
        matching_config: Optional[MatchingConfig] = None,
    ):
        """Initialize the shoreline matcher.

        Args:
            water_config: Water configuration.
            matching_config: Matching configuration.
        """
        self.water_config = water_config or WaterConfig()
        self.matching_config = matching_config or MatchingConfig()

    def compute_curvature_signature(
        self, contour: np.ndarray, window_size: Optional[int] = None
    ) -> np.ndarray:
        """Compute curvature at each point along a contour.

        Curvature k = (x'y'' - y'x'') / (x'^2 + y'^2)^1.5

        Args:
            contour: (N, 2) array of (x, y) coordinates.
            window_size: Smoothing window size.

        Returns:
            (N,) array of curvature values.
        """
        if len(contour) < 5:
            return np.zeros(len(contour))

        window_size = window_size or self.water_config.curvature_window_size

        # Smooth the contour first to reduce noise
        kernel = np.ones(window_size) / window_size
        x = np.convolve(contour[:, 0], kernel, mode="same")
        y = np.convolve(contour[:, 1], kernel, mode="same")

        # First derivatives
        dx = np.gradient(x)
        dy = np.gradient(y)

        # Second derivatives
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        # Curvature formula
        denominator = (dx ** 2 + dy ** 2 + 1e-10) ** 1.5
        curvature = (dx * ddy - dy * ddx) / denominator

        return curvature

    def compute_arc_length(self, contour: np.ndarray) -> np.ndarray:
        """Compute cumulative arc length along a contour.

        Args:
            contour: (N, 2) array of coordinates.

        Returns:
            (N,) array of cumulative arc lengths.
        """
        diffs = np.diff(contour, axis=0)
        segment_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
        arc_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])
        return arc_lengths

    def sample_contour_by_arc_length(
        self, contour: np.ndarray, num_samples: int
    ) -> np.ndarray:
        """Sample a contour at equal arc-length intervals.

        Args:
            contour: (N, 2) array of coordinates.
            num_samples: Number of samples to extract.

        Returns:
            (num_samples, 2) array of sampled points.
        """
        arc_lengths = self.compute_arc_length(contour)
        total_length = arc_lengths[-1]

        if total_length < 1e-10:
            return contour[:num_samples] if len(contour) >= num_samples else contour

        # Sample at equal intervals
        sample_distances = np.linspace(0, total_length, num_samples)
        sampled_points = np.zeros((num_samples, 2))

        for i, dist in enumerate(sample_distances):
            # Find the segment containing this distance
            idx = np.searchsorted(arc_lengths, dist, side="right") - 1
            idx = max(0, min(idx, len(contour) - 2))

            # Interpolate within the segment
            segment_start = arc_lengths[idx]
            segment_end = arc_lengths[idx + 1]
            segment_length = segment_end - segment_start

            if segment_length > 0:
                t = (dist - segment_start) / segment_length
            else:
                t = 0

            sampled_points[i] = contour[idx] + t * (contour[idx + 1] - contour[idx])

        return sampled_points

    def align_by_curvature(
        self,
        img_curvature: np.ndarray,
        osm_curvature: np.ndarray,
        scale_range: Tuple[float, float] = (0.5, 2.0),
        num_scales: int = 10,
    ) -> Tuple[int, float, float]:
        """Find best alignment between two curvature signatures.

        Uses cross-correlation at multiple scales to find:
        - Offset: where the image contour aligns with OSM contour
        - Scale: length ratio between contours
        - Correlation: alignment quality score

        Args:
            img_curvature: Curvature signature of image contour.
            osm_curvature: Curvature signature of OSM contour.
            scale_range: Range of scales to search.
            num_scales: Number of scales to try.

        Returns:
            Tuple of (best_offset, best_scale, best_correlation).
        """
        if len(img_curvature) < 5 or len(osm_curvature) < 5:
            return 0, 1.0, 0.0

        # Normalize curvatures
        img_std = np.std(img_curvature)
        osm_std = np.std(osm_curvature)

        if img_std < 1e-10 or osm_std < 1e-10:
            # Flat contours - not enough curvature variation
            return 0, 1.0, 0.0

        img_norm = (img_curvature - np.mean(img_curvature)) / img_std
        osm_norm = (osm_curvature - np.mean(osm_curvature)) / osm_std

        best_correlation = -np.inf
        best_offset = 0
        best_scale = 1.0

        # Search over scales
        scales = np.linspace(scale_range[0], scale_range[1], num_scales)

        for scale in scales:
            # Resample OSM curvature to different length
            resampled_length = int(len(osm_norm) * scale)
            if resampled_length < 5:
                continue

            osm_resampled = np.interp(
                np.linspace(0, 1, resampled_length),
                np.linspace(0, 1, len(osm_norm)),
                osm_norm,
            )

            # Cross-correlation
            if len(img_norm) >= len(osm_resampled):
                correlation = np.correlate(img_norm, osm_resampled, mode="valid")
            else:
                correlation = np.correlate(osm_resampled, img_norm, mode="valid")

            if len(correlation) > 0:
                max_idx = np.argmax(correlation)
                max_corr = correlation[max_idx]

                # Normalize by length
                max_corr /= min(len(img_norm), len(osm_resampled))

                if max_corr > best_correlation:
                    best_correlation = max_corr
                    best_offset = max_idx if len(img_norm) >= len(osm_resampled) else -max_idx
                    best_scale = scale

        return best_offset, best_scale, best_correlation

    def find_point_correspondences(
        self,
        image_contour: np.ndarray,
        osm_contour: np.ndarray,
        offset: int,
        scale: float,
        num_correspondences: int = 10,
    ) -> List[ShorelineCorrespondence]:
        """Extract point correspondences from aligned contours.

        Args:
            image_contour: (N, 2) array of image coordinates (pixels).
            osm_contour: (M, 2) array of OSM coordinates (meters).
            offset: Alignment offset from curvature matching.
            scale: Length scale from curvature matching.
            num_correspondences: Number of points to extract.

        Returns:
            List of ShorelineCorrespondence objects.
        """
        correspondences = []

        # Sample points at equal arc-length intervals
        img_sampled = self.sample_contour_by_arc_length(image_contour, num_correspondences)

        # Compute how many samples to take from OSM contour (accounting for scale)
        osm_num_samples = int(num_correspondences * scale)
        osm_num_samples = max(3, min(osm_num_samples, len(osm_contour)))
        osm_sampled = self.sample_contour_by_arc_length(osm_contour, osm_num_samples)

        # Match samples based on normalized position along contour
        for i, img_point in enumerate(img_sampled):
            # Find corresponding OSM point (accounting for offset and scale)
            img_frac = i / (len(img_sampled) - 1) if len(img_sampled) > 1 else 0
            osm_idx = int((img_frac * (osm_num_samples - 1)))
            osm_idx = max(0, min(osm_idx, len(osm_sampled) - 1))

            osm_point = osm_sampled[osm_idx]

            # Compute local curvature similarity as quality score
            img_curv = self.compute_curvature_signature(
                image_contour[max(0, int(i * len(image_contour) / num_correspondences) - 5):
                             min(len(image_contour), int(i * len(image_contour) / num_correspondences) + 5)]
            )
            osm_curv = self.compute_curvature_signature(
                osm_contour[max(0, osm_idx * len(osm_contour) // osm_num_samples - 5):
                           min(len(osm_contour), osm_idx * len(osm_contour) // osm_num_samples + 5)]
            )

            if len(img_curv) > 0 and len(osm_curv) > 0:
                distance = np.abs(np.mean(img_curv) - np.mean(osm_curv))
            else:
                distance = 1.0

            correspondences.append(
                ShorelineCorrespondence(
                    image_point=tuple(img_point),
                    osm_point=tuple(osm_point),
                    distance=distance,
                    segment_id=0,
                )
            )

        return correspondences

    def match_single_pair(
        self,
        image_contour: np.ndarray,
        osm_contour: np.ndarray,
        max_correspondences: int = 10,
    ) -> Tuple[List[ShorelineCorrespondence], float]:
        """Match a single image contour to a single OSM contour.

        Args:
            image_contour: (N, 2) array of image coordinates (pixels).
            osm_contour: (M, 2) array of OSM coordinates (meters).
            max_correspondences: Maximum points to extract.

        Returns:
            Tuple of (correspondences, match_quality).
        """
        if len(image_contour) < 5 or len(osm_contour) < 5:
            return [], 0.0

        # Compute curvature signatures
        img_curvature = self.compute_curvature_signature(image_contour)
        osm_curvature = self.compute_curvature_signature(osm_contour)

        # Find best alignment
        offset, scale, correlation = self.align_by_curvature(
            img_curvature, osm_curvature
        )

        if correlation < 0.1:  # Poor alignment
            return [], correlation

        # Extract point correspondences
        correspondences = self.find_point_correspondences(
            image_contour, osm_contour, offset, scale, max_correspondences
        )

        return correspondences, correlation

    def match(
        self,
        image_shorelines: List[np.ndarray],
        osm_shorelines: List[np.ndarray],
        max_correspondences: Optional[int] = None,
    ) -> List[ShorelineCorrespondence]:
        """Match image shorelines to OSM shorelines.

        Finds the best pairing between image and OSM contours,
        then extracts point correspondences.

        Args:
            image_shorelines: List of (N, 2) arrays in pixel coordinates.
            osm_shorelines: List of (M, 2) arrays in meter coordinates.
            max_correspondences: Maximum total correspondences to return.

        Returns:
            List of ShorelineCorrespondence objects.
        """
        max_correspondences = max_correspondences or self.water_config.max_shoreline_correspondences

        if not image_shorelines or not osm_shorelines:
            return []

        all_correspondences = []
        all_scores = []

        # Try all pairings of image and OSM contours
        for img_idx, img_contour in enumerate(image_shorelines):
            for osm_idx, osm_contour in enumerate(osm_shorelines):
                correspondences, score = self.match_single_pair(
                    img_contour,
                    osm_contour,
                    max_correspondences=max_correspondences // max(1, len(image_shorelines)),
                )

                # Tag with segment IDs
                for c in correspondences:
                    c.segment_id = img_idx * 100 + osm_idx

                if correspondences:
                    all_correspondences.extend(correspondences)
                    all_scores.extend([score] * len(correspondences))

        if not all_correspondences:
            return []

        # Sort by quality (lower distance = better)
        sorted_pairs = sorted(
            zip(all_correspondences, all_scores),
            key=lambda x: x[0].distance - x[1] * 0.5,  # Prefer high correlation, low distance
        )

        # Return top correspondences
        result = [c for c, _ in sorted_pairs[:max_correspondences]]

        logger.info(
            f"Shoreline matching: {len(image_shorelines)} image contours, "
            f"{len(osm_shorelines)} OSM contours -> {len(result)} correspondences"
        )

        return result
