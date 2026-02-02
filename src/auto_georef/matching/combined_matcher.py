"""Combine road and shoreline correspondences for unified RANSAC matching."""

import numpy as np
import networkx as nx
from dataclasses import dataclass
from typing import List, Tuple, Optional
import logging

from .shoreline_matcher import ShorelineCorrespondence
from ..config import WaterConfig

logger = logging.getLogger(__name__)


@dataclass
class CombinedCorrespondence:
    """A correspondence from either road or shoreline matching."""

    image_point: Tuple[float, float]  # (x, y) in pixels
    osm_point: Tuple[float, float]  # (x, y) in meters
    distance: float  # Match quality score
    source: str  # 'road' or 'shoreline'
    weight: float  # Confidence weight for RANSAC


class CombinedMatcher:
    """Combine road and shoreline correspondences for unified RANSAC.

    Shorelines are typically weighted higher than roads because they are
    more temporally stable (coastlines rarely change over 50-90 years,
    while roads may have been built, removed, or realigned).
    """

    def __init__(
        self,
        road_weight: float = 1.0,
        shoreline_weight: Optional[float] = None,
        water_config: Optional[WaterConfig] = None,
    ):
        """Initialize the combined matcher.

        Args:
            road_weight: Weight for road correspondences.
            shoreline_weight: Weight for shoreline correspondences.
            water_config: Water configuration (for default weights).
        """
        self.road_weight = road_weight
        self.water_config = water_config or WaterConfig()
        self.shoreline_weight = shoreline_weight or self.water_config.shoreline_weight

    def merge_correspondences(
        self,
        road_candidates: List[Tuple[int, int, float]],
        shoreline_correspondences: List[ShorelineCorrespondence],
        image_graph: nx.Graph,
        osm_graph: nx.Graph,
    ) -> List[CombinedCorrespondence]:
        """Merge road and shoreline correspondences.

        Args:
            road_candidates: Road correspondences as (img_node, osm_node, distance).
            shoreline_correspondences: Shoreline correspondences.
            image_graph: Image road graph (for node coordinates).
            osm_graph: OSM road graph (for node coordinates).

        Returns:
            List of CombinedCorrespondence objects.
        """
        combined = []

        # Convert road correspondences to combined format
        for img_node, osm_node, dist in road_candidates:
            try:
                img_data = image_graph.nodes[img_node]
                osm_data = osm_graph.nodes[osm_node]

                # Get coordinates
                img_x = img_data.get("x", img_data.get("pos", (0, 0))[0])
                img_y = img_data.get("y", img_data.get("pos", (0, 0))[1])

                osm_x = osm_data.get("x_meters", osm_data.get("x", 0))
                osm_y = osm_data.get("y_meters", osm_data.get("y", 0))

                combined.append(
                    CombinedCorrespondence(
                        image_point=(img_x, img_y),
                        osm_point=(osm_x, osm_y),
                        distance=dist,
                        source="road",
                        weight=self.road_weight,
                    )
                )
            except (KeyError, TypeError) as e:
                logger.debug(f"Skipping road correspondence: {e}")
                continue

        # Add shoreline correspondences
        for sc in shoreline_correspondences:
            combined.append(
                CombinedCorrespondence(
                    image_point=sc.image_point,
                    osm_point=sc.osm_point,
                    distance=sc.distance,
                    source="shoreline",
                    weight=self.shoreline_weight,
                )
            )

        # Remove spatially conflicting correspondences
        combined = self._remove_conflicting(combined)

        logger.info(
            f"Combined matcher: {len(road_candidates)} road + "
            f"{len(shoreline_correspondences)} shoreline = {len(combined)} total"
        )

        return combined

    def _remove_conflicting(
        self,
        correspondences: List[CombinedCorrespondence],
        min_pixel_distance: float = 10.0,
    ) -> List[CombinedCorrespondence]:
        """Remove spatially conflicting correspondences.

        If two correspondences have similar image points but very different
        OSM points, keep only the one with higher weight (or lower distance).

        Args:
            correspondences: List of combined correspondences.
            min_pixel_distance: Minimum distance between image points to consider distinct.

        Returns:
            Filtered list of correspondences.
        """
        if len(correspondences) <= 1:
            return correspondences

        # Sort by quality (higher weight, lower distance)
        sorted_corr = sorted(
            correspondences,
            key=lambda c: (-c.weight, c.distance),
        )

        kept = []
        kept_img_points = []

        for corr in sorted_corr:
            img_pt = np.array(corr.image_point)

            # Check if too close to an already-kept correspondence
            is_conflict = False
            for kept_pt in kept_img_points:
                dist = np.linalg.norm(img_pt - kept_pt)
                if dist < min_pixel_distance:
                    is_conflict = True
                    break

            if not is_conflict:
                kept.append(corr)
                kept_img_points.append(img_pt)

        return kept

    def to_ransac_format(
        self, correspondences: List[CombinedCorrespondence]
    ) -> List[Tuple[Tuple[float, float], Tuple[float, float], float]]:
        """Convert combined correspondences to RANSAC input format.

        The RANSAC matcher expects tuples of (image_point, osm_point, distance).
        We adjust the distance by the weight to favor higher-weighted correspondences.

        Args:
            correspondences: List of CombinedCorrespondence objects.

        Returns:
            List of (image_point, osm_point, adjusted_distance) tuples.
        """
        return [
            (c.image_point, c.osm_point, c.distance / c.weight)
            for c in correspondences
        ]

    def get_statistics(
        self, correspondences: List[CombinedCorrespondence]
    ) -> dict:
        """Compute statistics about combined correspondences.

        Args:
            correspondences: List of combined correspondences.

        Returns:
            Dictionary of statistics.
        """
        road_count = sum(1 for c in correspondences if c.source == "road")
        shoreline_count = sum(1 for c in correspondences if c.source == "shoreline")

        road_distances = [c.distance for c in correspondences if c.source == "road"]
        shoreline_distances = [c.distance for c in correspondences if c.source == "shoreline"]

        return {
            "total": len(correspondences),
            "road_count": road_count,
            "shoreline_count": shoreline_count,
            "road_fraction": road_count / max(1, len(correspondences)),
            "shoreline_fraction": shoreline_count / max(1, len(correspondences)),
            "avg_road_distance": np.mean(road_distances) if road_distances else 0.0,
            "avg_shoreline_distance": np.mean(shoreline_distances) if shoreline_distances else 0.0,
        }
