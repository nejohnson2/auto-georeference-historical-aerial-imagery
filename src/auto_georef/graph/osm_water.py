"""Fetch and process OpenStreetMap water features (coastlines, lakes, rivers)."""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import hashlib
import logging

from ..config import WaterConfig, OSMConfig

logger = logging.getLogger(__name__)


@dataclass
class OSMWaterResult:
    """Result of fetching OSM water features."""

    coastlines: List[np.ndarray]  # List of (N, 2) arrays of (lon, lat) points
    lakes: List[np.ndarray]  # List of (N, 2) arrays (polygon boundaries)
    rivers: List[np.ndarray]  # List of (N, 2) arrays (river centerlines)
    combined_shoreline_meters: List[np.ndarray] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


class OSMWaterFetcher:
    """Fetch water features from OpenStreetMap.

    Uses osmnx to fetch:
    - Coastlines (natural=coastline)
    - Lakes/ponds (natural=water)
    - Rivers (waterway=river|stream|canal)
    """

    # Conversion factors
    MILES_TO_METERS = 1609.34

    def __init__(
        self,
        water_config: Optional[WaterConfig] = None,
        osm_config: Optional[OSMConfig] = None,
    ):
        """Initialize the OSM water fetcher.

        Args:
            water_config: Water configuration.
            osm_config: OSM configuration (for caching, radius).
        """
        self.water_config = water_config or WaterConfig()
        self.osm_config = osm_config or OSMConfig()
        self._cache: dict = {}

    def _get_cache_key(self, lat: float, lon: float, radius_miles: float) -> str:
        """Generate a cache key for a location query."""
        lat_rounded = round(lat, 4)
        lon_rounded = round(lon, 4)
        radius_rounded = round(radius_miles, 2)
        key_str = f"water_{lat_rounded}_{lon_rounded}_{radius_rounded}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def fetch_water_features(
        self, lat: float, lon: float, radius_miles: Optional[float] = None
    ) -> OSMWaterResult:
        """Fetch water features from OSM within a radius of a point.

        Args:
            lat: Latitude of center point.
            lon: Longitude of center point.
            radius_miles: Search radius in miles.

        Returns:
            OSMWaterResult with coastlines, lakes, and rivers.
        """
        radius_miles = radius_miles or self.osm_config.search_radius_miles
        radius_meters = radius_miles * self.MILES_TO_METERS

        # Check cache
        cache_key = self._get_cache_key(lat, lon, radius_miles)
        if self.osm_config.use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        coastlines = []
        lakes = []
        rivers = []

        try:
            import osmnx as ox

            # Fetch coastlines
            if self.water_config.fetch_coastlines:
                coastlines = self._fetch_coastlines(lat, lon, radius_meters)

            # Fetch lakes
            if self.water_config.fetch_lakes:
                lakes = self._fetch_lakes(lat, lon, radius_meters)

            # Fetch rivers
            if self.water_config.fetch_rivers:
                rivers = self._fetch_rivers(lat, lon, radius_meters)

        except Exception as e:
            logger.warning(f"Failed to fetch OSM water data: {e}")

        # Compute stats
        stats = {
            "num_coastlines": len(coastlines),
            "num_lakes": len(lakes),
            "num_rivers": len(rivers),
            "total_features": len(coastlines) + len(lakes) + len(rivers),
            "total_coastline_points": sum(len(c) for c in coastlines),
            "total_lake_boundary_points": sum(len(l) for l in lakes),
        }

        result = OSMWaterResult(
            coastlines=coastlines,
            lakes=lakes,
            rivers=rivers,
            stats=stats,
        )

        # Cache result
        if self.osm_config.use_cache:
            self._cache[cache_key] = result

        return result

    def _fetch_coastlines(
        self, lat: float, lon: float, radius_meters: float
    ) -> List[np.ndarray]:
        """Fetch coastlines from OSM.

        Args:
            lat: Center latitude.
            lon: Center longitude.
            radius_meters: Search radius.

        Returns:
            List of coastline contours as (N, 2) arrays.
        """
        import osmnx as ox

        coastlines = []

        try:
            # Fetch features with natural=coastline tag
            gdf = ox.features_from_point(
                (lat, lon),
                tags={"natural": "coastline"},
                dist=radius_meters,
            )

            if gdf is not None and len(gdf) > 0:
                for _, row in gdf.iterrows():
                    geom = row.geometry
                    coords = self._extract_coordinates(geom)
                    if coords is not None and len(coords) > 2:
                        coastlines.append(coords)

        except Exception as e:
            logger.debug(f"No coastlines found: {e}")

        return coastlines

    def _fetch_lakes(
        self, lat: float, lon: float, radius_meters: float
    ) -> List[np.ndarray]:
        """Fetch lakes and water bodies from OSM.

        Args:
            lat: Center latitude.
            lon: Center longitude.
            radius_meters: Search radius.

        Returns:
            List of lake boundaries as (N, 2) arrays.
        """
        import osmnx as ox

        lakes = []
        min_area = self.water_config.min_lake_area_m2

        try:
            # Fetch features with natural=water tag
            gdf = ox.features_from_point(
                (lat, lon),
                tags={"natural": "water"},
                dist=radius_meters,
            )

            if gdf is not None and len(gdf) > 0:
                for _, row in gdf.iterrows():
                    geom = row.geometry

                    # Check minimum area for polygons
                    if hasattr(geom, "area"):
                        # Approximate area in m^2 (very rough at this scale)
                        area_approx = geom.area * (111320 ** 2) * np.cos(np.radians(lat))
                        if area_approx < min_area:
                            continue

                    coords = self._extract_coordinates(geom)
                    if coords is not None and len(coords) > 2:
                        lakes.append(coords)

        except Exception as e:
            logger.debug(f"No lakes found: {e}")

        return lakes

    def _fetch_rivers(
        self, lat: float, lon: float, radius_meters: float
    ) -> List[np.ndarray]:
        """Fetch rivers and waterways from OSM.

        Args:
            lat: Center latitude.
            lon: Center longitude.
            radius_meters: Search radius.

        Returns:
            List of river centerlines as (N, 2) arrays.
        """
        import osmnx as ox

        rivers = []

        try:
            # Fetch features with waterway tags
            gdf = ox.features_from_point(
                (lat, lon),
                tags={"waterway": ["river", "stream", "canal"]},
                dist=radius_meters,
            )

            if gdf is not None and len(gdf) > 0:
                for _, row in gdf.iterrows():
                    geom = row.geometry
                    coords = self._extract_coordinates(geom)
                    if coords is not None and len(coords) > 2:
                        rivers.append(coords)

        except Exception as e:
            logger.debug(f"No rivers found: {e}")

        return rivers

    def _extract_coordinates(self, geometry) -> Optional[np.ndarray]:
        """Extract coordinates from a shapely geometry.

        Handles Point, LineString, Polygon, and Multi* geometries.

        Args:
            geometry: Shapely geometry object.

        Returns:
            (N, 2) array of (lon, lat) coordinates, or None.
        """
        from shapely.geometry import (
            Point, LineString, Polygon,
            MultiPoint, MultiLineString, MultiPolygon
        )

        if geometry is None:
            return None

        try:
            if isinstance(geometry, Point):
                return np.array([[geometry.x, geometry.y]])

            elif isinstance(geometry, LineString):
                return np.array(geometry.coords)

            elif isinstance(geometry, Polygon):
                # Return exterior ring
                return np.array(geometry.exterior.coords)

            elif isinstance(geometry, MultiLineString):
                # Combine all lines
                coords = []
                for line in geometry.geoms:
                    coords.extend(line.coords)
                return np.array(coords) if coords else None

            elif isinstance(geometry, MultiPolygon):
                # Use largest polygon
                largest = max(geometry.geoms, key=lambda p: p.area)
                return np.array(largest.exterior.coords)

            elif isinstance(geometry, MultiPoint):
                return np.array([(p.x, p.y) for p in geometry.geoms])

            else:
                # Try to get coords directly
                if hasattr(geometry, "coords"):
                    return np.array(geometry.coords)
                elif hasattr(geometry, "exterior"):
                    return np.array(geometry.exterior.coords)

        except Exception as e:
            logger.debug(f"Failed to extract coordinates: {e}")

        return None

    def project_to_local_crs(
        self, result: OSMWaterResult, center_lat: float, center_lon: float
    ) -> Tuple[OSMWaterResult, dict]:
        """Project water features to local meter-based CRS.

        Args:
            result: OSMWaterResult with lat/lon coordinates.
            center_lat: Latitude of projection center.
            center_lon: Longitude of projection center.

        Returns:
            Tuple of (projected result, projection info dict).
        """
        # Meters per degree at this latitude
        meters_per_deg_lat = 111320
        meters_per_deg_lon = 111320 * np.cos(np.radians(center_lat))

        def project_coords(coords: np.ndarray) -> np.ndarray:
            """Project lon/lat array to meters."""
            projected = np.zeros_like(coords)
            projected[:, 0] = (coords[:, 0] - center_lon) * meters_per_deg_lon
            projected[:, 1] = (coords[:, 1] - center_lat) * meters_per_deg_lat
            return projected

        # Project all features
        projected_coastlines = [project_coords(c) for c in result.coastlines]
        projected_lakes = [project_coords(l) for l in result.lakes]
        projected_rivers = [project_coords(r) for r in result.rivers]

        # Combine all shorelines (coastlines + lake boundaries)
        combined_shoreline_meters = projected_coastlines + projected_lakes

        projection_info = {
            "center_lat": center_lat,
            "center_lon": center_lon,
            "meters_per_deg_lat": meters_per_deg_lat,
            "meters_per_deg_lon": meters_per_deg_lon,
        }

        projected_result = OSMWaterResult(
            coastlines=projected_coastlines,
            lakes=projected_lakes,
            rivers=projected_rivers,
            combined_shoreline_meters=combined_shoreline_meters,
            stats=result.stats.copy(),
        )

        return projected_result, projection_info

    def get_combined_shorelines(self, result: OSMWaterResult) -> List[np.ndarray]:
        """Get combined list of all shoreline contours.

        Combines coastlines and lake boundaries into a single list.

        Args:
            result: OSMWaterResult.

        Returns:
            List of shoreline contours.
        """
        if result.combined_shoreline_meters:
            return result.combined_shoreline_meters
        return result.coastlines + result.lakes
