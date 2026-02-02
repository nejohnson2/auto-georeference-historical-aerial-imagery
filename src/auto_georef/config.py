"""Configuration dataclasses for the auto-georeferencing pipeline."""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional
from pathlib import Path


@dataclass
class PreprocessingConfig:
    """Configuration for image preprocessing and road extraction."""

    # CLAHE parameters
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)

    # Bilateral filter parameters
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0

    # Canny edge detection parameters
    canny_low_threshold: int = 50
    canny_high_threshold: int = 150

    # Morphological operations
    morph_kernel_size: int = 3
    morph_close_iterations: int = 2

    # Filtering
    min_road_length_px: int = 20
    min_road_coverage_ratio: float = 0.05  # Minimum road pixels / total pixels


@dataclass
class OSMConfig:
    """Configuration for OpenStreetMap data fetching."""

    # Search parameters
    search_radius_miles: float = 1.0
    expanded_radius_miles: float = 2.0  # Used when road coverage is low

    # Road type filtering
    road_types: Tuple[str, ...] = (
        "motorway",
        "trunk",
        "primary",
        "secondary",
        "tertiary",
        "unclassified",
        "residential",
    )
    network_type: str = "drive"

    # Graph simplification
    simplification_tolerance: float = 0.0001  # degrees

    # Caching
    cache_dir: Optional[Path] = None
    use_cache: bool = True


@dataclass
class MatchingConfig:
    """Configuration for graph matching algorithms."""

    # Correspondence requirements
    min_correspondences: int = 3
    min_correspondences_sparse: int = 3  # For low road coverage images

    # RANSAC parameters
    # Note: Historical images need larger threshold due to road network changes
    ransac_threshold_meters: float = 50.0
    ransac_max_iterations: int = 2000
    ransac_confidence: float = 0.99

    # Spectral matching
    spectral_embedding_dim: int = 10

    # Scale/rotation search ranges
    # Wide ranges to handle unknown historical image scale/orientation
    scale_range: Tuple[float, float] = (0.2, 3.0)
    scale_steps: int = 15
    rotation_range_deg: Tuple[float, float] = (-90.0, 90.0)
    rotation_steps: int = 18

    # Feature matching tolerances
    angle_tolerance_deg: float = 20.0
    length_ratio_tolerance: float = 0.4
    degree_must_match: bool = True


@dataclass
class QualityConfig:
    """Configuration for quality assessment."""

    # Confidence thresholds
    # Relaxed for historical images where roads may have changed
    min_confidence_score: float = 0.3
    high_confidence_threshold: float = 0.7

    # Error thresholds
    # Historical images typically have 20-50m accuracy
    max_residual_error_meters: float = 50.0
    max_rmse_meters: float = 50.0

    # Coverage requirements
    # Many historical images have sparse road networks
    min_matched_roads: int = 3
    min_coverage_ratio: float = 0.1  # Matched area / image area
    min_distribution_score: float = 0.3


@dataclass
class OutputConfig:
    """Configuration for output generation."""

    # Coordinate reference system
    output_crs: str = "EPSG:4326"

    # Output files
    write_geotiff: bool = True
    write_worldfile: bool = True
    write_metadata_json: bool = True

    # Visualization
    write_debug_images: bool = False


@dataclass
class WaterConfig:
    """Configuration for water/shoreline detection and matching."""

    # Enable/disable water feature matching
    enabled: bool = True

    # Water detection parameters
    water_intensity_threshold: float = -1.5  # Std devs below median for water
    min_water_area_px: int = 500  # Minimum water body size in pixels
    texture_window_size: int = 15  # Window for local variance calculation
    texture_variance_threshold: float = 100.0  # Max variance for water regions

    # Shoreline extraction
    min_shoreline_length_px: int = 50  # Minimum shoreline segment length
    contour_smoothing_epsilon: float = 0.005  # Fraction of arc length for smoothing

    # OSM water features
    fetch_coastlines: bool = True
    fetch_lakes: bool = True
    fetch_rivers: bool = True
    min_lake_area_m2: float = 1000.0  # Minimum lake area to fetch

    # Matching parameters
    max_shoreline_correspondences: int = 20  # Max points to extract per image
    curvature_window_size: int = 5  # Window for curvature calculation
    shoreline_weight: float = 1.5  # Weight relative to roads in RANSAC


@dataclass
class GeoreferenceConfig:
    """Main configuration combining all sub-configs."""

    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    osm: OSMConfig = field(default_factory=OSMConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    water: WaterConfig = field(default_factory=WaterConfig)

    @classmethod
    def default(cls) -> "GeoreferenceConfig":
        """Create a default configuration."""
        return cls()

    @classmethod
    def from_yaml(cls, path: Path) -> "GeoreferenceConfig":
        """Load configuration from a YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(
            preprocessing=PreprocessingConfig(**data.get("preprocessing", {})),
            osm=OSMConfig(**data.get("osm", {})),
            matching=MatchingConfig(**data.get("matching", {})),
            quality=QualityConfig(**data.get("quality", {})),
            output=OutputConfig(**data.get("output", {})),
            water=WaterConfig(**data.get("water", {})),
        )


@dataclass
class ImageMetadata:
    """Metadata for a single image to be georeferenced."""

    latitude: float
    longitude: float
    confidence: str
    source_path: Path
    date: Optional[str] = None
    description: Optional[str] = None
    coverage: Optional[str] = None

    @classmethod
    def from_json_files(
        cls, coordinates_path: Path, metadata_path: Optional[Path] = None
    ) -> "ImageMetadata":
        """Load metadata from coordinates.json and optionally metadata.json."""
        import json

        with open(coordinates_path) as f:
            coords = json.load(f)

        date = None
        description = None
        coverage = None

        if metadata_path and metadata_path.exists():
            with open(metadata_path) as f:
                meta = json.load(f)
                fields = meta.get("fields", {})
                date = fields.get("Date")
                description = fields.get("Description")
                coverage = fields.get("Coverage")

        return cls(
            latitude=coords["latitude"],
            longitude=coords["longitude"],
            confidence=coords.get("confidence", "unknown"),
            source_path=coordinates_path.parent,
            date=date,
            description=description,
            coverage=coverage,
        )
