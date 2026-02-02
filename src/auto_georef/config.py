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
    min_correspondences: int = 4
    min_correspondences_sparse: int = 3  # For low road coverage images

    # RANSAC parameters
    ransac_threshold_meters: float = 10.0
    ransac_max_iterations: int = 1000
    ransac_confidence: float = 0.99

    # Spectral matching
    spectral_embedding_dim: int = 10

    # Scale/rotation search ranges
    scale_range: Tuple[float, float] = (0.5, 2.0)
    scale_steps: int = 10
    rotation_range_deg: Tuple[float, float] = (-30.0, 30.0)
    rotation_steps: int = 12

    # Feature matching tolerances
    angle_tolerance_deg: float = 15.0
    length_ratio_tolerance: float = 0.3
    degree_must_match: bool = True


@dataclass
class QualityConfig:
    """Configuration for quality assessment."""

    # Confidence thresholds
    min_confidence_score: float = 0.5
    high_confidence_threshold: float = 0.8

    # Error thresholds
    max_residual_error_meters: float = 10.0
    max_rmse_meters: float = 15.0

    # Coverage requirements
    min_matched_roads: int = 3
    min_coverage_ratio: float = 0.3  # Matched area / image area
    min_distribution_score: float = 0.5


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
class GeoreferenceConfig:
    """Main configuration combining all sub-configs."""

    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    osm: OSMConfig = field(default_factory=OSMConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

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
