"""Main georeferencing pipeline orchestration."""

import numpy as np
import networkx as nx
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import logging

from .config import GeoreferenceConfig, ImageMetadata, PreprocessingConfig
from .preprocessing.enhancement import ImageEnhancer
from .preprocessing.road_extraction import RoadExtractor, RoadExtractionResult
from .graph.image_graph import ImageGraphBuilder
from .graph.osm_graph import OSMGraphBuilder
from .matching.features import FeatureExtractor
from .matching.spectral import SpectralMatcher
from .matching.ransac import RANSACMatcher, RANSACResult
from .output.transform import TransformEstimator, GeoTransform
from .output.quality import QualityAssessor, QualityMetrics
from .output.geotiff import GeoTIFFWriter

logger = logging.getLogger(__name__)


@dataclass
class GeoreferenceResult:
    """Result of georeferencing a single image."""

    success: bool
    confidence_score: float
    transform: Optional[GeoTransform]
    quality: Optional[QualityMetrics]
    output_paths: Dict[str, Path]
    error_message: Optional[str]
    debug_info: Dict[str, Any]


class GeoreferencePipeline:
    """Main pipeline for automatic georeferencing."""

    def __init__(self, config: Optional[GeoreferenceConfig] = None):
        """Initialize the pipeline.

        Args:
            config: Pipeline configuration. Uses defaults if not provided.
        """
        self.config = config or GeoreferenceConfig.default()

        # Initialize components
        self.enhancer = ImageEnhancer(self.config.preprocessing)
        self.road_extractor = RoadExtractor(self.config.preprocessing)
        self.image_graph_builder = ImageGraphBuilder()
        self.osm_graph_builder = OSMGraphBuilder(self.config.osm)
        self.feature_extractor = FeatureExtractor()
        self.spectral_matcher = SpectralMatcher(
            embedding_dim=self.config.matching.spectral_embedding_dim
        )
        self.ransac_matcher = RANSACMatcher(
            threshold_meters=self.config.matching.ransac_threshold_meters,
            max_iterations=self.config.matching.ransac_max_iterations,
            min_inliers=self.config.matching.min_correspondences,
        )
        self.quality_assessor = QualityAssessor(
            min_confidence=self.config.quality.min_confidence_score,
            max_rmse_meters=self.config.quality.max_rmse_meters,
            min_correspondences=self.config.quality.min_matched_roads,
        )
        self.geotiff_writer = GeoTIFFWriter(output_crs=self.config.output.output_crs)

    def process(
        self,
        image_path: Path,
        metadata: ImageMetadata,
        output_dir: Path,
    ) -> GeoreferenceResult:
        """Process a single image through the georeferencing pipeline.

        Args:
            image_path: Path to the input image.
            metadata: Image metadata with approximate coordinates.
            output_dir: Directory for output files.

        Returns:
            GeoreferenceResult with status and outputs.
        """
        debug_info = {}

        try:
            # Step 1: Load and preprocess image
            logger.info(f"Loading image: {image_path}")
            image = self.enhancer.load_image(image_path)
            debug_info["image_shape"] = image.shape

            # Step 2: Extract roads
            logger.info("Extracting road network from image")
            road_result = self.road_extractor.extract_with_retry(image)
            debug_info["road_stats"] = road_result.stats

            if road_result.stats["road_pixels"] < 100:
                logger.warning("Very few roads detected in image")

            # Step 3: Build image graph
            logger.info("Building graph from detected roads")
            image_graph = self.image_graph_builder.build_graph(road_result.skeleton)
            debug_info["image_graph_nodes"] = image_graph.number_of_nodes()
            debug_info["image_graph_edges"] = image_graph.number_of_edges()

            if image_graph.number_of_nodes() < 3:
                return GeoreferenceResult(
                    success=False,
                    confidence_score=0.0,
                    transform=None,
                    quality=None,
                    output_paths={},
                    error_message="Insufficient road junctions detected",
                    debug_info=debug_info,
                )

            # Step 4: Fetch OSM reference data
            logger.info(
                f"Fetching OSM data near ({metadata.latitude}, {metadata.longitude})"
            )

            # Use expanded radius if road coverage is low
            radius = self.config.osm.search_radius_miles
            if road_result.stats.get("is_sparse", False):
                radius = self.config.osm.expanded_radius_miles
                logger.info(f"Using expanded search radius: {radius} miles")

            osm_graph = self.osm_graph_builder.fetch_road_network(
                metadata.latitude, metadata.longitude, radius
            )
            debug_info["osm_graph_nodes"] = osm_graph.number_of_nodes()
            debug_info["osm_graph_edges"] = osm_graph.number_of_edges()

            if osm_graph.number_of_nodes() < 3:
                return GeoreferenceResult(
                    success=False,
                    confidence_score=0.0,
                    transform=None,
                    quality=None,
                    output_paths={},
                    error_message="Insufficient OSM road data in area",
                    debug_info=debug_info,
                )

            # Project OSM to local coordinates
            osm_graph, projection_info = self.osm_graph_builder.project_to_local_crs(
                osm_graph, metadata.latitude, metadata.longitude
            )

            # Apply historical weighting if date is available
            if metadata.date:
                try:
                    year = int(metadata.date[:4])
                    osm_graph = self.osm_graph_builder.estimate_historical_relevance(
                        osm_graph, year
                    )
                except (ValueError, TypeError):
                    pass

            # Step 5: Find candidate correspondences
            logger.info("Finding candidate correspondences")
            candidates = self.feature_extractor.find_candidate_correspondences(
                image_graph, osm_graph
            )
            debug_info["num_candidates"] = len(candidates)

            if len(candidates) < self.config.matching.min_correspondences:
                # Try spectral matching as fallback
                logger.info("Trying spectral matching")
                spectral_candidates = self.spectral_matcher.match_by_embedding(
                    image_graph, osm_graph
                )
                if spectral_candidates:
                    candidates = [
                        (c[0], c[1], c[2]) for c in spectral_candidates
                    ]
                debug_info["spectral_candidates"] = len(candidates)

            if len(candidates) < self.config.matching.min_correspondences:
                return GeoreferenceResult(
                    success=False,
                    confidence_score=0.0,
                    transform=None,
                    quality=None,
                    output_paths={},
                    error_message="Insufficient correspondences found",
                    debug_info=debug_info,
                )

            # Step 6: RANSAC matching
            logger.info("Running RANSAC for robust matching")
            ransac_result = self.ransac_matcher.iterative_ransac(
                image_graph, osm_graph, candidates
            )

            if ransac_result is None:
                return GeoreferenceResult(
                    success=False,
                    confidence_score=0.0,
                    transform=None,
                    quality=None,
                    output_paths={},
                    error_message="RANSAC matching failed",
                    debug_info=debug_info,
                )

            debug_info["ransac_inliers"] = len(ransac_result.inlier_correspondences)
            debug_info["ransac_rmse"] = ransac_result.rmse

            # Step 7: Estimate transformation
            logger.info("Estimating geographic transformation")
            transform_estimator = TransformEstimator(projection_info)

            # Extract point coordinates for transformation
            image_points = []
            geo_points = []
            for img_node, osm_node in ransac_result.inlier_correspondences:
                img_data = image_graph.nodes[img_node]
                osm_data = osm_graph.nodes[osm_node]

                image_points.append([img_data["x"], img_data["y"]])
                geo_points.append(
                    [osm_data.get("x_meters", 0), osm_data.get("y_meters", 0)]
                )

            image_points = np.array(image_points)
            geo_points = np.array(geo_points)

            transform = transform_estimator.estimate_from_correspondences(
                image_points, geo_points, image.shape
            )

            if transform is None:
                return GeoreferenceResult(
                    success=False,
                    confidence_score=0.0,
                    transform=None,
                    quality=None,
                    output_paths={},
                    error_message="Transformation estimation failed",
                    debug_info=debug_info,
                )

            # Step 8: Assess quality
            logger.info("Assessing georeferencing quality")
            residuals = transform_estimator.compute_residuals(
                image_points, geo_points, transform
            )

            quality = self.quality_assessor.assess(
                residuals,
                image_points,
                image.shape,
                ransac_result.inlier_ratio,
            )

            debug_info["quality"] = {
                "confidence": quality.confidence_score,
                "rmse": quality.rmse_meters,
                "coverage": quality.coverage_ratio,
            }

            # Step 9: Generate outputs
            logger.info("Writing output files")
            base_name = image_path.stem
            source_info = {
                "source_path": str(metadata.source_path),
                "original_lat": metadata.latitude,
                "original_lon": metadata.longitude,
                "date": metadata.date,
                "description": metadata.description,
            }

            correspondences = [
                {
                    "image_node": int(img_node),
                    "osm_node": int(osm_node),
                    "image_x": float(image_graph.nodes[img_node]["x"]),
                    "image_y": float(image_graph.nodes[img_node]["y"]),
                }
                for img_node, osm_node in ransac_result.inlier_correspondences
            ]

            output_paths = self.geotiff_writer.write_all(
                image,
                transform,
                quality,
                output_dir,
                base_name,
                source_info,
                correspondences,
                write_worldfile=self.config.output.write_worldfile,
            )

            return GeoreferenceResult(
                success=quality.passed_threshold,
                confidence_score=quality.confidence_score,
                transform=transform,
                quality=quality,
                output_paths=output_paths,
                error_message=None if quality.passed_threshold else "Low confidence",
                debug_info=debug_info,
            )

        except Exception as e:
            logger.exception(f"Error processing {image_path}: {e}")
            return GeoreferenceResult(
                success=False,
                confidence_score=0.0,
                transform=None,
                quality=None,
                output_paths={},
                error_message=str(e),
                debug_info=debug_info,
            )

    def process_directory(
        self, input_dir: Path, output_dir: Path
    ) -> Tuple[Path, ...]:
        """Process a single item directory (e.g., data/000001/).

        Args:
            input_dir: Directory containing image and metadata files.
            output_dir: Directory for output files.

        Returns:
            GeoreferenceResult.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        # Find image file
        image_path = input_dir / "image_native.tif"
        if not image_path.exists():
            image_path = input_dir / "image_medium.jpg"

        if not image_path.exists():
            raise FileNotFoundError(f"No image file found in {input_dir}")

        # Load metadata
        coords_path = input_dir / "coordinates.json"
        metadata_path = input_dir / "metadata.json"

        if not coords_path.exists():
            raise FileNotFoundError(f"No coordinates.json found in {input_dir}")

        metadata = ImageMetadata.from_json_files(coords_path, metadata_path)

        # Process
        return self.process(image_path, metadata, output_dir)
