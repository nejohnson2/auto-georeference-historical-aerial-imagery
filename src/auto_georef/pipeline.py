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
from .preprocessing.water_extraction import WaterExtractor, WaterExtractionResult
from .graph.image_graph import ImageGraphBuilder
from .graph.osm_graph import OSMGraphBuilder
from .graph.osm_water import OSMWaterFetcher, OSMWaterResult
from .matching.features import FeatureExtractor
from .matching.spectral import SpectralMatcher
from .matching.ransac import RANSACMatcher, RANSACResult
from .matching.shoreline_matcher import ShorelineMatcher
from .matching.combined_matcher import CombinedMatcher
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

        # Initialize water feature components (if enabled)
        self.water_extractor = None
        self.osm_water_fetcher = None
        self.shoreline_matcher = None
        self.combined_matcher = None

        if self.config.water.enabled:
            self.water_extractor = WaterExtractor(self.config.water)
            self.osm_water_fetcher = OSMWaterFetcher(
                self.config.water, self.config.osm
            )
            self.shoreline_matcher = ShorelineMatcher(
                self.config.water, self.config.matching
            )
            self.combined_matcher = CombinedMatcher(
                road_weight=1.0,
                shoreline_weight=self.config.water.shoreline_weight,
                water_config=self.config.water,
            )

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

            # Step 2b: Extract water features (if enabled)
            water_result = None
            if self.config.water.enabled and self.water_extractor:
                logger.info("Extracting water bodies and shorelines")
                try:
                    water_result = self.water_extractor.extract(image)
                    debug_info["water_stats"] = water_result.stats
                    if water_result.stats["has_significant_water"]:
                        logger.info(
                            f"Found {water_result.stats['num_water_bodies']} water bodies, "
                            f"{water_result.stats['water_coverage_ratio']:.1%} water coverage"
                        )
                except Exception as e:
                    logger.warning(f"Water extraction failed: {e}")

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

            # Step 4b: Fetch OSM water features (if enabled)
            osm_water = None
            if self.config.water.enabled and self.osm_water_fetcher:
                logger.info("Fetching OSM water features")
                try:
                    osm_water = self.osm_water_fetcher.fetch_water_features(
                        metadata.latitude, metadata.longitude, radius
                    )
                    if osm_water.stats["total_features"] > 0:
                        osm_water, _ = self.osm_water_fetcher.project_to_local_crs(
                            osm_water, metadata.latitude, metadata.longitude
                        )
                        debug_info["osm_water_stats"] = osm_water.stats
                        logger.info(
                            f"Found {osm_water.stats['num_coastlines']} coastlines, "
                            f"{osm_water.stats['num_lakes']} lakes, "
                            f"{osm_water.stats['num_rivers']} rivers"
                        )
                    else:
                        osm_water = None
                except Exception as e:
                    logger.warning(f"OSM water fetch failed: {e}")
                    osm_water = None

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

            # Step 5b: Find shoreline correspondences (if water features available)
            shoreline_correspondences = []
            if (
                self.config.water.enabled
                and water_result
                and osm_water
                and water_result.shoreline_contours
                and osm_water.combined_shoreline_meters
                and self.shoreline_matcher
            ):
                logger.info("Matching shorelines")
                try:
                    shoreline_correspondences = self.shoreline_matcher.match(
                        water_result.shoreline_contours,
                        osm_water.combined_shoreline_meters,
                    )
                    debug_info["shoreline_correspondences"] = len(shoreline_correspondences)
                    if shoreline_correspondences:
                        logger.info(f"Found {len(shoreline_correspondences)} shoreline correspondences")
                except Exception as e:
                    logger.warning(f"Shoreline matching failed: {e}")

            # Step 5c: Combine road and shoreline correspondences
            combined_candidates = None
            if shoreline_correspondences and self.combined_matcher:
                logger.info("Combining road and shoreline correspondences")
                combined_corr = self.combined_matcher.merge_correspondences(
                    candidates, shoreline_correspondences, image_graph, osm_graph
                )
                debug_info["combined_stats"] = self.combined_matcher.get_statistics(combined_corr)

                # Convert to RANSAC format (point tuples instead of node IDs)
                combined_candidates = self.combined_matcher.to_ransac_format(combined_corr)
                debug_info["combined_correspondences"] = len(combined_candidates)

            # Check if we have enough correspondences (road + shoreline combined)
            total_correspondences = len(candidates) + len(shoreline_correspondences)
            if total_correspondences < self.config.matching.min_correspondences:
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

            # Track whether we're using combined or road-only correspondences
            using_combined = False
            ransac_result = None

            # Try combined correspondences first (if available)
            if combined_candidates and len(combined_candidates) >= self.config.matching.min_correspondences:
                logger.info(f"Trying RANSAC with {len(combined_candidates)} combined (road + shoreline) correspondences")
                ransac_result = self.ransac_matcher.ransac_from_points(combined_candidates)
                if ransac_result:
                    using_combined = True
                    debug_info["ransac_source"] = "combined"

            # Fall back to road-only correspondences
            if ransac_result is None:
                logger.info("Trying RANSAC with road correspondences only")
                ransac_result = self.ransac_matcher.iterative_ransac(
                    image_graph, osm_graph, candidates
                )
                if ransac_result:
                    debug_info["ransac_source"] = "road_only"

            # If RANSAC fails, try scale/rotation hypothesis search
            if ransac_result is None:
                logger.info("RANSAC failed, trying scale/rotation hypothesis search")
                hypotheses = self.spectral_matcher.match_with_scale_rotation(
                    image_graph,
                    osm_graph,
                    projection_info,
                    scale_range=self.config.matching.scale_range,
                    scale_steps=self.config.matching.scale_steps,
                    rotation_range=self.config.matching.rotation_range_deg,
                    rotation_steps=self.config.matching.rotation_steps,
                )

                if hypotheses:
                    best = hypotheses[0]
                    logger.info(
                        f"Best hypothesis: scale={best['scale']:.3f}, "
                        f"rotation={best['rotation']:.1f}°, "
                        f"matches={best['num_matches']}"
                    )
                    debug_info["hypothesis_search"] = {
                        "num_hypotheses": len(hypotheses),
                        "best_scale": best["scale"],
                        "best_rotation": best["rotation"],
                        "best_matches": best["num_matches"],
                    }

                    # Try RANSAC on the best hypothesis candidates
                    best_candidates = best["correspondences"]
                    ransac_result = self.ransac_matcher.iterative_ransac(
                        image_graph, osm_graph, best_candidates
                    )
                    if ransac_result:
                        debug_info["ransac_source"] = "hypothesis_search"

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

            if using_combined and combined_candidates:
                # For combined correspondences, inlier indices refer to combined_candidates
                for idx, _ in ransac_result.inlier_correspondences:
                    if idx < len(combined_candidates):
                        img_pt, osm_pt, _ = combined_candidates[idx]
                        image_points.append(list(img_pt))
                        geo_points.append(list(osm_pt))
            else:
                # For road-only correspondences, use node IDs
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

    def _find_native_image(self, input_dir: Path) -> Optional[Path]:
        """Find the native (high-resolution) image file in a directory.

        Searches for files containing 'native' in the name with common image
        extensions (.tif, .tiff, .jpg, .jpeg, .png).

        Args:
            input_dir: Directory to search.

        Returns:
            Path to native image if found, None otherwise.
        """
        input_dir = Path(input_dir)
        extensions = [".tif", ".tiff", ".jpg", ".jpeg", ".png"]

        for ext in extensions:
            # Try exact pattern first
            candidates = list(input_dir.glob(f"*native*{ext}"))
            if candidates:
                return candidates[0]

            # Also try case-insensitive on extension
            candidates = list(input_dir.glob(f"*native*{ext.upper()}"))
            if candidates:
                return candidates[0]

        return None

    def _find_fallback_image(self, input_dir: Path) -> Optional[Path]:
        """Find any suitable image file as fallback.

        Args:
            input_dir: Directory to search.

        Returns:
            Path to image if found, None otherwise.
        """
        input_dir = Path(input_dir)
        extensions = [".tif", ".tiff", ".jpg", ".jpeg", ".png"]

        for ext in extensions:
            candidates = list(input_dir.glob(f"*{ext}")) + list(input_dir.glob(f"*{ext.upper()}"))
            # Prefer non-thumbnail images
            for c in candidates:
                if "thumbnail" not in c.name.lower():
                    return c

        return None

    def process_directory(
        self, input_dir: Path, output_dir: Path
    ) -> "GeoreferenceResult":
        """Process a single item directory (e.g., data/000001/).

        Args:
            input_dir: Directory containing image and metadata files.
            output_dir: Directory for output files.

        Returns:
            GeoreferenceResult.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        # Find image file - always prefer native (high-resolution) images
        image_path = self._find_native_image(input_dir)
        if image_path is None:
            image_path = self._find_fallback_image(input_dir)

        if image_path is None:
            raise FileNotFoundError(f"No image file found in {input_dir}")

        logger.info(f"Using image: {image_path.name}")

        # Load metadata
        coords_path = input_dir / "coordinates.json"
        metadata_path = input_dir / "metadata.json"

        if not coords_path.exists():
            raise FileNotFoundError(f"No coordinates.json found in {input_dir}")

        metadata = ImageMetadata.from_json_files(coords_path, metadata_path)

        # Use directory name as base name for outputs (e.g., "000001")
        base_name = input_dir.name

        # Process with custom base name
        return self.process_with_name(image_path, metadata, output_dir, base_name)

    def process_with_name(
        self,
        image_path: Path,
        metadata: ImageMetadata,
        output_dir: Path,
        base_name: str,
    ) -> "GeoreferenceResult":
        """Process a single image with a specified output base name.

        Args:
            image_path: Path to the input image.
            metadata: Image metadata with approximate coordinates.
            output_dir: Directory for output files.
            base_name: Base name for output files.

        Returns:
            GeoreferenceResult with status and outputs.
        """
        # Run the processing pipeline
        result = self.process(image_path, metadata, output_dir)

        # If successful, rename outputs to use the specified base name
        if result.success and result.output_paths:
            # The process method already wrote files with image_path.stem
            # We need to rename them to use base_name
            old_stem = image_path.stem
            if old_stem != base_name:
                new_output_paths = {}
                for key, old_path in result.output_paths.items():
                    if old_path.exists():
                        new_name = old_path.name.replace(old_stem, base_name)
                        new_path = old_path.parent / new_name
                        old_path.rename(new_path)
                        new_output_paths[key] = new_path
                    else:
                        new_output_paths[key] = old_path

                # Return result with updated paths
                return GeoreferenceResult(
                    success=result.success,
                    confidence_score=result.confidence_score,
                    transform=result.transform,
                    quality=result.quality,
                    output_paths=new_output_paths,
                    error_message=result.error_message,
                    debug_info=result.debug_info,
                )

        return result
