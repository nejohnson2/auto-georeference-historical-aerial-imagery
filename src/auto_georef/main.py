"""CLI entry point for auto-georeferencing."""

import click
import logging
from pathlib import Path

from .config import GeoreferenceConfig, ImageMetadata
from .pipeline import GeoreferencePipeline
from .batch import BatchProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def cli(verbose: bool):
    """Auto-Georeference: Automatic georeferencing of historical aerial imagery."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.argument("input_dir", type=click.Path(exists=True, path_type=Path))
@click.argument("output_dir", type=click.Path(path_type=Path))
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration YAML file",
)
def single(input_dir: Path, output_dir: Path, config: Path):
    """Process a single image directory.

    INPUT_DIR: Directory containing image_native.tif and coordinates.json
    OUTPUT_DIR: Directory for output files
    """
    # Load config
    if config:
        georef_config = GeoreferenceConfig.from_yaml(config)
    else:
        georef_config = GeoreferenceConfig.default()

    # Process
    pipeline = GeoreferencePipeline(georef_config)
    result = pipeline.process_directory(input_dir, output_dir)

    # Report result
    if result.success:
        click.echo(click.style("Success!", fg="green"))
        click.echo(f"Confidence: {result.confidence_score:.2%}")
        click.echo("Output files:")
        for name, path in result.output_paths.items():
            click.echo(f"  {name}: {path}")
    else:
        click.echo(click.style("Failed or low confidence", fg="yellow"))
        click.echo(f"Confidence: {result.confidence_score:.2%}")
        if result.error_message:
            click.echo(f"Error: {result.error_message}")
        if result.quality and result.quality.warnings:
            click.echo("Warnings:")
            for warning in result.quality.warnings:
                click.echo(f"  - {warning}")


@cli.command()
@click.argument("input_dir", type=click.Path(exists=True, path_type=Path))
@click.argument("output_dir", type=click.Path(path_type=Path))
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration YAML file",
)
@click.option(
    "--workers",
    "-w",
    default=1,
    type=int,
    help="Number of parallel workers",
)
def batch(input_dir: Path, output_dir: Path, config: Path, workers: int):
    """Process all images in a directory.

    INPUT_DIR: Root directory containing image subdirectories
    OUTPUT_DIR: Directory for output files
    """
    # Load config
    if config:
        georef_config = GeoreferenceConfig.from_yaml(config)
    else:
        georef_config = GeoreferenceConfig.default()

    # Process
    processor = BatchProcessor(georef_config)
    result = processor.process_batch(input_dir, output_dir, num_workers=workers)

    # Report summary
    click.echo("\n" + "=" * 50)
    click.echo("BATCH PROCESSING COMPLETE")
    click.echo("=" * 50)
    click.echo(f"Total images:    {result.total_images}")
    click.echo(click.style(f"Successful:      {result.successful}", fg="green"))
    click.echo(click.style(f"Low confidence:  {result.low_confidence}", fg="yellow"))
    click.echo(click.style(f"Failed:          {result.failed}", fg="red"))
    click.echo(f"Total time:      {result.total_seconds:.1f} seconds")
    click.echo(f"\nReport: {output_dir / 'batch_report.json'}")


@cli.command("debug-roads")
@click.argument("image_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output path for debug image",
)
def debug_roads(image_path: Path, output: Path):
    """Debug road extraction on a single image.

    IMAGE_PATH: Path to the image file
    """
    from .preprocessing.enhancement import ImageEnhancer
    from .preprocessing.road_extraction import RoadExtractor
    from .graph.image_graph import ImageGraphBuilder

    # Load and process
    enhancer = ImageEnhancer()
    extractor = RoadExtractor()
    graph_builder = ImageGraphBuilder()

    image = enhancer.load_image(image_path)
    click.echo(f"Image shape: {image.shape}")

    result = extractor.extract(image)
    click.echo(f"Road extraction stats: {result.stats}")

    graph = graph_builder.build_graph(result.skeleton)
    stats = graph_builder.compute_graph_stats(graph)
    click.echo(f"Graph stats: {stats}")

    # Save debug image if output specified
    if output:
        import cv2

        # Create visualization
        debug_img = cv2.cvtColor(result.enhanced_image, cv2.COLOR_GRAY2BGR)

        # Draw skeleton in red
        skeleton_coords = result.skeleton > 0
        debug_img[skeleton_coords] = [0, 0, 255]

        # Draw nodes
        for node in graph.nodes():
            x = int(graph.nodes[node]["x"])
            y = int(graph.nodes[node]["y"])
            color = (0, 255, 0) if graph.nodes[node].get("is_junction") else (255, 0, 0)
            cv2.circle(debug_img, (x, y), 3, color, -1)

        cv2.imwrite(str(output), debug_img)
        click.echo(f"Debug image saved to: {output}")


@cli.command("visualize")
@click.argument("geotiff_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output HTML file path",
)
@click.option(
    "--opacity",
    default=0.7,
    type=float,
    help="Initial opacity of the overlay (0-1)",
)
@click.option(
    "--result-json",
    type=click.Path(exists=True, path_type=Path),
    help="Path to result JSON for metadata display",
)
@click.option(
    "--open-browser",
    is_flag=True,
    help="Open the visualization in default browser",
)
def visualize(
    geotiff_path: Path,
    output: Path,
    opacity: float,
    result_json: Path,
    open_browser: bool,
):
    """Visualize a georeferenced image on an interactive map.

    GEOTIFF_PATH: Path to the georeferenced GeoTIFF file
    """
    from .output.visualize import create_map_overlay

    # Default output path
    if output is None:
        output = geotiff_path.parent / f"{geotiff_path.stem}_map.html"

    # Try to find result JSON automatically
    if result_json is None:
        auto_json = geotiff_path.parent / f"{geotiff_path.stem.replace('_georef', '_result')}.json"
        if auto_json.exists():
            result_json = auto_json

    click.echo(f"Creating visualization for: {geotiff_path}")

    html_path = create_map_overlay(
        geotiff_path=geotiff_path,
        output_html=output,
        opacity=opacity,
        result_json_path=result_json,
    )

    click.echo(click.style(f"Visualization saved to: {html_path}", fg="green"))

    if open_browser:
        import webbrowser
        webbrowser.open(f"file://{html_path.absolute()}")
        click.echo("Opened in browser")
    else:
        click.echo("Open the HTML file in a browser to view the map")


@cli.command("debug-osm")
@click.argument("lat", type=float)
@click.argument("lon", type=float)
@click.option("--radius", "-r", default=1.0, type=float, help="Radius in miles")
def debug_osm(lat: float, lon: float, radius: float):
    """Debug OSM data fetching for a location.

    LAT: Latitude
    LON: Longitude
    """
    from .graph.osm_graph import OSMGraphBuilder

    builder = OSMGraphBuilder()

    click.echo(f"Fetching OSM data near ({lat}, {lon}) with {radius} mile radius...")

    graph = builder.fetch_road_network(lat, lon, radius)
    stats = builder.compute_graph_stats(graph)

    click.echo(f"OSM graph stats: {stats}")

    if graph.number_of_edges() > 0:
        # Show sample roads
        click.echo("\nSample roads:")
        for i, (u, v, data) in enumerate(graph.edges(data=True)):
            if i >= 5:
                break
            name = data.get("name", "Unnamed")
            highway = data.get("highway", "unknown")
            length = data.get("length", 0)
            click.echo(f"  {name} ({highway}): {length:.0f}m")


@cli.command("tune")
@click.argument("input_dir", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration YAML file",
)
def tune(input_dir: Path, config: Path):
    """Analyze an image and suggest parameter tuning.

    INPUT_DIR: Directory containing image_native.tif and coordinates.json
    """
    import json
    import numpy as np
    from .preprocessing.enhancement import ImageEnhancer
    from .preprocessing.road_extraction import RoadExtractor
    from .graph.image_graph import ImageGraphBuilder
    from .graph.osm_graph import OSMGraphBuilder
    from .matching.features import FeatureExtractor
    from .matching.spectral import SpectralMatcher

    # Load config
    if config:
        georef_config = GeoreferenceConfig.from_yaml(config)
    else:
        georef_config = GeoreferenceConfig.default()

    click.echo(f"\n{'='*60}")
    click.echo(f"Tuning Analysis: {input_dir}")
    click.echo(f"{'='*60}\n")

    # Load coordinates
    coords_file = input_dir / "coordinates.json"
    with open(coords_file) as f:
        coords = json.load(f)
    lat = coords.get("latitude", coords.get("lat"))
    lon = coords.get("longitude", coords.get("lon"))

    # Initialize components
    enhancer = ImageEnhancer(georef_config.preprocessing)
    extractor = RoadExtractor(georef_config.preprocessing)
    graph_builder = ImageGraphBuilder()
    osm_builder = OSMGraphBuilder(georef_config.osm)
    feature_extractor = FeatureExtractor()
    spectral_matcher = SpectralMatcher()

    # Load and process image
    image_file = input_dir / "image_native.tif"
    if not image_file.exists():
        image_file = input_dir / "image_medium.jpg"

    click.echo("1. Analyzing image...")
    image = enhancer.load_image(image_file)
    road_result = extractor.extract(image)

    click.echo(f"   Image size: {image.shape[1]} x {image.shape[0]}")
    click.echo(f"   Road pixels: {road_result.stats['road_pixels']}")
    click.echo(f"   Road coverage: {road_result.stats['road_coverage_ratio']:.2%}")

    # Build graph
    click.echo("\n2. Analyzing road graph...")
    image_graph = graph_builder.build_graph(road_result.skeleton)
    img_stats = graph_builder.compute_graph_stats(image_graph)

    click.echo(f"   Nodes: {img_stats['num_nodes']}")
    click.echo(f"   Edges: {img_stats['num_edges']}")
    click.echo(f"   Junctions (degree>=3): {img_stats['num_junctions']}")

    # Degree distribution
    degrees = [image_graph.degree(n) for n in image_graph.nodes()]
    degree_counts = {}
    for d in degrees:
        degree_counts[d] = degree_counts.get(d, 0) + 1

    click.echo("   Degree distribution:")
    for d in sorted(degree_counts.keys())[:6]:
        click.echo(f"     Degree {d}: {degree_counts[d]} nodes")

    # Fetch OSM data
    click.echo("\n3. Analyzing OSM data...")
    osm_graph = osm_builder.fetch_road_network(lat, lon, radius_miles=2.0)
    osm_graph, proj_info = osm_builder.project_to_local_crs(osm_graph, lat, lon)
    osm_stats = osm_builder.compute_graph_stats(osm_graph)

    click.echo(f"   OSM nodes: {osm_stats['num_nodes']}")
    click.echo(f"   OSM edges: {osm_stats['num_edges']}")

    # Try matching
    click.echo("\n4. Testing feature matching...")
    candidates = feature_extractor.find_candidate_correspondences(
        image_graph, osm_graph, min_degree=2
    )

    unique_img = len(set(c[0] for c in candidates))
    unique_osm = len(set(c[1] for c in candidates))
    click.echo(f"   Candidates (min_degree=2): {len(candidates)}")
    click.echo(f"   Unique image nodes: {unique_img}")
    click.echo(f"   Unique OSM nodes: {unique_osm}")

    # Also try with min_degree=1
    candidates_d1 = feature_extractor.find_candidate_correspondences(
        image_graph, osm_graph, min_degree=1
    )
    click.echo(f"   Candidates (min_degree=1): {len(candidates_d1)}")

    # Try scale/rotation hypothesis
    click.echo("\n5. Testing scale/rotation hypotheses...")
    hypotheses = spectral_matcher.match_with_scale_rotation(
        image_graph, osm_graph, proj_info,
        scale_range=(0.3, 3.0),
        scale_steps=15,
        rotation_range=(-90, 90),
        rotation_steps=18,
    )

    if hypotheses:
        click.echo(f"   Found {len(hypotheses)} valid hypotheses")
        click.echo("   Top 3 hypotheses:")
        for i, h in enumerate(hypotheses[:3]):
            click.echo(
                f"     {i+1}. scale={h['scale']:.2f} m/px, "
                f"rotation={h['rotation']:.0f}°, "
                f"matches={h['num_matches']}, "
                f"quality={h['quality']:.2f}"
            )
    else:
        click.echo("   No valid hypotheses found")

    # Generate recommendations
    click.echo(f"\n{'='*60}")
    click.echo("RECOMMENDATIONS:")
    click.echo(f"{'='*60}\n")

    issues = []
    recommendations = []

    # Check road coverage
    if road_result.stats['road_coverage_ratio'] < 0.02:
        issues.append("Very low road coverage (<2%)")
        recommendations.append("Image may be mostly water/vegetation - try adjusting canny thresholds")
        recommendations.append("  canny_low_threshold: 30 (lower = more edges)")
        recommendations.append("  canny_high_threshold: 100")

    # Check graph quality
    if img_stats['num_junctions'] == 0:
        issues.append("No road intersections detected")
        recommendations.append("Road extraction is fragmented - try:")
        recommendations.append("  morph_kernel_size: 5 (larger = more connected)")
        recommendations.append("  morph_close_iterations: 3")

    if img_stats['num_edges'] < 20:
        issues.append(f"Very few road segments ({img_stats['num_edges']})")
        recommendations.append("Increase morphological closing to connect fragments")

    # Check matching
    if unique_osm < 5:
        issues.append(f"Only {unique_osm} unique OSM nodes matched")
        recommendations.append("Try relaxing matching constraints:")
        recommendations.append("  degree_must_match: false")
        recommendations.append("  angle_tolerance_deg: 30")

    if len(candidates) < 10:
        issues.append("Very few matching candidates")
        recommendations.append("Expand search radius or relax matching")

    # Check hypotheses
    if hypotheses and hypotheses[0]['num_matches'] >= 5:
        best = hypotheses[0]
        recommendations.append(f"Scale/rotation search found a good match!")
        recommendations.append(f"  Estimated scale: {best['scale']:.2f} m/pixel")
        recommendations.append(f"  Estimated rotation: {best['rotation']:.0f}°")
    elif not hypotheses:
        issues.append("No valid scale/rotation hypotheses")
        recommendations.append("The road pattern may not match modern OSM roads")
        recommendations.append("Try expanding the search ranges:")
        recommendations.append("  scale_range: [0.2, 5.0]")
        recommendations.append("  rotation_range_deg: [-180, 180]")

    if issues:
        click.echo(click.style("Issues found:", fg="yellow"))
        for issue in issues:
            click.echo(f"  - {issue}")
        click.echo()

    if recommendations:
        click.echo(click.style("Suggested changes to config.yaml:", fg="cyan"))
        for rec in recommendations:
            click.echo(f"  {rec}")
    else:
        click.echo(click.style("No specific issues found - try running with default config", fg="green"))


@cli.command("diagnose")
@click.argument("subcommand", type=click.Choice(["geotiff", "transform", "matching", "steps"]))
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output directory for steps")
def diagnose(subcommand: str, path: Path, output: Path):
    """Run diagnostic checks on georeferencing results.

    SUBCOMMAND: One of 'geotiff', 'transform', 'matching', 'steps'
    PATH: Path to file or directory to diagnose
    """
    from .diagnostics import check_geotiff, check_transform, check_matching, visualize_steps

    ctx = click.Context(click.Command("dummy"))

    if subcommand == "geotiff":
        ctx.invoke(check_geotiff, geotiff_path=path)
    elif subcommand == "transform":
        ctx.invoke(check_transform, result_json=path)
    elif subcommand == "matching":
        ctx.invoke(check_matching, input_dir=path)
    elif subcommand == "steps":
        if output is None:
            output = Path("diagnostic_output")
        ctx.invoke(visualize_steps, input_dir=path, output_dir=output)


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
