"""Diagnostic tools for evaluating georeferencing intermediate steps."""

import json
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

import click


@click.group()
def diagnostics():
    """Diagnostic commands for evaluating georeferencing."""
    pass


@diagnostics.command("check-geotiff")
@click.argument("geotiff_path", type=click.Path(exists=True, path_type=Path))
def check_geotiff(geotiff_path: Path):
    """Inspect a GeoTIFF file's georeferencing parameters."""
    import rasterio
    import math

    click.echo(f"\n{'='*60}")
    click.echo(f"GeoTIFF Diagnostic: {geotiff_path}")
    click.echo(f"{'='*60}\n")

    with rasterio.open(geotiff_path) as src:
        click.echo(f"CRS: {src.crs}")
        click.echo(f"Shape: {src.height} x {src.width} pixels")
        click.echo(f"\nRasterio Transform:")
        click.echo(f"  {src.transform}")

        bounds = src.bounds
        click.echo(f"\nBounds:")
        click.echo(f"  Left (min lon):   {bounds.left:.8f}")
        click.echo(f"  Right (max lon):  {bounds.right:.8f}")
        click.echo(f"  Bottom (min lat): {bounds.bottom:.8f}")
        click.echo(f"  Top (max lat):    {bounds.top:.8f}")

        # Calculate spans
        lon_span = bounds.right - bounds.left
        lat_span = bounds.top - bounds.bottom

        click.echo(f"\nGeographic Span:")
        click.echo(f"  Longitude: {lon_span:.8f} degrees")
        click.echo(f"  Latitude:  {lat_span:.8f} degrees")

        # Convert to meters
        lat_mid = (bounds.bottom + bounds.top) / 2
        meters_per_deg_lon = 111320 * math.cos(math.radians(lat_mid))
        meters_per_deg_lat = 111320

        width_m = abs(lon_span) * meters_per_deg_lon
        height_m = abs(lat_span) * meters_per_deg_lat

        click.echo(f"\nApproximate Physical Size:")
        click.echo(f"  Width:  {width_m:.1f} meters")
        click.echo(f"  Height: {height_m:.1f} meters")

        # Pixel resolution
        pixel_res_x = width_m / src.width if src.width > 0 else 0
        pixel_res_y = height_m / src.height if src.height > 0 else 0

        click.echo(f"\nPixel Resolution:")
        click.echo(f"  X: {pixel_res_x:.4f} m/pixel")
        click.echo(f"  Y: {pixel_res_y:.4f} m/pixel")

        # Check for issues
        click.echo(f"\n{'='*60}")
        click.echo("DIAGNOSTIC CHECKS:")
        click.echo(f"{'='*60}")

        issues = []
        if abs(lon_span) < 0.0001:
            issues.append("CRITICAL: Longitude span is too small (< 0.0001 deg)")
        if abs(lat_span) < 0.0001:
            issues.append("CRITICAL: Latitude span is too small (< 0.0001 deg)")
        if width_m < 100:
            issues.append(f"WARNING: Image width ({width_m:.1f}m) seems too small for aerial imagery")
        if height_m < 100:
            issues.append(f"WARNING: Image height ({height_m:.1f}m) seems too small for aerial imagery")
        if pixel_res_x < 0.01:
            issues.append("CRITICAL: Pixel resolution X is essentially zero")
        if pixel_res_y < 0.01:
            issues.append("CRITICAL: Pixel resolution Y is essentially zero")

        # Expected for typical aerial/historical imagery: 0.1-5.0 m/pixel
        if pixel_res_x > 0.01 and (pixel_res_x < 0.05 or pixel_res_x > 5.0):
            issues.append(f"WARNING: Unusual pixel resolution ({pixel_res_x:.4f} m/px) - expected 0.05-5.0 m/px")

        if issues:
            click.echo(click.style("Issues found:", fg="red"))
            for issue in issues:
                click.echo(f"  - {issue}")
        else:
            click.echo(click.style("No issues detected", fg="green"))


@diagnostics.command("check-transform")
@click.argument("result_json", type=click.Path(exists=True, path_type=Path))
def check_transform(result_json: Path):
    """Inspect transformation parameters from result JSON."""
    click.echo(f"\n{'='*60}")
    click.echo(f"Transform Diagnostic: {result_json}")
    click.echo(f"{'='*60}\n")

    with open(result_json) as f:
        data = json.load(f)

    transform = data.get("transformation", {})
    georef = data.get("georeferencing", {})
    correspondences = data.get("correspondences", [])

    click.echo("Transformation Parameters:")
    click.echo(f"  Scale X:     {transform.get('scale_x', 'N/A')} m/pixel")
    click.echo(f"  Scale Y:     {transform.get('scale_y', 'N/A')} m/pixel")
    click.echo(f"  Rotation:    {transform.get('rotation_deg', 'N/A')} degrees")
    click.echo(f"  Origin Lon:  {transform.get('origin_lon', 'N/A')}")
    click.echo(f"  Origin Lat:  {transform.get('origin_lat', 'N/A')}")

    click.echo(f"\nQuality Metrics:")
    click.echo(f"  Confidence:  {georef.get('confidence_score', 'N/A')}")
    click.echo(f"  RMSE:        {georef.get('rmse_meters', 'N/A')} meters")
    click.echo(f"  Correspondences: {georef.get('num_correspondences', 'N/A')}")
    click.echo(f"  Coverage:    {georef.get('coverage_ratio', 'N/A')}")

    # Analyze correspondences
    click.echo(f"\nCorrespondence Analysis:")
    click.echo(f"  Total correspondences: {len(correspondences)}")

    if correspondences:
        osm_nodes = set(c.get("osm_node") for c in correspondences)
        click.echo(f"  Unique OSM nodes matched: {len(osm_nodes)}")

        if len(osm_nodes) < 3:
            click.echo(click.style(
                f"  CRITICAL: Only {len(osm_nodes)} unique OSM nodes! Need at least 3 for proper transformation.",
                fg="red"
            ))

        # Check spatial distribution of image points
        img_xs = [c.get("image_x", 0) for c in correspondences]
        img_ys = [c.get("image_y", 0) for c in correspondences]

        click.echo(f"\n  Image point distribution:")
        click.echo(f"    X range: {min(img_xs):.1f} - {max(img_xs):.1f}")
        click.echo(f"    Y range: {min(img_ys):.1f} - {max(img_ys):.1f}")

        x_span = max(img_xs) - min(img_xs)
        y_span = max(img_ys) - min(img_ys)
        click.echo(f"    X span: {x_span:.1f} pixels")
        click.echo(f"    Y span: {y_span:.1f} pixels")

        if x_span < 100 or y_span < 100:
            click.echo(click.style(
                "  WARNING: Correspondences are clustered in a small area!",
                fg="yellow"
            ))


@diagnostics.command("check-matching")
@click.argument("input_dir", type=click.Path(exists=True, path_type=Path))
def check_matching(input_dir: Path):
    """Debug the matching process for a single image."""
    from .preprocessing.enhancement import ImageEnhancer
    from .preprocessing.road_extraction import RoadExtractor
    from .graph.image_graph import ImageGraphBuilder
    from .graph.osm_graph import OSMGraphBuilder
    from .matching.features import FeatureExtractor
    from .matching.spectral import SpectralMatcher
    from .config import ImageMetadata

    click.echo(f"\n{'='*60}")
    click.echo(f"Matching Diagnostic: {input_dir}")
    click.echo(f"{'='*60}\n")

    # Load metadata
    coords_file = input_dir / "coordinates.json"
    if not coords_file.exists():
        click.echo(click.style("ERROR: coordinates.json not found", fg="red"))
        return

    with open(coords_file) as f:
        coords = json.load(f)

    lat = coords.get("latitude", coords.get("lat"))
    lon = coords.get("longitude", coords.get("lon"))
    click.echo(f"Approximate location: ({lat}, {lon})")

    # Load and process image
    image_file = input_dir / "image_native.tif"
    if not image_file.exists():
        click.echo(click.style("ERROR: image_native.tif not found", fg="red"))
        return

    enhancer = ImageEnhancer()
    extractor = RoadExtractor()
    graph_builder = ImageGraphBuilder()

    click.echo(f"\n1. Loading image...")
    image = enhancer.load_image(image_file)
    click.echo(f"   Image shape: {image.shape}")

    click.echo(f"\n2. Extracting roads...")
    road_result = extractor.extract(image)
    click.echo(f"   Road pixels: {np.sum(road_result.skeleton > 0)}")
    click.echo(f"   Stats: {road_result.stats}")

    click.echo(f"\n3. Building image graph...")
    image_graph = graph_builder.build_graph(road_result.skeleton)
    img_stats = graph_builder.compute_graph_stats(image_graph)
    click.echo(f"   Nodes: {img_stats['num_nodes']}")
    click.echo(f"   Edges: {img_stats['num_edges']}")
    click.echo(f"   Junctions: {img_stats['num_junctions']}")

    click.echo(f"\n4. Fetching OSM data...")
    osm_builder = OSMGraphBuilder()
    osm_graph = osm_builder.fetch_road_network(lat, lon, radius_miles=1.0)
    osm_stats = osm_builder.compute_graph_stats(osm_graph)
    click.echo(f"   Nodes: {osm_stats['num_nodes']}")
    click.echo(f"   Edges: {osm_stats['num_edges']}")

    # Project to local CRS
    osm_graph, proj_info = osm_builder.project_to_local_crs(osm_graph, lat, lon)
    click.echo(f"   Projection center: ({proj_info['center_lat']}, {proj_info['center_lon']})")
    click.echo(f"   Meters/deg lon: {proj_info['meters_per_deg_lon']:.2f}")
    click.echo(f"   Meters/deg lat: {proj_info['meters_per_deg_lat']:.2f}")

    click.echo(f"\n5. Computing features...")
    feature_extractor = FeatureExtractor()
    image_features = feature_extractor.compute_all_node_signatures(image_graph)
    osm_features = feature_extractor.compute_all_node_signatures(osm_graph)
    click.echo(f"   Image features computed: {len(image_features)}")
    click.echo(f"   OSM features computed: {len(osm_features)}")

    click.echo(f"\n6. Running feature matching...")
    # Use feature-based matching
    candidates = feature_extractor.find_candidate_correspondences(
        image_graph, osm_graph
    )
    click.echo(f"   Initial candidates: {len(candidates)}")

    # Analyze candidate quality
    if candidates:
        osm_nodes_in_candidates = set(c[1] for c in candidates)
        click.echo(f"   Unique OSM nodes in candidates: {len(osm_nodes_in_candidates)}")

        # Check if candidates span the image
        img_nodes = [c[0] for c in candidates]
        img_xs = [image_graph.nodes[n].get('x', 0) for n in img_nodes]
        img_ys = [image_graph.nodes[n].get('y', 0) for n in img_nodes]

        click.echo(f"   Candidate image X range: {min(img_xs):.0f} - {max(img_xs):.0f}")
        click.echo(f"   Candidate image Y range: {min(img_ys):.0f} - {max(img_ys):.0f}")

    click.echo(f"\n{'='*60}")
    click.echo("DIAGNOSTIC SUMMARY:")
    click.echo(f"{'='*60}")

    issues = []
    if img_stats['num_nodes'] < 10:
        issues.append("Too few image graph nodes - road extraction may have failed")
    if osm_stats['num_nodes'] < 10:
        issues.append("Too few OSM nodes - check location or expand search radius")
    if len(candidates) < 10:
        issues.append("Too few matching candidates - features may not align")
    if candidates and len(osm_nodes_in_candidates) < 5:
        issues.append(f"Only {len(osm_nodes_in_candidates)} unique OSM nodes matched - poor spatial distribution")

    if issues:
        click.echo(click.style("Issues found:", fg="red"))
        for issue in issues:
            click.echo(f"  - {issue}")
    else:
        click.echo(click.style("No major issues detected", fg="green"))


@diagnostics.command("visualize-steps")
@click.argument("input_dir", type=click.Path(exists=True, path_type=Path))
@click.argument("output_dir", type=click.Path(path_type=Path))
def visualize_steps(input_dir: Path, output_dir: Path):
    """Generate visualization of all intermediate steps."""
    import cv2
    from .preprocessing.enhancement import ImageEnhancer
    from .preprocessing.road_extraction import RoadExtractor
    from .preprocessing.water_extraction import WaterExtractor
    from .graph.image_graph import ImageGraphBuilder
    from .graph.osm_graph import OSMGraphBuilder
    from .graph.osm_water import OSMWaterFetcher
    from .matching.features import FeatureExtractor
    from .matching.spectral import SpectralMatcher
    from .matching.shoreline_matcher import ShorelineMatcher

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"\nGenerating step visualizations in {output_dir}...\n")

    # Load metadata
    coords_file = input_dir / "coordinates.json"
    with open(coords_file) as f:
        coords = json.load(f)
    lat = coords.get("latitude", coords.get("lat"))
    lon = coords.get("longitude", coords.get("lon"))

    # Step 1: Load original
    enhancer = ImageEnhancer()
    image_file = input_dir / "image_native.tif"
    original = enhancer.load_image(image_file)
    cv2.imwrite(str(output_dir / "01_original.png"), original)
    click.echo("  1. Saved original image")

    # Step 2: Enhanced
    enhanced = enhancer.enhance(original)
    cv2.imwrite(str(output_dir / "02_enhanced.png"), enhanced)
    click.echo("  2. Saved enhanced image")

    # Step 3: Road extraction
    extractor = RoadExtractor()
    road_result = extractor.extract(original)

    # Save edge map
    cv2.imwrite(str(output_dir / "03_edges.png"), road_result.edge_map)
    click.echo("  3. Saved edge detection result")

    # Save binary roads
    cv2.imwrite(str(output_dir / "04_binary_roads.png"), road_result.binary_roads)
    click.echo("  4. Saved binary road mask")

    # Save skeleton
    skeleton_vis = (road_result.skeleton * 255).astype(np.uint8)
    cv2.imwrite(str(output_dir / "05_skeleton.png"), skeleton_vis)
    click.echo("  5. Saved road skeleton")

    # Step 4b: Water extraction
    click.echo("  5b. Extracting water features...")
    water_extractor = WaterExtractor()
    water_result = water_extractor.extract(original)

    # Save water mask
    cv2.imwrite(str(output_dir / "05b_water_mask.png"), water_result.water_mask)
    click.echo(f"     Water coverage: {water_result.stats['water_coverage_ratio']:.1%}")
    click.echo(f"     Water bodies found: {water_result.stats['num_water_bodies']}")

    # Save shoreline contours overlaid on image
    shoreline_vis = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    for contour in water_result.shoreline_contours:
        # Draw contour as cyan line
        pts = contour.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(shoreline_vis, [pts], isClosed=True, color=(255, 255, 0), thickness=2)
    cv2.imwrite(str(output_dir / "05c_shoreline_contours.png"), shoreline_vis)
    click.echo(f"  5c. Saved shoreline contours ({len(water_result.shoreline_contours)} contours)")

    # Step 4: Graph overlay
    graph_builder = ImageGraphBuilder()
    image_graph = graph_builder.build_graph(road_result.skeleton)

    graph_vis = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    skeleton_coords = road_result.skeleton > 0
    graph_vis[skeleton_coords] = [0, 0, 255]  # Red skeleton

    for node in image_graph.nodes():
        x = int(image_graph.nodes[node].get("x", 0))
        y = int(image_graph.nodes[node].get("y", 0))
        is_junction = image_graph.nodes[node].get("is_junction", False)
        color = (0, 255, 0) if is_junction else (255, 255, 0)  # Green junctions, cyan endpoints
        radius = 4 if is_junction else 2
        cv2.circle(graph_vis, (x, y), radius, color, -1)

    cv2.imwrite(str(output_dir / "06_graph_overlay.png"), graph_vis)
    click.echo("  6. Saved graph overlay")

    # Step 5: OSM visualization
    click.echo("  7. Fetching OSM data...")
    osm_builder = OSMGraphBuilder()
    osm_graph = osm_builder.fetch_road_network(lat, lon, radius_miles=1.0)
    osm_graph, proj_info = osm_builder.project_to_local_crs(osm_graph, lat, lon)

    # Create OSM visualization
    osm_vis = np.zeros((800, 800, 3), dtype=np.uint8)
    osm_vis.fill(40)  # Dark gray background

    # Get bounds
    xs = [osm_graph.nodes[n].get('x_meters', 0) for n in osm_graph.nodes()]
    ys = [osm_graph.nodes[n].get('y_meters', 0) for n in osm_graph.nodes()]

    if xs and ys:
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        scale = 700 / max(max_x - min_x + 1, max_y - min_y + 1)

        def to_vis(x, y):
            vx = int((x - min_x) * scale + 50)
            vy = int((max_y - y) * scale + 50)  # Flip Y
            return vx, vy

        # Draw edges
        for u, v in osm_graph.edges():
            x1, y1 = osm_graph.nodes[u].get('x_meters', 0), osm_graph.nodes[u].get('y_meters', 0)
            x2, y2 = osm_graph.nodes[v].get('x_meters', 0), osm_graph.nodes[v].get('y_meters', 0)
            cv2.line(osm_vis, to_vis(x1, y1), to_vis(x2, y2), (100, 100, 100), 1)

        # Draw nodes
        for n in osm_graph.nodes():
            x, y = osm_graph.nodes[n].get('x_meters', 0), osm_graph.nodes[n].get('y_meters', 0)
            cv2.circle(osm_vis, to_vis(x, y), 2, (0, 200, 200), -1)

    cv2.imwrite(str(output_dir / "07_osm_network.png"), osm_vis)
    click.echo("  7. Saved OSM network visualization")

    # Step 7b: OSM water features
    click.echo("  7b. Fetching OSM water features...")
    osm_water_fetcher = OSMWaterFetcher()
    osm_water = osm_water_fetcher.fetch_water_features(lat, lon, radius_miles=1.0)

    if osm_water.stats["total_features"] > 0:
        osm_water, _ = osm_water_fetcher.project_to_local_crs(osm_water, lat, lon)

        # Create OSM water visualization
        water_osm_vis = np.zeros((800, 800, 3), dtype=np.uint8)
        water_osm_vis.fill(40)  # Dark gray background

        # Get bounds from all water features
        all_water_points = []
        for shoreline in osm_water.combined_shoreline_meters:
            all_water_points.extend(shoreline.tolist())

        if all_water_points:
            water_xs = [p[0] for p in all_water_points]
            water_ys = [p[1] for p in all_water_points]
            w_min_x, w_max_x = min(water_xs), max(water_xs)
            w_min_y, w_max_y = min(water_ys), max(water_ys)
            w_scale = 700 / max(w_max_x - w_min_x + 1, w_max_y - w_min_y + 1)

            def to_water_vis(x, y):
                vx = int((x - w_min_x) * w_scale + 50)
                vy = int((w_max_y - y) * w_scale + 50)
                return vx, vy

            # Draw coastlines in blue
            for coastline in osm_water.coastlines:
                pts = np.array([to_water_vis(p[0], p[1]) for p in coastline], dtype=np.int32)
                cv2.polylines(water_osm_vis, [pts], isClosed=False, color=(255, 100, 0), thickness=2)

            # Draw lakes in cyan
            for lake in osm_water.lakes:
                pts = np.array([to_water_vis(p[0], p[1]) for p in lake], dtype=np.int32)
                cv2.polylines(water_osm_vis, [pts], isClosed=True, color=(255, 255, 0), thickness=1)

            # Draw rivers in light blue
            for river in osm_water.rivers:
                pts = np.array([to_water_vis(p[0], p[1]) for p in river], dtype=np.int32)
                cv2.polylines(water_osm_vis, [pts], isClosed=False, color=(255, 150, 50), thickness=1)

        cv2.imwrite(str(output_dir / "07b_osm_water.png"), water_osm_vis)
        click.echo(f"  7b. Saved OSM water visualization (coastlines: {osm_water.stats['num_coastlines']}, lakes: {osm_water.stats['num_lakes']}, rivers: {osm_water.stats['num_rivers']})")
    else:
        click.echo("  7b. No OSM water features found in area")

    # Step 6: Matching candidates
    click.echo("  8. Computing matches...")
    feature_extractor = FeatureExtractor()

    candidates = feature_extractor.find_candidate_correspondences(
        image_graph, osm_graph
    )

    # Visualize candidates on image
    match_vis = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    for img_node, osm_node, score in candidates[:50]:  # Top 50
        x = int(image_graph.nodes[img_node].get("x", 0))
        y = int(image_graph.nodes[img_node].get("y", 0))
        cv2.circle(match_vis, (x, y), 5, (0, 255, 0), -1)

    cv2.imwrite(str(output_dir / "08_match_candidates.png"), match_vis)
    click.echo("  8. Saved matching candidates")

    # Step 8b: Shoreline matching (if water features available)
    shoreline_matches = []
    if water_result.shoreline_contours and osm_water.stats["total_features"] > 0:
        click.echo("  8b. Computing shoreline matches...")
        shoreline_matcher = ShorelineMatcher()
        shoreline_matches = shoreline_matcher.match(
            water_result.shoreline_contours,
            osm_water.combined_shoreline_meters,
        )

        # Visualize shoreline matches on image
        shoreline_match_vis = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        for corr in shoreline_matches:
            x, y = int(corr.image_point[0]), int(corr.image_point[1])
            cv2.circle(shoreline_match_vis, (x, y), 8, (0, 255, 255), 2)  # Yellow circles
            cv2.circle(shoreline_match_vis, (x, y), 3, (0, 255, 255), -1)

        cv2.imwrite(str(output_dir / "08b_shoreline_matches.png"), shoreline_match_vis)
        click.echo(f"  8b. Saved shoreline matches ({len(shoreline_matches)} correspondences)")
    else:
        click.echo("  8b. Skipped shoreline matching (no water features)")

    # Write summary
    summary = {
        "input_dir": str(input_dir),
        "location": {"lat": lat, "lon": lon},
        "image_shape": list(original.shape),
        "graph_stats": {
            "image_nodes": image_graph.number_of_nodes(),
            "image_edges": image_graph.number_of_edges(),
            "osm_nodes": osm_graph.number_of_nodes(),
            "osm_edges": osm_graph.number_of_edges(),
        },
        "water_stats": {
            "water_coverage_ratio": water_result.stats["water_coverage_ratio"],
            "num_water_bodies": water_result.stats["num_water_bodies"],
            "total_shoreline_length_px": water_result.stats["total_shoreline_length_px"],
            "osm_coastlines": osm_water.stats.get("num_coastlines", 0),
            "osm_lakes": osm_water.stats.get("num_lakes", 0),
            "osm_rivers": osm_water.stats.get("num_rivers", 0),
        },
        "matching": {
            "road_candidates": len(candidates),
            "unique_osm_nodes": len(set(c[1] for c in candidates)),
            "shoreline_correspondences": len(shoreline_matches),
            "total_correspondences": len(candidates) + len(shoreline_matches),
        },
        "projection": proj_info,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    click.echo(f"\n  Summary written to {output_dir / 'summary.json'}")
    click.echo(click.style(f"\nAll visualizations saved to {output_dir}", fg="green"))


def main():
    """Entry point for diagnostics CLI."""
    diagnostics()


if __name__ == "__main__":
    main()
