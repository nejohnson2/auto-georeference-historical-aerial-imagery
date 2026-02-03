# Auto-Georeference Historical Aerial Imagery

Automatically georeference historical aerial images using road network and shoreline matching against OpenStreetMap data.

## Quick Start

```bash
# 1. Install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -e .

# 2. Process a single image
make single ID=000001

# 3. View the result
make visualize ID=000001

# 4. Debug if needed
make diagnose ID=000001
```

Or without Make:
```bash
python -m auto_georef single data/000001 output/000001
python -m auto_georef visualize output/000001/000001_georef.tif
python -m auto_georef diagnose steps data/000001 -o diagnostic_output/000001
```

## Overview

This tool processes historical aerial photographs (1930s-1980s) and attempts to automatically determine their geographic location and orientation by:

1. Detecting road networks in the image using computer vision
2. Detecting water bodies and extracting shoreline contours (coastlines, lakes, rivers)
3. Building graph representations of detected features
4. Fetching reference road and water data from OpenStreetMap
5. Matching image features against OSM using spectral and curvature-based methods
6. Estimating a geographic transformation (with rotation) using RANSAC
7. Generating a georeferenced GeoTIFF output

**Key features:**
- Supports arbitrary image rotations (north can be any direction)
- Water/shoreline matching for coastal areas (shorelines are more stable over time than roads)
- Automatically selects highest-resolution ("native") images

## Installation

### Prerequisites

- Python 3.10+
- pip

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd auto-georeference

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Input Data Format

Each image should be in its own directory with the following structure:

```
data/
└── 000001/
    ├── image_native.tif    # High-res aerial image (required, any extension)
    ├── image_medium.jpg    # Medium-res fallback (optional)
    ├── image_thumbnail.jpg # Thumbnail (ignored)
    ├── coordinates.json    # Approximate location (required)
    └── metadata.json       # Additional metadata (optional)
```

**Image file selection:**
- The tool automatically finds and uses files with "native" in the name
- Supported formats: `.tif`, `.tiff`, `.jpg`, `.jpeg`, `.png`
- Falls back to other images if no "native" file exists (excludes thumbnails)

### coordinates.json

```json
{
  "latitude": 40.8475,
  "longitude": -72.5208
}
```

### metadata.json (optional)

```json
{
  "date": "1938",
  "description": "Aerial photo of coastal area"
}
```

## Usage

### Using the Makefile (Recommended)

The Makefile provides convenient shortcuts for common operations:

```bash
# List available images
make list

# Process a single image
make single ID=000001

# Process all images
make batch

# Generate diagnostic visualizations
make diagnose ID=000001

# Create map visualization (after processing)
make visualize ID=000001

# Analyze image and suggest parameter tuning
make tune ID=000001

# Full pipeline: process + visualize
make full ID=000001

# Full pipeline with diagnostics
make full-debug ID=000001

# Clean all outputs
make clean

# Clean outputs for specific image
make clean-id ID=000001

# Show all available commands
make help
```

### Single Image Processing

```bash
python -m auto_georef single <input_dir> <output_dir>
```

Example:
```bash
python -m auto_georef single data/000001 output/000001
```

### Batch Processing

Process multiple images in a directory:

```bash
python -m auto_georef batch <data_dir> <output_dir>
```

Example:
```bash
python -m auto_georef batch data/ output/
```

### Visualization

View a georeferenced image overlaid on a map:

```bash
python -m auto_georef visualize <geotiff_path> [options]
```

Options:
- `--output, -o`: Output HTML file path
- `--opacity`: Initial overlay opacity (0-1, default: 0.7)
- `--result-json`: Path to result JSON for quality metadata
- `--open-browser`: Open visualization in default browser

Example:
```bash
python -m auto_georef visualize output/000001/000001_georef.tif --open-browser
```

### Diagnostic Commands

Debug and analyze the georeferencing process:

```bash
python -m auto_georef diagnose <subcommand> <path> [options]
```

Subcommands:

#### Check GeoTIFF
Inspect georeferencing parameters of a GeoTIFF:
```bash
python -m auto_georef diagnose geotiff output/000001/000001_georef.tif
```

#### Check Transform
Analyze transformation parameters from a result JSON:
```bash
python -m auto_georef diagnose transform output/000001/000001_result.json
```

#### Check Matching
Debug the matching process for an image:
```bash
python -m auto_georef diagnose matching data/000001
```

#### Visualize Steps
Generate images of all intermediate processing steps:
```bash
python -m auto_georef diagnose steps data/000001 -o diagnostic_output/
```

### Debug Commands

#### Debug Road Extraction
Visualize road detection on an image:
```bash
python -m auto_georef debug-roads <image_path> [--output debug_roads.png]
```

#### Debug OSM Data
Fetch and display OSM road data for a location:
```bash
python -m auto_georef debug-osm <lat> <lon> [--radius 1.0]
```

## Output Files

After processing, outputs are organized by image ID in subdirectories:

```
output/
├── 000001/
│   ├── 000001_georef.tif      # Georeferenced GeoTIFF image
│   ├── 000001_georef.tfw      # World file with transformation parameters
│   ├── 000001_result.json     # Processing results and quality metrics
│   └── 000001_map.html        # Interactive map visualization
├── 000242/
│   └── ...
└── batch_report.json          # Summary report (batch processing only)

diagnostic_output/
├── 000001/
│   ├── 01_original.png
│   ├── 02_enhanced.png
│   └── ...
└── 000242/
    └── ...
```

**Note:** Output files use the input directory name (e.g., `000001`) as the prefix, not the image filename. Each image gets its own subdirectory to keep outputs organized.

### Result JSON Structure

```json
{
  "georeferencing": {
    "status": "success",
    "confidence_score": 0.72,
    "rmse_meters": 8.5,
    "num_correspondences": 15,
    "coverage_ratio": 0.45,
    "warnings": []
  },
  "transformation": {
    "scale_x": 0.85,
    "scale_y": -0.85,
    "rotation_deg": 12.5,
    "origin_lon": -72.5208,
    "origin_lat": 40.8475
  },
  "source": {
    "original_lat": 40.8475,
    "original_lon": -72.5208,
    "date": "1938"
  }
}
```

## Evaluating Outputs

### Step 1: Check Quality Metrics

Review the result JSON for key indicators:

| Metric | Good | Warning | Poor |
|--------|------|---------|------|
| Confidence Score | > 0.7 | 0.5-0.7 | < 0.5 |
| RMSE (meters) | < 10 | 10-20 | > 20 |
| Correspondences | > 10 | 5-10 | < 5 |
| Coverage Ratio | > 0.3 | 0.1-0.3 | < 0.1 |

### Step 2: Visual Inspection

Open the HTML visualization in a browser:

```bash
make visualize ID=000001
# Or: python -m auto_georef visualize output/000001/000001_georef.tif --open-browser
```

**Evaluation checklist:**

1. **Toggle base maps** - Use the layer control to switch between:
   - Google Satellite (best for comparing landscape features)
   - Google Maps (best for road alignment)
   - OpenStreetMap (best for road names/labels)

2. **Adjust opacity** - Use the slider to fade between historical and modern:
   - Check road alignment at intersections
   - Verify coastlines and water bodies
   - Compare building footprints (if visible)

3. **Check scale** - The red boundary box should:
   - Cover a reasonable area (100m - 2km depending on image)
   - Not be extremely small (< 50m) or large (> 10km)

4. **Verify rotation** - Roads in the image should:
   - Run parallel to modern roads
   - Intersections should align

### Step 3: Run Diagnostics

If results look incorrect, run the diagnostic tools:

```bash
# Generate step-by-step visualizations
make diagnose ID=000001

# Check if GeoTIFF has valid bounds
make check-geotiff ID=000001

# Analyze transformation parameters
make check-transform ID=000001

# Debug the matching process
make check-matching ID=000001
```

### Step 4: Review Diagnostic Images

The `diagnose steps` command generates:

| Image | What to Check |
|-------|--------------|
| `01_original.png` | Input image quality |
| `02_enhanced.png` | Contrast enhancement |
| `03_edges.png` | Edge detection - roads should be visible |
| `04_binary_roads.png` | Road mask - should show thin lines, not blobs |
| `05_skeleton.png` | Skeletonized roads - single-pixel width |
| `05b_water_mask.png` | Detected water regions (white areas) |
| `05c_shoreline_contours.png` | Extracted shorelines (cyan lines) |
| `06_graph_overlay.png` | Detected junctions (green) and endpoints (cyan) |
| `07_osm_network.png` | Reference OSM road network |
| `07b_osm_water.png` | OSM water features (coastlines, lakes, rivers) |
| `08_match_candidates.png` | Road match candidates (green circles) |
| `08b_shoreline_matches.png` | Shoreline match correspondences (yellow circles) |

### Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| Image too small on map | Bounds span < 50m | Matching failed - check `diagnose matching` |
| Wrong location | Image appears in wrong place | Verify coordinates.json is correct |
| Wrong rotation | Roads don't align | Insufficient road intersections detected |
| Low confidence | Score < 0.5 | Image may have few visible roads |
| No roads detected | Empty skeleton | Image may be mostly water/vegetation |

## How It Works

### Pipeline Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Image     │────▶│   Road      │────▶│   Graph     │
│   Input     │     │ Extraction  │     │  Building   │
└─────────────┘     └─────────────┘     └─────────────┘
      │                                        │
      │             ┌─────────────┐            │
      └────────────▶│   Water     │────────────┤
                    │ Extraction  │            │
                    └─────────────┘            │
                                               ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   GeoTIFF   │◀────│  Transform  │◀────│  Combined   │
│   Output    │     │ Estimation  │     │  Matching   │
└─────────────┘     └─────────────┘     └─────────────┘
                                               ▲
                    ┌─────────────┐            │
                    │  OSM Roads  │────────────┤
                    └─────────────┘            │
                    ┌─────────────┐            │
                    │  OSM Water  │────────────┘
                    └─────────────┘
```

### Processing Steps

1. **Image Enhancement**
   - CLAHE (Contrast Limited Adaptive Histogram Equalization)
   - Bilateral filtering for edge-preserving smoothing

2. **Road Extraction**
   - Canny edge detection
   - Morphological closing to connect segments
   - Skeletonization to single-pixel width
   - Small component filtering

3. **Graph Construction**
   - Junction detection (pixels with 3+ neighbors)
   - Road segment tracing between junctions
   - NetworkX graph with node positions and edge properties

4. **OSM Reference Data**
   - Fetch road network via OSMnx within search radius
   - Project to local meter-based coordinates
   - Apply historical relevance weighting (older roads weighted higher)

5. **Graph Matching**
   - Compute node signatures (degree, angles, local density)
   - Find candidate correspondences based on signature similarity
   - RANSAC to find geometrically consistent matches

6. **Transformation Estimation**
   - Estimate similarity transform (rotation + scale + translation)
   - Compute quality metrics (RMSE, coverage, distribution)

7. **Output Generation**
   - Write georeferenced GeoTIFF with proper CRS
   - Generate world file (.tfw)
   - Save result JSON with metrics

## Configuration

Configuration can be modified in `config.yaml` or `src/auto_georef/config.py`:

### Key Configuration Options

**OSM Settings:**
```yaml
osm:
  search_radius_miles: 1.0      # Initial search radius
  expanded_radius_miles: 2.0    # Expanded radius for sparse images
  network_type: "drive"         # Road types to fetch
```

**Matching Settings:**
```yaml
matching:
  min_correspondences: 3        # Minimum matches required
  ransac_threshold_meters: 50.0 # Historical images need larger threshold
  rotation_range_deg: [-90, 90] # Full rotation search range
```

**Water/Shoreline Settings:**
```yaml
water:
  enabled: true                 # Enable water feature detection
  shoreline_weight: 1.5         # Shorelines weighted higher than roads
  min_water_area_px: 500        # Minimum water body size
  fetch_coastlines: true        # Fetch OSM coastlines
  fetch_lakes: true             # Fetch OSM lakes
  fetch_rivers: true            # Fetch OSM rivers
```

**Quality Thresholds:**
```yaml
quality:
  min_confidence_score: 0.3     # Relaxed for historical images
  max_rmse_meters: 50.0         # Historical images typically 20-50m accuracy
```

## Limitations

- **Feature-dependent**: Works best on images with visible roads OR shorelines
- **Dense vegetation**: Areas with dense vegetation and no roads/water may fail
- **Historical road changes**: Roads that no longer exist won't match (shorelines are more stable)
- **Scale estimation**: Requires some feature intersections for accurate scale
- **Featureless areas**: Open farmland or desert with few distinguishing features may fail

## License

[Add your license here]
