# Auto-Georeference

Automatically georeference historical aerial images using road network matching against OpenStreetMap data.

## Overview

This tool processes historical aerial photographs (1930s-1980s) and attempts to automatically determine their geographic location and orientation by:

1. Detecting road networks in the image using computer vision
2. Building a graph representation of the detected roads
3. Fetching reference road data from OpenStreetMap
4. Matching the image graph against the OSM graph using spectral and feature-based methods
5. Estimating a geographic transformation using RANSAC
6. Generating a georeferenced GeoTIFF output

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
    ├── image_native.tif    # The aerial image (required)
    ├── coordinates.json    # Approximate location (required)
    └── metadata.json       # Additional metadata (optional)
```

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

### Single Image Processing

```bash
python -m auto_georef single <input_dir> <output_dir>
```

Example:
```bash
python -m auto_georef single data/000001 output/
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
python -m auto_georef visualize output/image_native_georef.tif --open-browser
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
python -m auto_georef diagnose geotiff output/image_native_georef.tif
```

#### Check Transform
Analyze transformation parameters from a result JSON:
```bash
python -m auto_georef diagnose transform output/image_native_result.json
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

After processing, the output directory contains:

| File | Description |
|------|-------------|
| `*_georef.tif` | Georeferenced GeoTIFF image |
| `*_georef.tfw` | World file with transformation parameters |
| `*_result.json` | Processing results and quality metrics |
| `*_map.html` | Interactive map visualization (if generated) |

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
python -m auto_georef visualize output/image_georef.tif --open-browser
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
# Check if GeoTIFF has valid bounds
python -m auto_georef diagnose geotiff output/image_georef.tif

# Analyze transformation parameters
python -m auto_georef diagnose transform output/result.json

# Debug the matching process
python -m auto_georef diagnose matching data/000001

# Generate step-by-step visualizations
python -m auto_georef diagnose steps data/000001 -o diagnostic_output/
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
| `06_graph_overlay.png` | Detected junctions (green) and endpoints (cyan) |
| `07_osm_network.png` | Reference OSM road network |
| `08_match_candidates.png` | Matched points (green circles) |

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
                                               │
                                               ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   GeoTIFF   │◀────│  Transform  │◀────│   Graph     │
│   Output    │     │ Estimation  │     │  Matching   │
└─────────────┘     └─────────────┘     └─────────────┘
                                               ▲
                                               │
                          ┌─────────────┐      │
                          │    OSM      │──────┘
                          │    Data     │
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

Default configuration can be modified in `src/auto_georef/config.py`:

```python
@dataclass
class OSMConfig:
    search_radius_miles: float = 1.0
    expanded_radius_miles: float = 2.0
    network_type: str = "drive"

@dataclass
class MatchingConfig:
    min_correspondences: int = 5
    ransac_threshold_meters: float = 15.0
    ransac_max_iterations: int = 1000

@dataclass
class QualityConfig:
    min_confidence_score: float = 0.5
    max_rmse_meters: float = 15.0
    min_matched_roads: int = 3
```

## Limitations

- **Road-dependent**: Works best on images with visible road networks
- **Water/vegetation**: Areas with mostly water or dense vegetation may fail
- **Historical changes**: Roads that no longer exist won't match
- **Scale estimation**: Requires some road intersections for accurate scale
- **Rotation**: Large rotations (> 45°) may reduce matching accuracy

## License

[Add your license here]
