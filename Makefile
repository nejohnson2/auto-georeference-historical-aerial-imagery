# Auto-Georeference Makefile
# ===========================
# Simplifies running georeferencing pipeline stages

# Directories
DATA_DIR := data
OUTPUT_DIR := output
DIAG_DIR := diagnostic_output

# Python command (uses virtual environment if activated)
PYTHON := python

# Default target
.PHONY: help
help:
	@echo "Auto-Georeference Pipeline"
	@echo "=========================="
	@echo ""
	@echo "Usage:"
	@echo "  make single ID=000001    Process a single image"
	@echo "  make batch               Process all images in data/"
	@echo "  make diagnose ID=000001  Generate diagnostic visualizations"
	@echo "  make visualize ID=000001 Create map visualization"
	@echo "  make tune ID=000001      Analyze image and suggest parameters"
	@echo "  make list                List available image directories"
	@echo "  make clean               Remove all outputs"
	@echo ""
	@echo "Examples:"
	@echo "  make single ID=000001"
	@echo "  make diagnose ID=000001"
	@echo "  make batch"
	@echo ""

# Process a single image
# Usage: make single ID=000001
.PHONY: single
single:
ifndef ID
	$(error ID is required. Usage: make single ID=000001)
endif
	@echo "Processing $(DATA_DIR)/$(ID)..."
	@mkdir -p $(OUTPUT_DIR)/$(ID)
	$(PYTHON) -m auto_georef single $(DATA_DIR)/$(ID) $(OUTPUT_DIR)/$(ID)
	@echo ""
	@echo "Output files in $(OUTPUT_DIR)/$(ID)/"

# Process all images in data/
.PHONY: batch
batch:
	@echo "Batch processing all images in $(DATA_DIR)/..."
	$(PYTHON) -m auto_georef batch $(DATA_DIR) $(OUTPUT_DIR)
	@echo ""
	@echo "Results in $(OUTPUT_DIR)/"
	@echo "Report: $(OUTPUT_DIR)/batch_report.json"

# Generate diagnostic step visualizations
# Usage: make diagnose ID=000001
.PHONY: diagnose
diagnose:
ifndef ID
	$(error ID is required. Usage: make diagnose ID=000001)
endif
	@echo "Generating diagnostics for $(DATA_DIR)/$(ID)..."
	@mkdir -p $(DIAG_DIR)/$(ID)
	$(PYTHON) -m auto_georef diagnose steps $(DATA_DIR)/$(ID) -o $(DIAG_DIR)/$(ID)
	@echo ""
	@echo "Diagnostic images in $(DIAG_DIR)/$(ID)/"

# Create map visualization for a processed image
# Usage: make visualize ID=000001
.PHONY: visualize
visualize:
ifndef ID
	$(error ID is required. Usage: make visualize ID=000001)
endif
	@if [ ! -f "$(OUTPUT_DIR)/$(ID)/$(ID)_georef.tif" ]; then \
		echo "Error: $(OUTPUT_DIR)/$(ID)/$(ID)_georef.tif not found"; \
		echo "Run 'make single ID=$(ID)' first"; \
		exit 1; \
	fi
	@echo "Creating visualization for $(ID)..."
	$(PYTHON) -m auto_georef visualize $(OUTPUT_DIR)/$(ID)/$(ID)_georef.tif \
		-o $(OUTPUT_DIR)/$(ID)/$(ID)_map.html
	@echo ""
	@echo "Map: $(OUTPUT_DIR)/$(ID)/$(ID)_map.html"

# Analyze image and suggest parameter tuning
# Usage: make tune ID=000001
.PHONY: tune
tune:
ifndef ID
	$(error ID is required. Usage: make tune ID=000001)
endif
	@echo "Analyzing $(DATA_DIR)/$(ID)..."
	$(PYTHON) -m auto_georef tune $(DATA_DIR)/$(ID)

# Debug road extraction
# Usage: make debug-roads ID=000001
.PHONY: debug-roads
debug-roads:
ifndef ID
	$(error ID is required. Usage: make debug-roads ID=000001)
endif
	@mkdir -p $(DIAG_DIR)/$(ID)
	$(PYTHON) -m auto_georef debug-roads $(DATA_DIR)/$(ID) -o $(DIAG_DIR)/$(ID)/roads_debug.png
	@echo "Debug image: $(DIAG_DIR)/$(ID)/roads_debug.png"

# Check GeoTIFF parameters
# Usage: make check-geotiff ID=000001
.PHONY: check-geotiff
check-geotiff:
ifndef ID
	$(error ID is required. Usage: make check-geotiff ID=000001)
endif
	@if [ ! -f "$(OUTPUT_DIR)/$(ID)/$(ID)_georef.tif" ]; then \
		echo "Error: $(OUTPUT_DIR)/$(ID)/$(ID)_georef.tif not found"; \
		exit 1; \
	fi
	$(PYTHON) -m auto_georef diagnose geotiff $(OUTPUT_DIR)/$(ID)/$(ID)_georef.tif

# Check transform from result JSON
# Usage: make check-transform ID=000001
.PHONY: check-transform
check-transform:
ifndef ID
	$(error ID is required. Usage: make check-transform ID=000001)
endif
	@if [ ! -f "$(OUTPUT_DIR)/$(ID)/$(ID)_result.json" ]; then \
		echo "Error: $(OUTPUT_DIR)/$(ID)/$(ID)_result.json not found"; \
		exit 1; \
	fi
	$(PYTHON) -m auto_georef diagnose transform $(OUTPUT_DIR)/$(ID)/$(ID)_result.json

# Debug matching process
# Usage: make check-matching ID=000001
.PHONY: check-matching
check-matching:
ifndef ID
	$(error ID is required. Usage: make check-matching ID=000001)
endif
	$(PYTHON) -m auto_georef diagnose matching $(DATA_DIR)/$(ID)

# List available image directories
.PHONY: list
list:
	@echo "Available images in $(DATA_DIR)/:"
	@echo ""
	@if [ -d "$(DATA_DIR)" ]; then \
		for dir in $(DATA_DIR)/*/; do \
			if [ -f "$$dir/coordinates.json" ]; then \
				name=$$(basename $$dir); \
				echo "  $$name"; \
			fi; \
		done; \
	else \
		echo "  (no data directory found)"; \
	fi
	@echo ""

# Clean all output files
.PHONY: clean
clean:
	@echo "Removing output files..."
	rm -rf $(OUTPUT_DIR)
	rm -rf $(DIAG_DIR)
	@echo "Done."

# Clean output for a specific image
# Usage: make clean-id ID=000001
.PHONY: clean-id
clean-id:
ifndef ID
	$(error ID is required. Usage: make clean-id ID=000001)
endif
	@echo "Removing outputs for $(ID)..."
	rm -rf $(OUTPUT_DIR)/$(ID)
	rm -rf $(DIAG_DIR)/$(ID)
	@echo "Done."

# Full pipeline: process + visualize
# Usage: make full ID=000001
.PHONY: full
full:
ifndef ID
	$(error ID is required. Usage: make full ID=000001)
endif
	@$(MAKE) single ID=$(ID)
	@$(MAKE) visualize ID=$(ID)

# Full pipeline with diagnostics
# Usage: make full-debug ID=000001
.PHONY: full-debug
full-debug:
ifndef ID
	$(error ID is required. Usage: make full-debug ID=000001)
endif
	@$(MAKE) diagnose ID=$(ID)
	@$(MAKE) single ID=$(ID)
	@$(MAKE) visualize ID=$(ID)
