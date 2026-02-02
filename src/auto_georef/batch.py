"""Batch processing for multiple images."""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import logging
import numpy as np

from .config import GeoreferenceConfig, ImageMetadata
from .pipeline import GeoreferencePipeline, GeoreferenceResult

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


@dataclass
class BatchResult:
    """Result of batch processing."""

    total_images: int
    successful: int
    failed: int
    low_confidence: int
    results: List[Dict[str, Any]]
    start_time: str
    end_time: str
    total_seconds: float


class BatchProcessor:
    """Process multiple images in batch."""

    def __init__(self, config: Optional[GeoreferenceConfig] = None):
        """Initialize the batch processor.

        Args:
            config: Pipeline configuration.
        """
        self.config = config or GeoreferenceConfig.default()

    def discover_images(self, input_dir: Path) -> List[Path]:
        """Discover all processable image directories.

        Looks for directories containing both image files and coordinates.json.

        Args:
            input_dir: Root directory to search.

        Returns:
            List of directory paths that can be processed.
        """
        input_dir = Path(input_dir)
        processable = []

        for coords_file in input_dir.rglob("coordinates.json"):
            item_dir = coords_file.parent

            # Check for image file
            has_image = (
                (item_dir / "image_native.tif").exists()
                or (item_dir / "image_medium.jpg").exists()
            )

            if has_image:
                processable.append(item_dir)

        return sorted(processable)

    def process_single(
        self, item_dir: Path, output_dir: Path
    ) -> Dict[str, Any]:
        """Process a single image directory.

        Args:
            item_dir: Directory containing image and metadata.
            output_dir: Output directory.

        Returns:
            Dictionary with processing result.
        """
        item_id = item_dir.name
        item_output_dir = output_dir / item_id

        try:
            pipeline = GeoreferencePipeline(self.config)
            result = pipeline.process_directory(item_dir, item_output_dir)

            return {
                "item_id": item_id,
                "source_dir": str(item_dir),
                "status": "success" if result.success else "low_confidence",
                "confidence_score": result.confidence_score,
                "error_message": result.error_message,
                "output_paths": {k: str(v) for k, v in result.output_paths.items()},
                "debug_info": result.debug_info,
            }

        except Exception as e:
            logger.exception(f"Error processing {item_dir}: {e}")
            return {
                "item_id": item_id,
                "source_dir": str(item_dir),
                "status": "failed",
                "confidence_score": 0.0,
                "error_message": str(e),
                "output_paths": {},
                "debug_info": {},
            }

    def process_batch(
        self,
        input_dir: Path,
        output_dir: Path,
        num_workers: int = 1,
    ) -> BatchResult:
        """Process all images in a directory.

        Args:
            input_dir: Root input directory.
            output_dir: Root output directory.
            num_workers: Number of parallel workers (1 for sequential).

        Returns:
            BatchResult with all processing results.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Discover images
        image_dirs = self.discover_images(input_dir)
        logger.info(f"Found {len(image_dirs)} images to process")

        if not image_dirs:
            return BatchResult(
                total_images=0,
                successful=0,
                failed=0,
                low_confidence=0,
                results=[],
                start_time=datetime.now().isoformat(),
                end_time=datetime.now().isoformat(),
                total_seconds=0.0,
            )

        start_time = datetime.now()
        results = []

        if num_workers <= 1:
            # Sequential processing
            for i, item_dir in enumerate(image_dirs):
                logger.info(f"Processing {i+1}/{len(image_dirs)}: {item_dir.name}")
                result = self.process_single(item_dir, output_dir)
                results.append(result)
        else:
            # Parallel processing
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(self.process_single, item_dir, output_dir): item_dir
                    for item_dir in image_dirs
                }

                for i, future in enumerate(as_completed(futures)):
                    item_dir = futures[future]
                    logger.info(
                        f"Completed {i+1}/{len(image_dirs)}: {item_dir.name}"
                    )
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.exception(f"Error in worker for {item_dir}: {e}")
                        results.append({
                            "item_id": item_dir.name,
                            "source_dir": str(item_dir),
                            "status": "failed",
                            "confidence_score": 0.0,
                            "error_message": str(e),
                            "output_paths": {},
                            "debug_info": {},
                        })

        end_time = datetime.now()

        # Compute summary statistics
        successful = sum(1 for r in results if r["status"] == "success")
        failed = sum(1 for r in results if r["status"] == "failed")
        low_confidence = sum(1 for r in results if r["status"] == "low_confidence")

        batch_result = BatchResult(
            total_images=len(image_dirs),
            successful=successful,
            failed=failed,
            low_confidence=low_confidence,
            results=results,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            total_seconds=(end_time - start_time).total_seconds(),
        )

        # Write summary report
        self._write_report(batch_result, output_dir)

        return batch_result

    def _write_report(self, result: BatchResult, output_dir: Path) -> None:
        """Write batch processing report.

        Args:
            result: Batch processing result.
            output_dir: Output directory.
        """
        report_path = output_dir / "batch_report.json"

        report = {
            "summary": {
                "total_images": result.total_images,
                "successful": result.successful,
                "failed": result.failed,
                "low_confidence": result.low_confidence,
                "success_rate": (
                    result.successful / result.total_images
                    if result.total_images > 0
                    else 0
                ),
                "start_time": result.start_time,
                "end_time": result.end_time,
                "total_seconds": result.total_seconds,
            },
            "results": result.results,
        }

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)

        logger.info(f"Batch report written to {report_path}")

        # Also write a simple CSV for easy review
        csv_path = output_dir / "batch_results.csv"
        with open(csv_path, "w") as f:
            f.write("item_id,status,confidence_score,error_message\n")
            for r in result.results:
                error = (r.get("error_message") or "").replace(",", ";")
                f.write(
                    f"{r['item_id']},{r['status']},{r['confidence_score']:.3f},\"{error}\"\n"
                )

        logger.info(f"CSV results written to {csv_path}")
