"""RANSAC-based robust matching and transformation estimation."""

import numpy as np
import networkx as nx
from dataclasses import dataclass
from typing import List, Tuple, Optional
import random


@dataclass
class RANSACResult:
    """Result of RANSAC matching."""

    inlier_correspondences: List[Tuple[int, int]]
    outlier_correspondences: List[Tuple[int, int]]
    transformation_matrix: np.ndarray  # 3x3 affine matrix
    scale: float
    rotation_deg: float
    translation: Tuple[float, float]
    inlier_ratio: float
    rmse: float
    num_iterations: int


class RANSACMatcher:
    """RANSAC-based robust correspondence estimation."""

    def __init__(
        self,
        threshold_meters: float = 10.0,
        max_iterations: int = 1000,
        confidence: float = 0.99,
        min_inliers: int = 3,
    ):
        """Initialize RANSAC matcher.

        Args:
            threshold_meters: Maximum reprojection error for inliers.
            max_iterations: Maximum RANSAC iterations.
            confidence: Desired confidence level.
            min_inliers: Minimum inliers to accept a model.
        """
        self.threshold = threshold_meters
        self.max_iterations = max_iterations
        self.confidence = confidence
        self.min_inliers = min_inliers

    def estimate_affine_transform(
        self, src_points: np.ndarray, dst_points: np.ndarray
    ) -> np.ndarray:
        """Estimate affine transformation from point correspondences.

        Args:
            src_points: Source points (N x 2).
            dst_points: Destination points (N x 2).

        Returns:
            3x3 affine transformation matrix.
        """
        n = src_points.shape[0]
        if n < 3:
            return np.eye(3)

        # Build system: [x y 1 0 0 0; 0 0 0 x y 1] @ [a b c d e f]' = [x' y']
        A = np.zeros((2 * n, 6))
        b = np.zeros(2 * n)

        for i in range(n):
            x, y = src_points[i]
            x_prime, y_prime = dst_points[i]

            A[2 * i] = [x, y, 1, 0, 0, 0]
            A[2 * i + 1] = [0, 0, 0, x, y, 1]
            b[2 * i] = x_prime
            b[2 * i + 1] = y_prime

        # Solve using least squares
        params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        # Build transformation matrix
        T = np.array(
            [[params[0], params[1], params[2]], [params[3], params[4], params[5]], [0, 0, 1]]
        )

        return T

    def estimate_similarity_transform(
        self, src_points: np.ndarray, dst_points: np.ndarray
    ) -> Tuple[np.ndarray, float, float, Tuple[float, float]]:
        """Estimate similarity transformation (rotation + scale + translation).

        Args:
            src_points: Source points (N x 2).
            dst_points: Destination points (N x 2).

        Returns:
            Tuple of (3x3 matrix, scale, rotation_deg, translation).
        """
        n = src_points.shape[0]
        if n < 2:
            return np.eye(3), 1.0, 0.0, (0.0, 0.0)

        # Check for NaN or Inf values
        if not np.all(np.isfinite(src_points)) or not np.all(np.isfinite(dst_points)):
            return np.eye(3), 1.0, 0.0, (0.0, 0.0)

        # Compute centroids
        src_centroid = np.mean(src_points, axis=0)
        dst_centroid = np.mean(dst_points, axis=0)

        # Center points
        src_centered = src_points - src_centroid
        dst_centered = dst_points - dst_centroid

        # Compute scale
        src_scale = np.sqrt(np.sum(src_centered**2) / n)
        dst_scale = np.sqrt(np.sum(dst_centered**2) / n)

        if src_scale < 1e-10 or dst_scale < 1e-10:
            return np.eye(3), 1.0, 0.0, (0.0, 0.0)

        scale = dst_scale / src_scale

        # Normalize
        src_normalized = src_centered / src_scale
        dst_normalized = dst_centered / dst_scale

        # Compute rotation using SVD
        H = src_normalized.T @ dst_normalized

        # Check for valid matrix before SVD
        if not np.all(np.isfinite(H)):
            return np.eye(3), 1.0, 0.0, (0.0, 0.0)

        try:
            U, _, Vt = np.linalg.svd(H)
        except np.linalg.LinAlgError:
            return np.eye(3), 1.0, 0.0, (0.0, 0.0)

        R = Vt.T @ U.T

        # Ensure proper rotation (det = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Compute rotation angle
        rotation_rad = np.arctan2(R[1, 0], R[0, 0])
        rotation_deg = np.degrees(rotation_rad)

        # Compute translation
        translation = dst_centroid - scale * (R @ src_centroid)

        # Build transformation matrix
        T = np.eye(3)
        T[:2, :2] = scale * R
        T[:2, 2] = translation

        # Final sanity check
        if not np.all(np.isfinite(T)):
            return np.eye(3), 1.0, 0.0, (0.0, 0.0)

        return T, scale, rotation_deg, tuple(translation)

    def apply_transform(self, points: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Apply transformation matrix to points.

        Args:
            points: Points to transform (N x 2).
            T: 3x3 transformation matrix.

        Returns:
            Transformed points (N x 2).
        """
        n = points.shape[0]
        homogeneous = np.column_stack([points, np.ones(n)])
        transformed = (T @ homogeneous.T).T
        return transformed[:, :2]

    def compute_residuals(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray,
        T: np.ndarray,
    ) -> np.ndarray:
        """Compute reprojection residuals.

        Args:
            src_points: Source points (N x 2).
            dst_points: Destination points (N x 2).
            T: Transformation matrix.

        Returns:
            Array of residual distances.
        """
        projected = self.apply_transform(src_points, T)
        residuals = np.linalg.norm(projected - dst_points, axis=1)
        return residuals

    def find_inliers(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray,
        T: np.ndarray,
    ) -> np.ndarray:
        """Find inlier indices based on reprojection error.

        Args:
            src_points: Source points (N x 2).
            dst_points: Destination points (N x 2).
            T: Transformation matrix.

        Returns:
            Boolean array indicating inliers.
        """
        residuals = self.compute_residuals(src_points, dst_points, T)
        # Handle NaN/Inf residuals
        valid = np.isfinite(residuals)
        inliers = valid & (residuals < self.threshold)
        return inliers

    def ransac(
        self,
        image_graph: nx.Graph,
        osm_graph: nx.Graph,
        candidates: List[Tuple[int, int, float]],
    ) -> Optional[RANSACResult]:
        """Run RANSAC to find robust correspondences and transformation.

        Args:
            image_graph: Graph from detected roads (pixel coords).
            osm_graph: Reference OSM graph (meter coords).
            candidates: Candidate correspondences (img_node, osm_node, distance).

        Returns:
            RANSACResult if successful, None otherwise.
        """
        if len(candidates) < self.min_inliers:
            return None

        # Check for sufficient unique OSM nodes (need at least 3 for proper transform)
        unique_osm_nodes = set(c[1] for c in candidates)
        if len(unique_osm_nodes) < 3:
            return None

        # Filter to keep only candidates with unique OSM node mappings
        # (one image node per OSM node, keeping best match)
        osm_to_best_img = {}
        for img_node, osm_node, dist in candidates:
            if osm_node not in osm_to_best_img or dist < osm_to_best_img[osm_node][2]:
                osm_to_best_img[osm_node] = (img_node, osm_node, dist)

        # Also filter by unique image nodes
        img_to_best_osm = {}
        for img_node, osm_node, dist in osm_to_best_img.values():
            if img_node not in img_to_best_osm or dist < img_to_best_osm[img_node][2]:
                img_to_best_osm[img_node] = (img_node, osm_node, dist)

        filtered_candidates = list(img_to_best_osm.values())

        # Re-check minimum requirements after filtering
        if len(filtered_candidates) < self.min_inliers:
            return None

        unique_osm_after_filter = set(c[1] for c in filtered_candidates)
        if len(unique_osm_after_filter) < 3:
            return None

        # Use filtered candidates for RANSAC
        candidates = filtered_candidates

        # Extract point coordinates
        img_points = []
        osm_points = []
        for img_node, osm_node, _ in candidates:
            img_data = image_graph.nodes[img_node]
            osm_data = osm_graph.nodes[osm_node]

            img_points.append([img_data.get("x", 0), img_data.get("y", 0)])
            osm_points.append(
                [
                    osm_data.get("x_meters", osm_data.get("x", 0)),
                    osm_data.get("y_meters", osm_data.get("y", 0)),
                ]
            )

        img_points = np.array(img_points)
        osm_points = np.array(osm_points)

        best_inliers = None
        best_num_inliers = 0
        best_T = None
        best_scale = 1.0
        best_rotation = 0.0
        best_translation = (0.0, 0.0)

        # Adaptive iteration count
        n = len(candidates)
        iterations = 0
        max_iterations = self.max_iterations

        while iterations < max_iterations:
            iterations += 1

            # Sample minimal set (3 points for similarity transform)
            if n < 3:
                break

            sample_indices = random.sample(range(n), 3)
            sample_src = img_points[sample_indices]
            sample_dst = osm_points[sample_indices]

            # Estimate transformation
            T, scale, rotation, translation = self.estimate_similarity_transform(
                sample_src, sample_dst
            )

            # Find inliers
            inliers = self.find_inliers(img_points, osm_points, T)
            num_inliers = np.sum(inliers)

            if num_inliers > best_num_inliers:
                best_num_inliers = num_inliers
                best_inliers = inliers
                best_T = T
                best_scale = scale
                best_rotation = rotation
                best_translation = translation

                # Update max iterations based on inlier ratio
                inlier_ratio = num_inliers / n
                if inlier_ratio > 0:
                    # p = probability of selecting all inliers
                    # k = number of iterations for confidence level
                    p = inlier_ratio**3  # 3 points sampled
                    if p < 1:
                        k = np.log(1 - self.confidence) / np.log(1 - p)
                        max_iterations = min(self.max_iterations, int(k) + 1)

        if best_num_inliers < self.min_inliers:
            return None

        # Refine using all inliers
        inlier_indices = np.where(best_inliers)[0]
        inlier_src = img_points[inlier_indices]
        inlier_dst = osm_points[inlier_indices]

        refined_T, refined_scale, refined_rotation, refined_translation = (
            self.estimate_similarity_transform(inlier_src, inlier_dst)
        )

        # Compute final residuals
        residuals = self.compute_residuals(inlier_src, inlier_dst, refined_T)
        rmse = np.sqrt(np.mean(residuals**2))

        # Build result
        inlier_correspondences = [
            (candidates[i][0], candidates[i][1]) for i in inlier_indices
        ]
        outlier_indices = np.where(~best_inliers)[0]
        outlier_correspondences = [
            (candidates[i][0], candidates[i][1]) for i in outlier_indices
        ]

        return RANSACResult(
            inlier_correspondences=inlier_correspondences,
            outlier_correspondences=outlier_correspondences,
            transformation_matrix=refined_T,
            scale=refined_scale,
            rotation_deg=refined_rotation,
            translation=refined_translation,
            inlier_ratio=best_num_inliers / n,
            rmse=rmse,
            num_iterations=iterations,
        )

    def iterative_ransac(
        self,
        image_graph: nx.Graph,
        osm_graph: nx.Graph,
        candidates: List[Tuple[int, int, float]],
        max_rounds: int = 3,
    ) -> Optional[RANSACResult]:
        """Run RANSAC iteratively, refining correspondences each round.

        Args:
            image_graph: Graph from detected roads.
            osm_graph: Reference OSM graph.
            candidates: Initial candidate correspondences.
            max_rounds: Maximum refinement rounds.

        Returns:
            Best RANSACResult after all rounds.
        """
        current_candidates = candidates
        best_result = None

        for round_num in range(max_rounds):
            result = self.ransac(image_graph, osm_graph, current_candidates)

            if result is None:
                break

            if best_result is None or result.inlier_ratio > best_result.inlier_ratio:
                best_result = result

            # If we have enough inliers, try to find more correspondences
            if result.inlier_ratio > 0.5:
                break

            # Keep only inliers for next round
            inlier_set = set(result.inlier_correspondences)
            current_candidates = [
                c for c in current_candidates if (c[0], c[1]) in inlier_set
            ]

            if len(current_candidates) < self.min_inliers:
                break

        return best_result
