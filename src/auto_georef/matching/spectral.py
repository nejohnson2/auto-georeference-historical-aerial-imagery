"""Spectral graph matching for road network alignment."""

import numpy as np
import networkx as nx
from typing import List, Tuple, Optional
from scipy.optimize import linear_sum_assignment
from scipy.linalg import eigh

from .features import FeatureExtractor


class SpectralMatcher:
    """Match graphs using spectral methods."""

    def __init__(self, embedding_dim: int = 10):
        """Initialize the spectral matcher.

        Args:
            embedding_dim: Dimension for spectral embedding.
        """
        self.embedding_dim = embedding_dim
        self.feature_extractor = FeatureExtractor()

    def compute_laplacian_spectrum(
        self, G: nx.Graph, k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute eigenvalues and eigenvectors of normalized Laplacian.

        Args:
            G: Graph to analyze.
            k: Number of eigenvalues/vectors to compute.

        Returns:
            Tuple of (eigenvalues, eigenvectors).
        """
        if G.number_of_nodes() == 0:
            return np.array([]), np.array([])

        L = nx.normalized_laplacian_matrix(G).toarray()
        eigenvalues, eigenvectors = eigh(L)

        # Return k smallest (most informative for structure)
        k = min(k, len(eigenvalues))
        return eigenvalues[:k], eigenvectors[:, :k]

    def spectral_distance(self, G1: nx.Graph, G2: nx.Graph, k: int = 10) -> float:
        """Compute spectral distance between two graphs.

        Uses difference in Laplacian eigenvalue spectra.

        Args:
            G1, G2: Graphs to compare.
            k: Number of eigenvalues to use.

        Returns:
            Spectral distance (lower = more similar structure).
        """
        evals1, _ = self.compute_laplacian_spectrum(G1, k)
        evals2, _ = self.compute_laplacian_spectrum(G2, k)

        # Pad to same length
        max_len = max(len(evals1), len(evals2))
        evals1_padded = np.pad(evals1, (0, max_len - len(evals1)))
        evals2_padded = np.pad(evals2, (0, max_len - len(evals2)))

        return np.linalg.norm(evals1_padded - evals2_padded)

    def compute_node_embedding(self, G: nx.Graph) -> Optional[np.ndarray]:
        """Compute spectral embedding for all nodes.

        Args:
            G: Graph to embed.

        Returns:
            Embedding matrix (num_nodes x embedding_dim) or None if graph too small.
        """
        if G.number_of_nodes() < self.embedding_dim + 1:
            return None

        _, eigenvectors = self.compute_laplacian_spectrum(G, self.embedding_dim + 1)

        # Skip first eigenvector (constant), use next embedding_dim
        if eigenvectors.shape[1] <= self.embedding_dim:
            return eigenvectors[:, 1:]

        return eigenvectors[:, 1 : self.embedding_dim + 1]

    def match_by_embedding(
        self,
        image_graph: nx.Graph,
        osm_graph: nx.Graph,
        degree_must_match: bool = True,
    ) -> List[Tuple[int, int, float]]:
        """Match nodes using spectral embedding distance.

        Uses Hungarian algorithm to find optimal assignment.

        Args:
            image_graph: Graph from detected roads.
            osm_graph: Reference OSM graph.
            degree_must_match: If True, only match nodes with same degree.

        Returns:
            List of (image_node, osm_node, distance) correspondences.
        """
        # Compute embeddings
        img_embedding = self.compute_node_embedding(image_graph)
        osm_embedding = self.compute_node_embedding(osm_graph)

        if img_embedding is None or osm_embedding is None:
            return []

        img_nodes = list(image_graph.nodes())
        osm_nodes = list(osm_graph.nodes())

        # Build cost matrix
        n_img = len(img_nodes)
        n_osm = len(osm_nodes)

        # Use large cost (infinity) for impossible matches
        INF = 1e10
        cost_matrix = np.full((n_img, n_osm), INF)

        for i, img_node in enumerate(img_nodes):
            img_degree = image_graph.degree(img_node)

            for j, osm_node in enumerate(osm_nodes):
                osm_degree = osm_graph.degree(osm_node)

                # Check degree compatibility
                if degree_must_match and img_degree != osm_degree:
                    continue

                # Compute embedding distance
                dist = np.linalg.norm(img_embedding[i] - osm_embedding[j])
                cost_matrix[i, j] = dist

        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Extract valid correspondences
        correspondences = []
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < INF:
                correspondences.append((img_nodes[i], osm_nodes[j], cost_matrix[i, j]))

        return correspondences

    def match_with_scale_rotation(
        self,
        image_graph: nx.Graph,
        osm_graph: nx.Graph,
        osm_projection_info: dict,
        scale_range: Tuple[float, float] = (0.5, 2.0),
        scale_steps: int = 10,
        rotation_range: Tuple[float, float] = (-30, 30),
        rotation_steps: int = 12,
    ) -> List[dict]:
        """Try matching at multiple scale and rotation hypotheses.

        Args:
            image_graph: Graph from detected roads (pixel coordinates).
            osm_graph: Reference OSM graph (meter coordinates).
            osm_projection_info: Projection info from OSM graph builder.
            scale_range: (min_scale, max_scale) in meters per pixel.
            scale_steps: Number of scale values to try.
            rotation_range: (min_rotation, max_rotation) in degrees.
            rotation_steps: Number of rotation values to try.

        Returns:
            List of match hypotheses sorted by quality.
        """
        hypotheses = []

        scales = np.linspace(scale_range[0], scale_range[1], scale_steps)
        rotations = np.linspace(rotation_range[0], rotation_range[1], rotation_steps)

        for scale in scales:
            for rotation in rotations:
                # Transform image graph coordinates
                transformed_graph = self._transform_image_graph(
                    image_graph, scale, rotation
                )

                # Try matching
                correspondences = self.match_by_embedding(
                    transformed_graph, osm_graph, degree_must_match=True
                )

                if len(correspondences) >= 3:
                    # Compute match quality
                    avg_distance = np.mean([c[2] for c in correspondences])
                    quality = len(correspondences) / (1 + avg_distance)

                    hypotheses.append(
                        {
                            "scale": scale,
                            "rotation": rotation,
                            "correspondences": correspondences,
                            "num_matches": len(correspondences),
                            "avg_distance": avg_distance,
                            "quality": quality,
                        }
                    )

        # Sort by quality (higher is better)
        hypotheses.sort(key=lambda h: h["quality"], reverse=True)

        return hypotheses

    def _transform_image_graph(
        self, G: nx.Graph, scale: float, rotation_deg: float
    ) -> nx.Graph:
        """Apply scale and rotation to image graph coordinates.

        Args:
            G: Image graph with pixel coordinates.
            scale: Scale factor (meters per pixel).
            rotation_deg: Rotation angle in degrees.

        Returns:
            Transformed graph with scaled/rotated coordinates.
        """
        G_transformed = G.copy()

        # Rotation matrix
        theta = np.radians(rotation_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

        # Compute centroid
        xs = [G.nodes[n].get("x", 0) for n in G.nodes()]
        ys = [G.nodes[n].get("y", 0) for n in G.nodes()]
        cx, cy = np.mean(xs), np.mean(ys)

        for node in G_transformed.nodes():
            x = G_transformed.nodes[node].get("x", 0)
            y = G_transformed.nodes[node].get("y", 0)

            # Center
            p = np.array([x - cx, y - cy])

            # Scale
            p = p * scale

            # Rotate
            p = R @ p

            G_transformed.nodes[node]["x"] = p[0]
            G_transformed.nodes[node]["y"] = p[1]
            G_transformed.nodes[node]["pos"] = (p[0], p[1])

        # Update edge orientations
        for u, v in G_transformed.edges():
            old_orient = G_transformed.edges[u, v].get("orientation", 0)
            new_orient = (old_orient + rotation_deg) % 180
            G_transformed.edges[u, v]["orientation"] = new_orient

            # Update length
            old_length = G_transformed.edges[u, v].get("length", 0)
            G_transformed.edges[u, v]["length"] = old_length * scale

        return G_transformed

    def refine_correspondences(
        self,
        image_graph: nx.Graph,
        osm_graph: nx.Graph,
        initial_correspondences: List[Tuple[int, int, float]],
    ) -> List[Tuple[int, int, float]]:
        """Refine correspondences using local consistency.

        Check that neighbors of matched nodes also match.

        Args:
            image_graph: Graph from detected roads.
            osm_graph: Reference OSM graph.
            initial_correspondences: Initial node correspondences.

        Returns:
            Refined correspondences with inconsistent matches removed.
        """
        # Build correspondence dictionaries
        img_to_osm = {c[0]: c[1] for c in initial_correspondences}
        osm_to_img = {c[1]: c[0] for c in initial_correspondences}

        refined = []

        for img_node, osm_node, dist in initial_correspondences:
            # Check neighbor consistency
            img_neighbors = set(image_graph.neighbors(img_node))
            osm_neighbors = set(osm_graph.neighbors(osm_node))

            # Count consistent neighbor matches
            consistent_neighbors = 0
            for img_neighbor in img_neighbors:
                if img_neighbor in img_to_osm:
                    matched_osm = img_to_osm[img_neighbor]
                    if matched_osm in osm_neighbors:
                        consistent_neighbors += 1

            # Keep if at least half of matched neighbors are consistent
            matched_neighbors = sum(1 for n in img_neighbors if n in img_to_osm)
            if matched_neighbors == 0 or consistent_neighbors >= matched_neighbors * 0.5:
                refined.append((img_node, osm_node, dist))

        return refined
