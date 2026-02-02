"""Feature extraction for graph matching."""

import numpy as np
import networkx as nx
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class NodeSignature:
    """Signature describing a node's local structure."""

    node_id: int
    degree: int
    incident_angles: List[float]  # Sorted angles between incident edges
    neighbor_degrees: List[int]  # Degrees of neighboring nodes
    local_density: float  # Number of edges within 2 hops


@dataclass
class EdgeSignature:
    """Signature describing an edge's properties."""

    edge: Tuple[int, int]
    length: float
    orientation: float
    start_degree: int
    end_degree: int


class FeatureExtractor:
    """Extract discriminative features for graph matching."""

    def compute_incident_angles(self, G: nx.Graph, node: int) -> List[float]:
        """Compute angles between all pairs of incident edges at a node.

        Args:
            G: Road network graph.
            node: Node ID.

        Returns:
            Sorted list of angles between consecutive edges (in degrees).
        """
        neighbors = list(G.neighbors(node))
        if len(neighbors) < 2:
            return []

        # Get node position
        node_pos = np.array([G.nodes[node].get("x", 0), G.nodes[node].get("y", 0)])

        # Compute angles from node to each neighbor
        angles = []
        for neighbor in neighbors:
            neighbor_pos = np.array(
                [G.nodes[neighbor].get("x", 0), G.nodes[neighbor].get("y", 0)]
            )
            direction = neighbor_pos - node_pos
            angle = np.degrees(np.arctan2(direction[1], direction[0]))
            angles.append(angle)

        # Sort angles
        angles.sort()

        # Compute differences between consecutive angles
        angle_diffs = []
        for i in range(len(angles)):
            diff = angles[(i + 1) % len(angles)] - angles[i]
            if diff < 0:
                diff += 360
            angle_diffs.append(diff)

        # Sort angle differences for comparison
        angle_diffs.sort()

        return angle_diffs

    def compute_node_signature(self, G: nx.Graph, node: int) -> NodeSignature:
        """Compute a signature describing the local structure around a node.

        Args:
            G: Road network graph.
            node: Node ID.

        Returns:
            NodeSignature for the node.
        """
        degree = G.degree(node)
        incident_angles = self.compute_incident_angles(G, node)
        neighbor_degrees = sorted([G.degree(n) for n in G.neighbors(node)])

        # Compute local density (edges within 2 hops)
        local_edges = set()
        for neighbor in G.neighbors(node):
            local_edges.add(tuple(sorted([node, neighbor])))
            for n2 in G.neighbors(neighbor):
                local_edges.add(tuple(sorted([neighbor, n2])))
        local_density = len(local_edges)

        return NodeSignature(
            node_id=node,
            degree=degree,
            incident_angles=incident_angles,
            neighbor_degrees=neighbor_degrees,
            local_density=local_density,
        )

    def compute_edge_signature(self, G: nx.Graph, u: int, v: int) -> EdgeSignature:
        """Compute a signature for an edge.

        Args:
            G: Road network graph.
            u, v: Edge endpoints.

        Returns:
            EdgeSignature for the edge.
        """
        edge_data = G.edges[u, v]

        return EdgeSignature(
            edge=(u, v),
            length=edge_data.get("length", 0),
            orientation=edge_data.get("orientation", 0),
            start_degree=G.degree(u),
            end_degree=G.degree(v),
        )

    def node_signature_distance(
        self,
        sig1: NodeSignature,
        sig2: NodeSignature,
        angle_weight: float = 0.3,
        degree_weight: float = 0.4,
        density_weight: float = 0.3,
    ) -> float:
        """Compute distance between two node signatures.

        Args:
            sig1, sig2: Node signatures to compare.
            angle_weight: Weight for angle similarity.
            degree_weight: Weight for degree similarity.
            density_weight: Weight for local density similarity.

        Returns:
            Distance (lower = more similar). Returns float('inf') for incompatible nodes.
        """
        # Degree must match for a valid correspondence
        if sig1.degree != sig2.degree:
            return float("inf")

        # Compare incident angles
        angle_dist = 0.0
        if sig1.incident_angles and sig2.incident_angles:
            if len(sig1.incident_angles) == len(sig2.incident_angles):
                diffs = [
                    abs(a1 - a2)
                    for a1, a2 in zip(sig1.incident_angles, sig2.incident_angles)
                ]
                angle_dist = sum(diffs) / len(diffs) / 180.0  # Normalize by 180 degrees
            else:
                angle_dist = 1.0

        # Compare neighbor degrees
        degree_dist = 0.0
        if sig1.neighbor_degrees and sig2.neighbor_degrees:
            if len(sig1.neighbor_degrees) == len(sig2.neighbor_degrees):
                diffs = [
                    abs(d1 - d2)
                    for d1, d2 in zip(sig1.neighbor_degrees, sig2.neighbor_degrees)
                ]
                max_degree = max(max(sig1.neighbor_degrees), max(sig2.neighbor_degrees), 1)
                degree_dist = sum(diffs) / len(diffs) / max_degree
            else:
                degree_dist = 1.0

        # Compare local density
        max_density = max(sig1.local_density, sig2.local_density, 1)
        density_dist = abs(sig1.local_density - sig2.local_density) / max_density

        # Weighted combination
        total_dist = (
            angle_weight * angle_dist
            + degree_weight * degree_dist
            + density_weight * density_dist
        )

        return total_dist

    def edge_signature_distance(
        self,
        sig1: EdgeSignature,
        sig2: EdgeSignature,
        length_tolerance: float = 0.3,
        angle_tolerance: float = 15.0,
    ) -> float:
        """Compute distance between two edge signatures.

        Args:
            sig1, sig2: Edge signatures to compare.
            length_tolerance: Relative length tolerance (0.3 = 30%).
            angle_tolerance: Angle tolerance in degrees.

        Returns:
            Distance (lower = more similar). Returns float('inf') for incompatible edges.
        """
        # Check endpoint degree compatibility
        degrees1 = sorted([sig1.start_degree, sig1.end_degree])
        degrees2 = sorted([sig2.start_degree, sig2.end_degree])
        if degrees1 != degrees2:
            return float("inf")

        # Compare lengths (relative difference)
        max_len = max(sig1.length, sig2.length, 1e-6)
        length_diff = abs(sig1.length - sig2.length) / max_len
        if length_diff > length_tolerance:
            return float("inf")

        # Compare orientations (handle circular nature)
        angle_diff = abs(sig1.orientation - sig2.orientation)
        if angle_diff > 90:
            angle_diff = 180 - angle_diff  # Handle opposite directions
        angle_dist = angle_diff / angle_tolerance

        # Combine
        return 0.5 * length_diff + 0.5 * min(angle_dist, 1.0)

    def compute_all_node_signatures(self, G: nx.Graph) -> dict:
        """Compute signatures for all nodes in a graph.

        Args:
            G: Road network graph.

        Returns:
            Dictionary mapping node ID to NodeSignature.
        """
        signatures = {}
        for node in G.nodes():
            signatures[node] = self.compute_node_signature(G, node)
        return signatures

    def compute_spectral_embedding(
        self, G: nx.Graph, dim: int = 10
    ) -> Optional[np.ndarray]:
        """Compute spectral embedding of the graph.

        Uses eigenvectors of the normalized Laplacian to embed nodes
        in a low-dimensional space that captures graph structure.

        Args:
            G: Road network graph.
            dim: Embedding dimension.

        Returns:
            Node embedding matrix (num_nodes x dim) or None if graph is too small.
        """
        if G.number_of_nodes() < dim + 1:
            return None

        # Compute normalized Laplacian
        L = nx.normalized_laplacian_matrix(G).toarray()

        # Compute eigenvectors
        from scipy.linalg import eigh

        eigenvalues, eigenvectors = eigh(L)

        # Use smallest non-trivial eigenvectors (skip the constant eigenvector)
        # Start from index 1 to skip the trivial eigenvector
        embedding = eigenvectors[:, 1 : dim + 1]

        return embedding

    def find_candidate_correspondences(
        self,
        image_graph: nx.Graph,
        osm_graph: nx.Graph,
        max_candidates_per_node: int = 5,
        max_distance: float = 0.5,
        min_degree: int = 2,
    ) -> List[Tuple[int, int, float]]:
        """Find candidate node correspondences between two graphs.

        Args:
            image_graph: Graph built from detected roads.
            osm_graph: Reference OSM graph.
            max_candidates_per_node: Maximum candidates per image node.
            max_distance: Maximum signature distance for candidates.
            min_degree: Minimum node degree to consider (filters noise).

        Returns:
            List of (image_node, osm_node, distance) tuples.
        """
        # Compute signatures for both graphs
        img_sigs = self.compute_all_node_signatures(image_graph)
        osm_sigs = self.compute_all_node_signatures(osm_graph)

        candidates = []

        for img_node, img_sig in img_sigs.items():
            # Skip low-degree nodes (likely noise)
            if img_sig.degree < min_degree:
                continue

            # Find best matching OSM nodes
            node_candidates = []

            for osm_node, osm_sig in osm_sigs.items():
                # OSM nodes must also meet degree requirement
                if osm_sig.degree < min_degree:
                    continue

                dist = self.node_signature_distance(img_sig, osm_sig)
                if dist < max_distance:
                    node_candidates.append((osm_node, dist))

            # Keep top candidates
            node_candidates.sort(key=lambda x: x[1])
            for osm_node, dist in node_candidates[:max_candidates_per_node]:
                candidates.append((img_node, osm_node, dist))

        return candidates
