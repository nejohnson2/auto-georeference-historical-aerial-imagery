"""Build road network graph from skeleton image."""

import numpy as np
import networkx as nx
from dataclasses import dataclass
from typing import List, Tuple, Set, Optional
from collections import deque
from sklearn.cluster import DBSCAN


@dataclass
class RoadSegment:
    """A road segment connecting two nodes."""

    start_node: int
    end_node: int
    path_points: List[Tuple[int, int]]  # (y, x) pixel coordinates
    length_px: float
    orientation_deg: float  # Angle from horizontal


class ImageGraphBuilder:
    """Build a NetworkX graph from a skeletonized road image."""

    def __init__(self, junction_cluster_distance: int = 3):
        """Initialize the graph builder.

        Args:
            junction_cluster_distance: Maximum pixel distance to cluster junctions.
        """
        self.junction_cluster_distance = junction_cluster_distance

    def find_neighbor_count(self, skeleton: np.ndarray) -> np.ndarray:
        """Count 8-connected neighbors for each skeleton pixel.

        Args:
            skeleton: Binary skeleton image.

        Returns:
            Array with neighbor counts for each pixel.
        """
        binary = (skeleton > 0).astype(np.uint8)

        # Count neighbors using convolution with 3x3 kernel
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)

        from scipy.ndimage import convolve

        neighbor_count = convolve(binary, kernel, mode="constant", cval=0)
        neighbor_count = neighbor_count * binary  # Only count for skeleton pixels

        return neighbor_count

    def find_junctions(self, skeleton: np.ndarray) -> List[Tuple[int, int]]:
        """Find junction pixels (more than 2 neighbors).

        Args:
            skeleton: Binary skeleton image.

        Returns:
            List of (y, x) coordinates of junction pixels.
        """
        neighbor_count = self.find_neighbor_count(skeleton)
        junction_mask = (neighbor_count > 2) & (skeleton > 0)
        junctions = list(zip(*np.where(junction_mask)))
        return junctions

    def find_endpoints(self, skeleton: np.ndarray) -> List[Tuple[int, int]]:
        """Find endpoint pixels (exactly 1 neighbor).

        Args:
            skeleton: Binary skeleton image.

        Returns:
            List of (y, x) coordinates of endpoint pixels.
        """
        neighbor_count = self.find_neighbor_count(skeleton)
        endpoint_mask = (neighbor_count == 1) & (skeleton > 0)
        endpoints = list(zip(*np.where(endpoint_mask)))
        return endpoints

    def cluster_junctions(
        self, junctions: List[Tuple[int, int]]
    ) -> List[Tuple[float, float]]:
        """Cluster nearby junctions and return centroids.

        Historical images may have thick intersections that appear as multiple
        junction pixels. This clusters them into single nodes.

        Args:
            junctions: List of junction (y, x) coordinates.

        Returns:
            List of clustered junction centroids.
        """
        if not junctions:
            return []

        if len(junctions) == 1:
            return [junctions[0]]

        coords = np.array(junctions)

        # DBSCAN clustering
        clustering = DBSCAN(
            eps=self.junction_cluster_distance, min_samples=1, metric="euclidean"
        ).fit(coords)

        # Compute centroids for each cluster
        centroids = []
        for label in set(clustering.labels_):
            cluster_points = coords[clustering.labels_ == label]
            centroid = tuple(np.mean(cluster_points, axis=0))
            centroids.append(centroid)

        return centroids

    def get_neighbors(
        self, skeleton: np.ndarray, y: int, x: int
    ) -> List[Tuple[int, int]]:
        """Get 8-connected skeleton neighbors of a pixel.

        Args:
            skeleton: Binary skeleton image.
            y, x: Pixel coordinates.

        Returns:
            List of neighbor (y, x) coordinates.
        """
        neighbors = []
        h, w = skeleton.shape

        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx] > 0:
                    neighbors.append((ny, nx))

        return neighbors

    def trace_road_segment(
        self,
        skeleton: np.ndarray,
        start: Tuple[int, int],
        node_positions: Set[Tuple[int, int]],
    ) -> Optional[RoadSegment]:
        """Trace a road segment from start until reaching another node.

        Args:
            skeleton: Binary skeleton image.
            start: Starting (y, x) position.
            node_positions: Set of all node positions (junctions and endpoints).

        Returns:
            RoadSegment if a valid segment is found, None otherwise.
        """
        path = [start]
        visited = {start}
        current = start

        while True:
            neighbors = self.get_neighbors(skeleton, current[0], current[1])
            unvisited = [n for n in neighbors if n not in visited]

            if not unvisited:
                # Dead end
                break

            # Move to next pixel
            next_pixel = unvisited[0]
            visited.add(next_pixel)
            path.append(next_pixel)

            # Check if we've reached another node
            # Use a small tolerance for floating point centroids
            reached_node = False
            for node_pos in node_positions:
                if node_pos == start:
                    continue
                dist = np.sqrt(
                    (next_pixel[0] - node_pos[0]) ** 2
                    + (next_pixel[1] - node_pos[1]) ** 2
                )
                if dist < 2:  # Within 2 pixels of a node
                    reached_node = True
                    break

            if reached_node or len(unvisited) > 1:
                # Reached a node or a junction
                break

            current = next_pixel

        if len(path) < 2:
            return None

        # Compute segment properties
        length_px = self._compute_path_length(path)
        orientation_deg = self._compute_orientation(path[0], path[-1])

        return RoadSegment(
            start_node=-1,  # Will be assigned later
            end_node=-1,
            path_points=path,
            length_px=length_px,
            orientation_deg=orientation_deg,
        )

    def _compute_path_length(self, path: List[Tuple[int, int]]) -> float:
        """Compute total length of a path in pixels."""
        length = 0.0
        for i in range(len(path) - 1):
            dy = path[i + 1][0] - path[i][0]
            dx = path[i + 1][1] - path[i][1]
            length += np.sqrt(dy * dy + dx * dx)
        return length

    def _compute_orientation(
        self, start: Tuple[int, int], end: Tuple[int, int]
    ) -> float:
        """Compute orientation angle of a line segment in degrees [0, 180)."""
        dy = end[0] - start[0]
        dx = end[1] - start[1]
        angle = np.degrees(np.arctan2(dy, dx))
        # Normalize to [0, 180)
        if angle < 0:
            angle += 180
        if angle >= 180:
            angle -= 180
        return angle

    def _find_closest_node(
        self, point: Tuple[int, int], nodes: List[Tuple[float, float]]
    ) -> int:
        """Find the index of the closest node to a point."""
        min_dist = float("inf")
        closest_idx = -1

        for i, node in enumerate(nodes):
            dist = np.sqrt((point[0] - node[0]) ** 2 + (point[1] - node[1]) ** 2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        return closest_idx

    def build_graph(self, skeleton: np.ndarray) -> nx.Graph:
        """Build a NetworkX graph from the skeleton image.

        Args:
            skeleton: Binary skeleton image (single-pixel width roads).

        Returns:
            NetworkX Graph with:
            - Nodes: Junction/endpoint positions with (x, y) coordinates
            - Edges: Road segments with length, orientation, and path attributes
        """
        # Find junctions and endpoints
        raw_junctions = self.find_junctions(skeleton)
        endpoints = self.find_endpoints(skeleton)

        # Cluster nearby junctions
        clustered_junctions = self.cluster_junctions(raw_junctions)

        # Combine all nodes (junctions first, then endpoints)
        all_nodes = list(clustered_junctions) + [(float(e[0]), float(e[1])) for e in endpoints]

        if not all_nodes:
            # No nodes found - return empty graph
            return nx.Graph()

        # Create set of node positions for tracing
        node_positions = set()
        for node in all_nodes:
            # Add integer-rounded positions
            node_positions.add((int(round(node[0])), int(round(node[1]))))
            # Add nearby pixels to catch traced paths
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    node_positions.add(
                        (int(round(node[0])) + dy, int(round(node[1])) + dx)
                    )

        # Create graph
        G = nx.Graph()

        # Add nodes with coordinates
        for i, (y, x) in enumerate(all_nodes):
            is_junction = i < len(clustered_junctions)
            G.add_node(
                i,
                y=y,
                x=x,
                pos=(x, y),  # (x, y) for visualization
                is_junction=is_junction,
            )

        # Trace segments from each junction/endpoint
        visited_edges = set()

        for node_idx, (node_y, node_x) in enumerate(all_nodes):
            # Get starting pixel (rounded to integer)
            start_y, start_x = int(round(node_y)), int(round(node_x))

            # Find skeleton pixels near this node
            h, w = skeleton.shape
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    py, px = start_y + dy, start_x + dx
                    if 0 <= py < h and 0 <= px < w and skeleton[py, px] > 0:
                        # Try tracing from this pixel
                        segment = self.trace_road_segment(
                            skeleton, (py, px), node_positions
                        )

                        if segment and len(segment.path_points) >= 2:
                            # Find end node
                            end_point = segment.path_points[-1]
                            end_node_idx = self._find_closest_node(end_point, all_nodes)

                            if end_node_idx >= 0 and end_node_idx != node_idx:
                                # Create edge key (sorted to avoid duplicates)
                                edge_key = tuple(sorted([node_idx, end_node_idx]))

                                if edge_key not in visited_edges:
                                    visited_edges.add(edge_key)

                                    G.add_edge(
                                        node_idx,
                                        end_node_idx,
                                        length=segment.length_px,
                                        orientation=segment.orientation_deg,
                                        path=segment.path_points,
                                    )

        # Compute node degrees
        for node in G.nodes():
            G.nodes[node]["degree"] = G.degree(node)

        return G

    def compute_graph_stats(self, G: nx.Graph) -> dict:
        """Compute statistics about the road network graph.

        Args:
            G: Road network graph.

        Returns:
            Dictionary of graph statistics.
        """
        if G.number_of_nodes() == 0:
            return {
                "num_nodes": 0,
                "num_edges": 0,
                "num_junctions": 0,
                "num_endpoints": 0,
                "total_road_length_px": 0,
                "avg_edge_length_px": 0,
            }

        num_junctions = sum(1 for n in G.nodes() if G.nodes[n].get("is_junction", False))
        num_endpoints = G.number_of_nodes() - num_junctions

        edge_lengths = [G.edges[e].get("length", 0) for e in G.edges()]
        total_length = sum(edge_lengths)
        avg_length = total_length / len(edge_lengths) if edge_lengths else 0

        return {
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "num_junctions": num_junctions,
            "num_endpoints": num_endpoints,
            "total_road_length_px": total_length,
            "avg_edge_length_px": avg_length,
        }
