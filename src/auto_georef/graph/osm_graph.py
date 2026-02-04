"""Fetch and process OpenStreetMap road network data."""

import logging
import numpy as np
import networkx as nx
from pathlib import Path
from typing import Optional, Tuple
import hashlib
import json

from ..config import OSMConfig

logger = logging.getLogger(__name__)


class OSMGraphBuilder:
    """Fetch road network from OpenStreetMap and build a graph."""

    # Conversion factors
    MILES_TO_METERS = 1609.34

    def __init__(self, config: Optional[OSMConfig] = None):
        """Initialize the OSM graph builder.

        Args:
            config: OSM configuration. Uses defaults if not provided.
        """
        self.config = config or OSMConfig()
        self._cache: dict = {}

    def _get_cache_key(self, lat: float, lon: float, radius_miles: float) -> str:
        """Generate a cache key for a location query."""
        # Round to reduce cache misses for nearby queries
        lat_rounded = round(lat, 4)
        lon_rounded = round(lon, 4)
        radius_rounded = round(radius_miles, 2)
        key_str = f"{lat_rounded}_{lon_rounded}_{radius_rounded}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def fetch_road_network(
        self, lat: float, lon: float, radius_miles: Optional[float] = None
    ) -> nx.Graph:
        """Fetch road network from OSM within a radius of a point.

        Args:
            lat: Latitude of center point.
            lon: Longitude of center point.
            radius_miles: Search radius in miles. Uses config default if not provided.

        Returns:
            NetworkX Graph of the road network.
        """
        import osmnx as ox

        radius_miles = radius_miles or self.config.search_radius_miles
        radius_meters = radius_miles * self.MILES_TO_METERS

        # Check cache
        cache_key = self._get_cache_key(lat, lon, radius_miles)
        if self.config.use_cache and cache_key in self._cache:
            return self._cache[cache_key].copy()

        # Fetch from OSM
        try:
            G = ox.graph_from_point(
                center_point=(lat, lon),
                dist=radius_meters,
                network_type=self.config.network_type,
                simplify=True,
            )
        except Exception as e:
            # Return empty graph on failure
            logger.warning(f"Failed to fetch OSM data: {e}")
            return nx.Graph()

        # Convert to undirected graph
        G_undirected = self._convert_to_undirected(G)

        # Filter road types if specified
        G_filtered = self._filter_road_types(G_undirected)

        # Add computed features
        G_featured = self._add_features(G_filtered)

        # Cache result
        if self.config.use_cache:
            self._cache[cache_key] = G_featured.copy()

        return G_featured

    def _convert_to_undirected(self, G: nx.MultiDiGraph) -> nx.Graph:
        """Convert OSM MultiDiGraph to simple undirected Graph.

        Args:
            G: OSM MultiDiGraph from osmnx.

        Returns:
            Simple undirected Graph.
        """
        # Create undirected graph
        G_undirected = nx.Graph()

        # Copy nodes with their attributes
        for node, data in G.nodes(data=True):
            G_undirected.add_node(
                node,
                y=data.get("y", 0),
                x=data.get("x", 0),
                pos=(data.get("x", 0), data.get("y", 0)),
            )

        # Copy edges, keeping only one per node pair
        seen_edges = set()
        for u, v, data in G.edges(data=True):
            edge_key = tuple(sorted([u, v]))
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)

                # Get edge attributes
                length = data.get("length", 0)
                highway = data.get("highway", "unclassified")
                name = data.get("name", "")

                # Handle list attributes (osmnx sometimes returns lists)
                if isinstance(highway, list):
                    highway = highway[0] if highway else "unclassified"
                if isinstance(name, list):
                    name = name[0] if name else ""

                G_undirected.add_edge(
                    u,
                    v,
                    length=length,
                    highway=highway,
                    name=name,
                )

        return G_undirected

    def _filter_road_types(self, G: nx.Graph) -> nx.Graph:
        """Filter graph to include only specified road types.

        Args:
            G: Road network graph.

        Returns:
            Filtered graph.
        """
        if not self.config.road_types:
            return G

        edges_to_remove = []
        for u, v, data in G.edges(data=True):
            highway = data.get("highway", "unclassified")
            if highway not in self.config.road_types:
                edges_to_remove.append((u, v))

        G_filtered = G.copy()
        G_filtered.remove_edges_from(edges_to_remove)

        # Remove isolated nodes
        isolated = list(nx.isolates(G_filtered))
        G_filtered.remove_nodes_from(isolated)

        return G_filtered

    def _add_features(self, G: nx.Graph) -> nx.Graph:
        """Add computed features to nodes and edges.

        Args:
            G: Road network graph.

        Returns:
            Graph with additional features.
        """
        # Add node degrees
        for node in G.nodes():
            G.nodes[node]["degree"] = G.degree(node)
            G.nodes[node]["is_junction"] = G.degree(node) > 2

        # Add edge orientations
        for u, v in G.edges():
            u_data = G.nodes[u]
            v_data = G.nodes[v]

            dx = v_data["x"] - u_data["x"]
            dy = v_data["y"] - u_data["y"]

            orientation = np.degrees(np.arctan2(dy, dx))
            if orientation < 0:
                orientation += 180
            if orientation >= 180:
                orientation -= 180

            G.edges[u, v]["orientation"] = orientation

        return G

    def project_to_local_crs(
        self, G: nx.Graph, center_lat: float, center_lon: float
    ) -> Tuple[nx.Graph, dict]:
        """Project graph coordinates to local meter-based CRS.

        Uses a simple equirectangular projection centered on the given point.
        This is accurate enough for small areas (< 10 km).

        Args:
            G: Graph with lat/lon coordinates.
            center_lat: Latitude of projection center.
            center_lon: Longitude of projection center.

        Returns:
            Tuple of (projected graph, projection info dict).
        """
        # Meters per degree at this latitude
        meters_per_deg_lat = 111320  # Approximately constant
        meters_per_deg_lon = 111320 * np.cos(np.radians(center_lat))

        G_projected = G.copy()

        for node in G_projected.nodes():
            lat = G_projected.nodes[node]["y"]
            lon = G_projected.nodes[node]["x"]

            # Project to meters relative to center
            x_meters = (lon - center_lon) * meters_per_deg_lon
            y_meters = (lat - center_lat) * meters_per_deg_lat

            G_projected.nodes[node]["x_meters"] = x_meters
            G_projected.nodes[node]["y_meters"] = y_meters
            G_projected.nodes[node]["pos_meters"] = (x_meters, y_meters)

        projection_info = {
            "center_lat": center_lat,
            "center_lon": center_lon,
            "meters_per_deg_lat": meters_per_deg_lat,
            "meters_per_deg_lon": meters_per_deg_lon,
        }

        return G_projected, projection_info

    def get_graph_bounds(self, G: nx.Graph) -> Tuple[float, float, float, float]:
        """Get the geographic bounds of a graph.

        Args:
            G: Road network graph.

        Returns:
            Tuple of (min_lon, min_lat, max_lon, max_lat).
        """
        if G.number_of_nodes() == 0:
            return (0, 0, 0, 0)

        lons = [G.nodes[n]["x"] for n in G.nodes()]
        lats = [G.nodes[n]["y"] for n in G.nodes()]

        return (min(lons), min(lats), max(lons), max(lats))

    def estimate_historical_relevance(self, G: nx.Graph, year: int) -> nx.Graph:
        """Assign relevance weights based on estimated road age.

        Major roads are more likely to have existed in historical imagery.
        This is a heuristic that helps prioritize matching on stable roads.

        Args:
            G: Road network graph.
            year: Year of the historical image.

        Returns:
            Graph with 'historical_weight' edge attribute.
        """
        # Road type weights (higher = more likely to exist historically)
        road_weights = {
            "motorway": 0.3 if year < 1960 else 0.9,  # Interstate system post-1956
            "trunk": 0.8,
            "primary": 0.9,
            "secondary": 0.8,
            "tertiary": 0.6,
            "residential": 0.4 if year < 1950 else 0.7,
            "unclassified": 0.5,
        }

        G_weighted = G.copy()

        for u, v in G_weighted.edges():
            highway = G_weighted.edges[u, v].get("highway", "unclassified")
            weight = road_weights.get(highway, 0.5)
            G_weighted.edges[u, v]["historical_weight"] = weight

        return G_weighted

    def compute_graph_stats(self, G: nx.Graph) -> dict:
        """Compute statistics about the OSM road network.

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
                "total_road_length_m": 0,
                "bounds": (0, 0, 0, 0),
            }

        num_junctions = sum(
            1 for n in G.nodes() if G.nodes[n].get("is_junction", False)
        )

        edge_lengths = [G.edges[e].get("length", 0) for e in G.edges()]
        total_length = sum(edge_lengths)

        return {
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "num_junctions": num_junctions,
            "total_road_length_m": total_length,
            "bounds": self.get_graph_bounds(G),
        }
