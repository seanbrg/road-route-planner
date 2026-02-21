import numpy as np
from skan import Skeleton, summarize
import networkx as nx
import cv2


class ShortestPathSolver:
    def __init__(self):
        self.skeleton = None
        self.graph = None
        self.df = None

    def convert(self, matrix):
        """
        Translates the skeleton matrix into a weighted network graph.
        """
        self.skeleton = Skeleton(matrix)
        self.df = summarize(self.skeleton, separator='-')
        self.graph = nx.Graph()

        for _, row in self.df.iterrows():
            src_idx = int(row['node-id-src'])
            dst_idx = int(row['node-id-dst'])

            # Use real branch distance to account for road curves
            self.graph.add_edge(
                src_idx,
                dst_idx,
                weight=row['branch-distance']
            )

            self.graph.nodes[src_idx]['pos'] = self.skeleton.coordinates[src_idx]
            self.graph.nodes[dst_idx]['pos'] = self.skeleton.coordinates[dst_idx]

        return self.graph

    def block_road(self, barrier_coords):
        """
        Finds the edge closest to the barrier point and removes it from the graph.
        """
        best_edge = None
        min_dist = float('inf')
        p_coords = np.array(barrier_coords)

        # Iterate through edges to identify which road segment contains the barrier
        for u, v, data in self.graph.edges(data=True):
            pos_u = self.graph.nodes[u]['pos']
            pos_v = self.graph.nodes[v]['pos']
            dist = self.point_to_line_dist(p_coords, pos_u, pos_v)

            if dist < min_dist:
                min_dist = dist
                best_edge = (u, v)

        # Remove the blocked edge from the graph so Dijkstra cannot use it
        if best_edge:
            self.graph.remove_edge(*best_edge)

    def find_shortest_path(self, start_coords, end_coords):
        """
        Calculates the path between points using a temporary copy of the graph.
        """
        temp_graph = self.graph.copy()

        # Inject start and end points into the topology
        start_node = self.add_temporary_node(temp_graph, np.array(start_coords), "start")
        end_node = self.add_temporary_node(temp_graph, np.array(end_coords), "end")

        try:
            path = nx.dijkstra_path(temp_graph, start_node, end_node, weight='weight')
            path_coords = [temp_graph.nodes[n]['pos'] for n in path]
            return path_coords
        except nx.NetworkXNoPath:
            return None

    def add_temporary_node(self, graph, p_coords, label):
        # Implementation from previous steps to split edges for injection
        best_edge = None
        min_dist = float('inf')
        new_node_id = f"temp_{label}"

        for u, v, data in graph.edges(data=True):
            pos_u = graph.nodes[u]['pos']
            pos_v = graph.nodes[v]['pos']
            dist = self.point_to_line_dist(p_coords, pos_u, pos_v)
            if dist < min_dist:
                min_dist = dist
                best_edge = (u, v, data['weight'])

        if best_edge:
            u, v, weight = best_edge
            pos_u = graph.nodes[u]['pos']
            pos_v = graph.nodes[v]['pos']
            dist_u = np.linalg.norm(p_coords - pos_u)
            dist_v = np.linalg.norm(p_coords - pos_v)
            total = dist_u + dist_v

            ratio = dist_u / total if total > 0 else 0.5
            graph.remove_edge(u, v)
            graph.add_node(new_node_id, pos=p_coords)
            graph.add_edge(u, new_node_id, weight=ratio * weight)
            graph.add_edge(new_node_id, v, weight=(1 - ratio) * weight)

        return new_node_id

    def point_to_line_dist(self, p, a, b):
        # Mathematical distance from point to a line segment
        if np.array_equal(a, b): return np.linalg.norm(p - a)
        l2 = np.sum((a - b) ** 2)
        t = max(0, min(1, np.dot(p - a, b - a) / l2))
        projection = a + t * (b - a)
        return np.linalg.norm(p - projection)

def get_satellite_navigation_map(satellite_img, skeleton_matrix, start, end, barriers):
    """
    Computes a path avoiding barriers and draws it on the satellite image.
    """
    # Create the solver instance and build the graph
    solver = ShortestPathSolver()
    solver.convert(skeleton_matrix)

    # Process all barriers by removing the corresponding edges from the graph
    for barrier_point in barriers:
        solver.block_road(barrier_point)

    # Calculate the shortest path coordinates
    path_coords = solver.find_shortest_path(start, end)

    # If no path exists after removing barriers, return None
    if path_coords is None:
        return None

    # Ensure the image is in BGR format for colored drawing
    if len(satellite_img.shape) == 2:
        output_image = cv2.cvtColor(satellite_img, cv2.COLOR_GRAY2BGR)
    else:
        output_image = satellite_img.copy()

    # Draw the path segments as a thick red line on the satellite image
    for i in range(len(path_coords) - 1):
        # Convert (y, x) to (x, y) for OpenCV plotting
        pt1 = (int(path_coords[i][1]), int(path_coords[i][0]))
        pt2 = (int(path_coords[i + 1][1]), int(path_coords[i + 1][0]))
        cv2.line(output_image, pt1, pt2, (0, 0, 255), 2)

    return output_image
