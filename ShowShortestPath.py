import cv2
import numpy as np
from skan import Skeleton, summarize
import networkx as nx
import matplotlib.pyplot as plt


class MatrixToGraphConverter:
    def __init__(self):
        self.skeleton = None
        self.graph = None
        self.df = None

    # Converts a binary skeleton matrix into a weighted NetworkX graph.
    def convert(self, matrix):
        # Create a skeleton object to analyze pixel connectivity
        self.skeleton = Skeleton(matrix)

        # Generate a summary table where each row is a path between junctions
        self.df = summarize(self.skeleton, separator='-')

        # Initialize an empty undirected graph
        self.graph = nx.Graph()

        for _, row in self.df.iterrows():
            # Identify the unique IDs for the start and end nodes of the branch
            src_idx = int(row['node-id-src'])
            dst_idx = int(row['node-id-dst'])

            # Add an edge using the actual curved path length as the weight
            self.graph.add_edge(
                src_idx,
                dst_idx,
                weight=row['branch-distance'],
                edge_id=_
            )

            # Map the node ID to its physical (y, x) coordinates in the matrix
            self.graph.nodes[src_idx]['pos'] = self.skeleton.coordinates[src_idx]
            self.graph.nodes[dst_idx]['pos'] = self.skeleton.coordinates[dst_idx]

        return self.graph

    def find_shortest_path(self, start_coords, end_coords):
        # Returns the node sequence, their coordinates, and the total path length.

        temp_graph = self.graph.copy()

        start_node = self.add_temporary_node(temp_graph, np.array(start_coords), "start")
        end_node = self.add_temporary_node(temp_graph, np.array(end_coords), "end")

        try:
            path = nx.dijkstra_path(temp_graph, start_node, end_node, weight='weight')
            path_coords = [temp_graph.nodes[n]['pos'] for n in path]
            path_length = nx.dijkstra_path_length(temp_graph, start_node, end_node, weight='weight')

            return path, path_coords, path_length

        except nx.NetworkXNoPath:
            # Handle cases where points are on disconnected components of the graph
            print("No path found between the points")
            return None, None, None

    def add_temporary_node(self, graph, p_coords, label):
        """
        Identifies the closest edge to a given coordinate and splits it into two segments.
        Distributes the original curved weight proportionally based on Euclidean distance.
        """

        best_edge = None
        min_dist = float('inf')
        new_node_id = f"temp_{label}"

        # find the nearest edge to the selected point
        for u, v, data in graph.edges(data=True):
            pos_u = graph.nodes[u]['pos']
            pos_v = graph.nodes[v]['pos']

            # Calculate the perpendicular distance from the point to the current edge
            dist = self.point_to_line_dist(p_coords, pos_u, pos_v)
            if dist < min_dist:
                min_dist = dist
                best_edge = (u, v, data['weight'])

        if best_edge:
            u, v, original_curved_weight = best_edge
            pos_u = graph.nodes[u]['pos']
            pos_v = graph.nodes[v]['pos']

            # Calculate straight-line distances to endpoints to determine the split ratio
            dist_to_u_lin = np.linalg.norm(p_coords - pos_u)
            dist_to_v_lin = np.linalg.norm(p_coords - pos_v)
            total_lin = dist_to_u_lin + dist_to_v_lin

            # Determine the proportional split of the original curved road weight
            # This ensures the new segments maintain the road's curvature characteristics
            ratio = dist_to_u_lin / total_lin if total_lin > 0 else 0.5
            weight_u = ratio * original_curved_weight
            weight_v = (1 - ratio) * original_curved_weight

            # Remove the original edge and insert the new node with its two connected segments
            graph.remove_edge(u, v)
            graph.add_node(new_node_id, pos=p_coords)
            graph.add_edge(u, new_node_id, weight=weight_u)
            graph.add_edge(new_node_id, v, weight=weight_v)

        return new_node_id

    def point_to_line_dist(self, p, a, b):
        if np.array_equal(a, b): return np.linalg.norm(p - a)
        l2 = np.sum((a - b) ** 2)
        t = max(0, min(1, np.dot(p - a, b - a) / l2))
        projection = a + t * (b - a)
        return np.linalg.norm(p - projection)

    def visualize_result(self, image, path_coords):
        """
        Renders a clean visualization showing only the image and the shortest path.
        """
        if path_coords is None:
            return

        plt.figure(figsize=(10, 10))
        plt.imshow(image, cmap='gray')

        # Convert coordinates for plotting
        path_coords = np.array(path_coords)

        # Draw the path
        plt.plot(path_coords[:, 1], path_coords[:, 0], color='red', linewidth=2.5)

        # Mark the start and end points
        plt.scatter([path_coords[0, 1], path_coords[-1, 1]], [path_coords[0, 0], path_coords[-1, 0]],
                    c='yellow', s=80, edgecolors='black', zorder=5)

        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    # Load the image in grayscale
    img = cv2.imread('road.png', 0)

    # Ensure the image is binary (0 and 1) for skan library
    # The image created with cv2.line has values of 0 and 255
    binary_road = (img > 0).astype(np.uint8)

    # Initialize the converter
    converter = MatrixToGraphConverter()

    # Build the graph from the road skeleton
    converter.convert(binary_road)

    # Define two points on the edges (y, x)
    # Point A: Midpoint of the top horizontal line (y=50, x=100)
    p1 = (50, 100)

    # Point B: Middle of the diagonal line (y=170, x=170)
    p2 = (170, 170)

    # Calculate the shortest path using Dijkstra
    path_nodes, path_coords, total_dist = converter.find_shortest_path(p1, p2)

    if path_coords:
        # Round the distance for cleaner output
        dist_rounded = round(total_dist, 2)
        print(f"Shortest path found! Total distance: {dist_rounded} pixels.")

        # Show the result on top of the original image
        converter.visualize_result(img, path_coords)
    else:
        print("Could not find a path between the selected points.")